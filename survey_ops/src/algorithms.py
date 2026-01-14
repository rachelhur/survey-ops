
import random
import numpy as np
import torch
import torch.nn.functional as F

from survey_ops.src.neural_nets import DQN
from survey_ops.utils.interpolate import interpolate_on_sphere
import logging

logger = logging.getLogger(__name__)

class AlgorithmBase:
    def __init__(self):
        super().__init__()
        
    def train_step(self, batch):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError

    def save(self, filepath):
        torch.save({'policy_net': self.policy_net.state_dict()}, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
    
class DDQN(AlgorithmBase):
    """
    Implementation of the DDQN algorithm. Uses AdamW optimizer and, optionally, a cosine annealing lr scheduler.

    Args
    ----
    obs_dim (int): size of each observation
    num_actions (int): number of total possible actions
    hidden_dim (int): hidden dimension size in DQN network
    gamma (float): 
    tau (float): 
    device (str): 
    lr (float): Learning rate
    loss_fxn (torch.nn.functional): Loss function (ie F.huber_loss, F.mse_loss)
    use_dqn (bool): 
    optimizer_kwargs (optional): 
    """

    def __init__(self, obs_dim, num_actions, hidden_dim, gamma, tau, device, lr, loss_fxn=None, use_double=True, \
                 lr_scheduler='cosine_annealing', lr_scheduler_kwargs=None, optimizer_kwargs={}):
        super().__init__()
        self.name = 'DDQN'
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.policy_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.use_double = use_double

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=False, **optimizer_kwargs)

        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs) if lr_scheduler == 'cosine_annealing' else None
        assert loss_fxn is not None
        self.loss_fxn = loss_fxn

    def train_step(self, batch):
        state, actions, rewards, next_state, dones, action_masks = batch

        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_state = torch.as_tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        action_masks = torch.as_tensor(np.array(action_masks), device=self.device, dtype=torch.bool)
            
        # need to input (batch_size, obs_dim) into net - if obs_dim is 1, we get 1d tensor. Need to reshape
        if state.dim() == 1:
            state = state.unsqueeze(1)
            next_state = next_state.unsqueeze(1)
        
        # get policy q vals for current state
        q_vals = self.policy_net(state)
        q_vals = q_vals.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                # get policy best actions (need to mask invalid actions)
                pol_next_q = self.policy_net(next_state)
                pol_next_q[~action_masks] = float('-inf') #-1e9
                pol_next_actions = pol_next_q.argmax(1).type(torch.long)

                # evaluate policy's next actions using target net
                target_next_q = self.target_net(next_state)
                next_q_targ = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze(1)

                td_target = rewards + self.gamma * (1 - dones) * next_q_targ
            else:
                next_q = self.target_net(next_state)
                # mask invalid actions
                next_q[~action_masks] = -1e9 # float('-inf')
                max_next_q = next_q.max(dim=1)[0]
                td_target = rewards + self.gamma * max_next_q * (1 - dones) # , dtype=torch.float32, device=device

        loss = self.loss_fxn(q_vals, td_target)
        
        # optimize w/ backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Debugging nans in gradient
        # for name, param in self.policy_net.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             logger.debug(f"NaN gradient in {name}")
        # print(f"Max Gradient Norm: {prev_grad_norm.item()}")
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
                    
        self._soft_update()

        # soft update done in agent
        return loss.item(), q_vals.mean().item()

    def _soft_update(self):
        # update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, obs, action_mask, epsilon=None):
        # if random sample less than epsilon, take random action
        if epsilon is not None:
            if np.random.random() < epsilon:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
                return int(action)

        # greedy selection from policy
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs).squeeze(0)
            # mask invalid actions
            action_mask = torch.tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~action_mask] = float('-inf')
            action = torch.argmax(q_values).item()
        return int(action)
    
    def test_step(self, eval_batch):
        state, actions, rewards, next_state, dones, action_masks = eval_batch

        with torch.no_grad():      
            # convert to tensors
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
            action_masks = torch.tensor(np.array(action_masks), device=self.device, dtype=torch.bool)

            if self.use_double:
                q_vals = self.policy_net(state)
                q_vals[~action_masks] = -1e9
                predicted_actions = q_vals.argmax(1).squeeze()

                # q_max = q_vals.max(dim=1)
                # q_dataset = q_vals.gather(1, actions).squeeze(1)
                accuracy = (predicted_actions == actions).float()

                print(q_vals.shape)                
                loss = self.loss_fxn(q_vals, actions)
            else:
                raise NotImplementedError
            return loss, q_vals[action_masks].mean(), accuracy.mean()#, q_max, q_dataset)
            
                    # eval_obs = torch.as_tensor(eval_obs, device=self.device, dtype=torch.float32)
                    # expert_actions = torch.as_tensor(expert_actions, device=self.device, dtype=torch.long)
                    
                    # all_q_vals = self.algorithm.policy_net(eval_obs)
                    # if self.algorithm.name != 'BehaviorCloning':
                    #     all_q_vals[~action_masks] = float('-inf')
                    # q_vals = all_q_vals.mean().item()
                    
                    # eval_loss = self.algorithm.loss_fxn(all_q_vals, expert_actions)
                    # predicted_actions = all_q_vals.argmax(dim=1)

                    # if self.algorithm.name != 'BehaviorCloning':
                    #     all_q_vals[~action_masks] = float('-inf')
    

class BehaviorCloning(AlgorithmBase):
    def __init__(self, obs_dim, num_actions, hidden_dim, loss_fxn=None, lr=1e-3, lr_scheduler=None, lr_scheduler_kwargs=None, device='cpu'):
        self.name = 'BehaviorCloning'
        self.device = device
        self.policy_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fxn = loss_fxn
        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs) if lr_scheduler == 'cosine_annealing' else None
        
    def train_step(self, batch):
        """
        Train the policy to mimic expert actions from offline data
        """
        state, expert_actions, rewards, next_state, dones, action_masks = batch
        
        # convert to tensors and appropriate dtypes
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        if not torch.is_tensor(expert_actions):
            expert_actions = torch.as_tensor(expert_actions, dtype=torch.long) # needs to be long for .gather()
        else:
            expert_actions = expert_actions.long() # needs to be long for .gather()
        expert_actions = expert_actions.to(device=self.device)
        action_logits = self.policy_net(state)
        
        loss = self.loss_fxn(action_logits, expert_actions)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item(), action_logits.mean().item()
    
    def test_step(self, batch):
        eval_obs, expert_actions, rewards, next_obs, dones, action_masks = batch

        with torch.no_grad():      
            # convert to tensors
            eval_obs = torch.as_tensor(eval_obs, device=self.device, dtype=torch.float32)
            expert_actions = torch.as_tensor(expert_actions, device=self.device, dtype=torch.long)
            
            action_logits = self.policy_net(eval_obs)
            predicted_actions = action_logits.argmax(dim=1)
            loss = self.loss_fxn(action_logits, expert_actions)

            accuracy = (predicted_actions == expert_actions).float().mean()
            
        return loss, action_logits.mean(), accuracy
    
    def select_action(self, obs, action_mask, epsilon=None):
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32)
                mask = torch.tensor(action_mask, dtype=torch.bool)
            obs = obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(self.device, dtype=torch.bool).unsqueeze(0)
            action_logits = self.policy_net(obs)
            # mask invalid actions
            action_logits[~mask] = float('-inf')
            action = torch.argmax(action_logits, dim=1)
            return action.cpu().numpy()[0] if action.size(0) == 1 else action.cpu().numpy()

    # def predict(self, state):
    #     """
    #     Get action for a given state
    #     """
    #     self.policy_net.eval()
    #     with torch.no_grad():
    #         state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
    #         if len(state_tensor.shape) == 1:
    #             state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                
    #         action_logits = self.policy_net(state_tensor)
    #         action = torch.argmax(action_logits, dim=1)
    #         return action.cpu().numpy()[0] if action.size(0) == 1 else action.cpu().numpy()
    
    # def evaluate(self, test_dataset):
    #     """
    #     Evaluate the trained policy on test data
    #     """
    #     states = torch.tensor(np.array([d['state'] for d in test_dataset]), dtype=torch.float32)
    #     expert_actions = torch.tensor([d['expert_action'] for d in test_dataset], dtype=torch.long)
        
    #     states = states.to(self.device)
    #     expert_actions = expert_actions.to(self.device)
        
    #     with torch.no_grad():
    #         action_logits = self.policy_net(states)
    #         predicted_actions = torch.argmax(action_logits, dim=1)
    #         accuracy = (predicted_actions == expert_actions).float().mean().item()
        
    #     return accuracy