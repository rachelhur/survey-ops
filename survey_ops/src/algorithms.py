
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

    def __init__(self, obs_dim, num_actions, hidden_dim, gamma, tau, device, lr, target_update_freq=1000, loss_fxn=None, use_double=True, \
                 lr_scheduler='cosine_annealing', lr_scheduler_kwargs=None, optimizer_kwargs={}, lr_scheduler_step_freq=100):
        super().__init__()
        self.name = 'DDQN'
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.policy_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.use_double = use_double
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=False, **optimizer_kwargs)

        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs) if lr_scheduler == 'cosine_annealing' else None
        if self.lr_scheduler is not None:
            self.lr_scheduler_step_freq = lr_scheduler_step_freq
        # Tmax needs to equal number of scheduler steps at end of training
        assert loss_fxn is not None
        self.loss_fxn = loss_fxn
        self.val_metrics = ['val_loss', 'q_policy', 'q_data', 'val_accuracy']

    def train_step(self, batch, step_num):
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
        
        # Get policy's q vals for current state
        q_vals = self.policy_net(state)
        q_current = q_vals.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                # Select best action with policy net
                pol_next_q = self.policy_net(next_state)
                pol_next_q[~action_masks] = float('-inf') #-1e9
                pol_next_actions = pol_next_q.argmax(1).type(torch.long)

                # Evaluate policy's next actions using target net
                target_next_q = self.target_net(next_state)
                next_q_targ = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze()

                td_target = rewards + self.gamma * (1 - dones) * next_q_targ
            else:
                next_q = self.target_net(next_state)
                # mask invalid actions
                next_q[~action_masks] = -1e9 # float('-inf')
                max_next_q = next_q.max(dim=1)[0]
                td_target = rewards + self.gamma * max_next_q * (1 - dones) # , dtype=torch.float32, device=device

        loss = self.loss_fxn(q_current, td_target)
        
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
        if self.lr_scheduler is not None and step_num % self.lr_scheduler_step_freq == 0:
            self.lr_scheduler.step()
        
        if step_num % self.target_update_freq == 0:
            self._soft_update()

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

            q_vals = self.policy_net(state)
            q_current = q_vals.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions).squeeze()
            predicted_actions = q_vals.argmax(1)

            # Compute TD targets for loss
            if self.use_double:
                pol_next_q = self.policy_net(next_state)
                pol_next_q[~action_masks] = -1e9
                pol_next_actions = pol_next_q.argmax(1)
                
                target_next_q = self.target_net(next_state)
                next_q_vals = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze()
            else:
                # DQN
                target_next_q = self.target_net(next_state).clone()
                target_next_q[~action_masks] = float('-inf')
                next_q_vals = target_next_q.max(1)[0]
            
            # Compute TD target: r + γ * Q(s', a') * (1 - done)
            td_target = rewards + self.gamma * next_q_vals * (1 - dones)
            
            # Compute TD error/loss
            loss = self.loss_fxn(q_current, td_target)

            mean_accuracy = (predicted_actions == actions).float().mean()

            q_dataset_mean = q_vals.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions).squeeze().mean()
            q_policy_mean = q_vals.max(dim=1)[0].mean()
            

            return loss, q_policy_mean.item(), q_dataset_mean.item(), mean_accuracy.item()
            
class BehaviorCloning(AlgorithmBase):
    def __init__(self, obs_dim, num_actions, hidden_dim, loss_fxn=None, lr=1e-3, lr_scheduler=None, lr_scheduler_kwargs=None, \
                 lr_scheduler_step_freq=10, lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=5, device='cpu'):
        self.name = 'BehaviorCloning'
        self.device = device
        self.policy_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        assert loss_fxn is not None, "loss_fxn needs to be passed"
        self.loss_fxn = loss_fxn
        
        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs) if lr_scheduler == 'cosine_annealing' else None

        if lr_scheduler is not None:
            logger.debug(f'BC lr_scheduler is {self.lr_scheduler}')
            self.lr_scheduler_step_freq = lr_scheduler_step_freq
            self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
            self.lr_scheduler_num_epochs = lr_scheduler_num_epochs

        self.val_metrics = ['val_loss', 'logp_expert_action', 'action_margin', 'entropy','accuracy']
        
    def train_step(self, batch, epoch_num, step_num):
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
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and step_num % self.lr_scheduler_step_freq == 0
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs
        )
        if do_lr_scheduler_step:
        # if self.lr_scheduler is not None and step_num % self.lr_scheduler_step_freq == 0 and self.lr_scheduler and step_num >= self.lr_scheduler_start:
            self.lr_scheduler.step()
            last_lr = self.lr_scheduler.get_last_lr()[0]
            print(f"Stepping lr scheduler at epoch {epoch_num} with lr {last_lr}")

        return loss.item(), None
    
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
            
            # Get logp(a_expert|state)
            logp = F.log_softmax(action_logits, dim=-1)
            logp_expert_actions = logp.gather(1, expert_actions.unsqueeze(1)).squeeze(1)

            # Get action margin
            _, num_actions = action_logits.shape
            # expert logit: (B,)
            z_expert = action_logits.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
            expert_mask = F.one_hot(expert_actions, num_classes=num_actions).bool()
            # max logit among non-expert actions
            z_max_other = action_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
            margin = (z_expert - z_max_other).mean()

            # Get policy entrop (p(a_i|s)logp(a_i|s))
            p = F.softmax(action_logits, dim=-1)
            entropy = -(p * logp).sum(dim=-1)

        return loss.item(), logp_expert_actions.mean().item(), margin.mean().item(), entropy.mean().item(), accuracy.item()
    
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