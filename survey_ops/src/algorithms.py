
import numpy as np
import torch
import torch.nn.functional as F

from survey_ops.src.neural_nets import DQN

class AlgorithmBase:
    def __init__(self):
        super().__init__()
        
    def train_step(self):
        raise NotImplementedError

    def select_action(self):
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

    def __init__(self, obs_dim, num_actions, hidden_dim, gamma, tau, device, lr, loss_fxn, use_double=True, \
                 use_lr_scheduler=False, num_steps=None, **optimizer_kwargs):
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
        self.use_lr_scheduler = use_lr_scheduler

        # TODO: I want to input number of epochs for num_steps for cosine annealing T_max param
        if use_lr_scheduler:
            lr_scheduler_kwargs = {'T_max': num_steps if num_steps is not None else 100_000, 'eta_min': 1e-6}
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs)
        self.loss_fxn = F.mse_loss if loss_fxn is None else loss_fxn

    def train_step(self, batch):
        
        obs, actions, rewards, next_obs, dones, action_masks = batch
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_obs = torch.tensor(np.array(next_obs), device=self.device, dtype=torch.float32)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
            action_masks = torch.tensor(np.array(action_masks), device=self.device, dtype=torch.bool)
            
        # convert to tensors
        else:
            obs = obs.to(device=self.device, dtype=torch.float32)
            actions = actions.to(device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
            rewards = rewards.to(device=self.device, dtype=torch.float32)
            next_obs = next_obs.to(device=self.device, dtype=torch.float32)
            dones = dones.to(device=self.device, dtype=torch.float32)
            action_masks = action_masks.to(device=self.device, dtype=torch.bool)
            
        # need to input (batch_size, obs_dim) into net - if obs_dim is 1, we get 1d tensor. Need to reshape
        if obs.dim() == 1:
            obs = obs.unsqueeze(1)
            next_obs = next_obs.unsqueeze(1)
    
        # get policy q vals for current state
        all_q_vals = self.policy_net(obs)
        current_q = all_q_vals.gather(1, actions).squeeze()

        with torch.no_grad():
            if self.use_double:
                # get policy best actions (need to mask invalid actions)
                next_q_pol = self.policy_net(next_obs)
                next_q_pol[~action_masks] = float('-inf') #-1e9
                next_actions_pol = next_q_pol.argmax(1).type(torch.long)

                # evaluate policy's next actions using target net
                next_q_targ = self.target_net(next_obs).gather(1, next_actions_pol.unsqueeze(1)).squeeze(1)

                td_target = rewards + self.gamma * (1 - dones) * next_q_targ
            else:
                next_q = self.target_net(next_obs)
                # mask invalid actions
                next_q[~action_masks] = -1e9 # float('-inf')
                max_next_q = next_q.max(dim=1)[0]
                td_target = rewards + self.gamma * max_next_q * (1 - dones) # , dtype=torch.float32, device=device

        loss = self.loss_fxn(current_q, td_target)
        
        # optimize w/ backprop
        self.optimizer.zero_grad()
        loss.backward()
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
        prev_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.)
        # print(f"Max Gradient Norm: {prev_grad_norm.item()}")
        self.optimizer.step()
        if self.use_lr_scheduler:
            self.scheduler.step()
                    
        self._soft_update()

        # soft update done in agent
        return loss.item(), current_q.mean().item()

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
            mask = torch.tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~mask] = float('-inf')
            action = torch.argmax(q_values).item()
        return int(action)
    

class BehaviorCloning(AlgorithmBase):
    def __init__(self, obs_dim, num_actions, hidden_dim, loss_fxn=None, lr=1e-3, device='cpu'):
        self.name = 'BehaviorCloning'
        self.device = device
        self.policy_net = DQN(obs_dim, num_actions, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fxn = torch.nn.CrossEntropyLoss(reduction='mean') if loss_fxn is None else loss_fxn
        self.lr_scheduler = None
        
    def train_step(self, batch):
        """
        Train the policy to mimic expert actions from offline data
        """
        obs, expert_actions, rewards, next_obs, dones, action_masks = batch
        
        # convert to tensors
        if not torch.is_tensor(obs):
            obs = torch.tensor(np.array(obs), dtype=torch.float32)
            expert_actions = torch.tensor(expert_actions, dtype=torch.long) # needs to be long for .gather()
        obs = obs.to(device=self.device, dtype=torch.float32)
        expert_actions = expert_actions.to(device=self.device, dtype=torch.long)
        action_logits = self.policy_net(obs)
        
        loss = self.loss_fxn(action_logits, expert_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), 0.0
    
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