
import numpy as np
import torch
from neural_nets import DQN
import torch.nn.functional as F

class AlgorithmBase:
    def __init__(self):
        super().__init__()
        
    def train_step(self):
        raise NotImplementedError

    def select_action(self):
        raise NotImplementedError
    
class DDQN(AlgorithmBase):
    """
    Implementation of the DDQN algorithm. Uses AdamW optimizer and, optionally, a cosine annealing lr scheduler.

    Args
    ----
    obs_dim (int): size of each observation
    action_dim (int): size of each action
    hidden_dim (int): hidden dimension size in DQN network
    gamma (float): 
    tau (float): 
    device (str): 
    lr (float): Learning rate
    loss_fxn (torch.nn.functional): Loss function (ie F.huber_loss, F.mse_loss)
    use_dqn (bool): 
    optimizer_kwargs (optional): 
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, gamma, tau, device, lr, loss_fxn, use_double=True, use_lr_scheduler=False, num_steps=None, **optimizer_kwargs):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.policy_net = DQN(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQN(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.use_double = use_double

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=False, **optimizer_kwargs)
        self.use_lr_scheduler = use_lr_scheduler

        # #TODO
        if use_lr_scheduler:
            lr_scheduler_kwargs = {'T_max': num_steps if num_steps is not None else 100_000, 'eta_min': 1e-6}
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler_kwargs)
        self.loss_fxn = F.mse_loss if loss_fxn is None else loss_fxn

    def train_step(self, batch):
        obs, actions, rewards, next_obs, dones, action_masks = batch

        # convert to tensors
        obs = torch.tensor(np.array(obs), device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_obs = torch.tensor(np.array(next_obs), device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        action_masks = torch.tensor(np.array(action_masks), device=self.device, dtype=torch.bool)
        
        # need to input (batch_size, obs_dim) into net - if obs_dim is 1, we get 1d tensor. Need to reshape
        if obs.dim() == 1:
            obs = obs.unsqueeze(1)
            next_obs = next_obs.unsqueeze(1)
    
        # get policy q vals for current state
        all_q_vals = self.policy_net(obs)
        current_q = all_q_vals.gather(1, actions).squeeze()

        # ddqn
        with torch.no_grad():
            if self.use_double:
                # get policy best actions (need to mask invalid actions)
                next_q_pol = self.policy_net(next_obs)
                # mask_tensor = torch.tensor(action_masks, device=device, dtype=torch.bool)
                next_q_pol[~action_masks] = -1e9

                next_actions_pol = next_q_pol.argmax(1).type(torch.long)
                next_q_targ = self.target_net(next_obs).gather(1, next_actions_pol.unsqueeze(1)).squeeze(1)

                # dqn style 
                # max_next_q = next_q.max(dim=1)[0]

                td_target = rewards + self.gamma * (1 - dones) * next_q_targ

            else:
                next_q = self.target_net(next_obs)

                # mask invalid actions
                next_q[~action_masks] = -1e9# float('-inf')
                # wandb_run.log({'masked next q': next_q},step=t_i)

                max_next_q = next_q.max(dim=1)[0]
                # print('(7) max_masked_next_q', max_next_q)

                td_target = rewards + self.gamma * max_next_q * (1 - dones) # , dtype=torch.float32, device=device



        # print('td_target', td_target)
        loss = self.loss_fxn(current_q, td_target)
        
        # optimize w/ backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=.5)
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
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs).squeeze(0)
            # mask invalid actions
            mask = torch.tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~mask] = float('-inf')
            action = torch.argmax(q_values).item()
        return int(action)
    
    def save(self, filepath):
        torch.save({'policy_net': self.policy_net.state_dict()}, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
    