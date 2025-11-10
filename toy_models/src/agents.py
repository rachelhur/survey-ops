import gymnasium as gym
from collections import defaultdict
from buffer import ReplayBuffer
import torch

from typing import Tuple

class DQNAgent:
    def __init__(
            self,
            env: gym.Env,
            replay_buffer: ReplayBuffer,
            device,
            ):
        """Base Agent class handling the interaction with the environment.
        """
        self.env = env
        self.replay_buffer = replay_buffer
        n_observations = len(self.env.reset()[0])
        n_actions = self.env.action_space.n
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_phase = True

        self.reset()
        self.steps_done = 0
        self.device = device
        self.start_time = time.time()


    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.obs, self.info = self.env.reset()

    def select_action(self, epsilon: float) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action

        """
        # if random sample less than epsilon, take random action
        if self.training_phase and np.random.random() < epsilon:
            valid_actions = np.where(self.info['action_mask'])[0]
            action = self.env.np_random.choice(valid_actions)
            return action

        # get action given obs using policy
        obs = torch.tensor(self.obs)
        if self.device == torch.device('cuda'):
            obs = obs.cuda(self.device)

        with torch.no_grad():
            obs_tensor = obs.unsqueeze(0)
            q_values = self.policy_net(obs_tensor).squeeze(0)

            # Apply mask: set invalid actions to -inf
            masked_q_values = q_values.clone()
            masked_q_values[torch.tensor(~self.info['action_mask'])] = float('-inf')
            action = torch.argmax(masked_q_values).item()
            # action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(
        self,
        epsilon: float = 0.0,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done

        """
        action_mask = self.info['action_mask']
        # select action
        action = self.select_action(epsilon)

        # interact with environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        next_action_mask = info['action_mask']

        # save to experiences
        exp_args = self.obs, action, reward, next_obs, terminated, action_mask, next_action_mask
        self.replay_buffer.append(*exp_args)

        # if finished survey, reset
        if terminated or truncated:
            self.reset()

        else:
            # set next_obs to current obs for next step
            self.obs = next_obs
            self.info = info

        return reward, terminated

    def predict(self, nsteps):
        obs = []
        for t_i in nsteps:
            reward, terminated = self.play_step()


#class QlearnAgent:
#    def __init__(self, env, lr, eps_init, eps_decay, eps_final, discount_factor):
#        self.env = ev
#        
#        # learning hyperparams
#        self.lr = lr # rate at which to scale q-val
#        self.discount_factor = discount_factor
#        self.eps = eps_init # parameterizes greediness
#        self.eps_decay = eps_decay # linear decay factor
#        self.eps_final = eps_final # low limit eps
#
#        self.training_error = []
#        
#        # q (action-value) function
#        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
#
#    def get_action(self, obs):
#        # if less than eps, explore, else exploit
#        if np.random.random() < self.eps:
#            return self.env.action_space.sample()
#        else:
#            return int(np.argmax(self.q_values[obs]))
#
#    def update(self, obs, action, reward, terminated, next_obs):
#
#        future_q_val = np.max(self.q_values[next_obs]) if not terminated else 0
#
#        # bellman eq'n
#        target = reward + self.discount_factor * future_q_val
#
#        # temporal diff
#        temp_diff = target - self.q_values[obs][action]
#
#        # update q
#        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temp_diff)
#
#        # track training error
#        self.training_error.append(temp_diff)
#
#    def decay_epsilon(self):
#        self.eps = max(self.final_eps, self.eps - self.eps_decay)
