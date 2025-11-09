import numpy as np
import matplotlib.pyplot as plt
import random

import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete


import optuna

import os
import time
import pickle
import sys
sys.path.append('../src/')

from scipy.spatial.distance import cdist
import argparse
import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F

from collections import namedtuple, deque

# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

def get_distance_matrix(X,Y):
  """"""
  return cdist(X,Y,metric='euclidean')

def get_distance(point1,point2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False

def get_target_sequence(coords, distance_matrix):
    """
    Given a list of coordinates, picks the field closest to the origin first, then 
    always picks the field closest unless it has already been visited
    """
    ordered_indices = np.argsort(distance_matrix, axis=1) # low to high

    start_ind = np.argmin(np.sum(coords**2, axis=1))
    target_indices = [start_ind]

    last_ind = start_ind
    for i in range(len(coords) - 1):
        j = 0
        current_ind = ordered_indices[last_ind][j]
        while current_ind in target_indices:
            j += 1
            current_ind = ordered_indices[last_ind][j]
        target_indices.append(current_ind)
        last_ind = current_ind
    return target_indices, coords[target_indices]

def generate_dataset(num_episodes=100):
    grid_max = 10
    ra_range = (-grid_max, grid_max)
    dec_range = (-grid_max, grid_max)
    n_points = grid_max
    nvisits = 1
    
    # generate random coords
    ra_list = np.random.randint(ra_range[0], ra_range[1], size=(num_episodes, n_points))
    dec_list = np.random.randint(dec_range[0], dec_range[1], size=(num_episodes, n_points))
    coords = np.stack([ra_list, dec_list], axis=2) # shape (num_ep, nra_points, ndec_points)
    coords_dict = {f'eps{i}': coord for i, coord in enumerate(coords)}
    
    distance_matrices = np.empty(shape=(num_episodes, grid_max, grid_max))
    full_target_fields = []
    full_target_coords = []
    # get distance matrices
    for i in range(num_episodes):
        distance_matrices[i] = get_distance_matrix(coords[i], coords[i])
        np.fill_diagonal(distance_matrices[i], np.inf)
        target_fields, target_coords = get_target_sequence(coords[i], distance_matrices[i])
        full_target_fields.append(target_fields)
        full_target_coords.append(target_coords)
        # full_target_coords.append(coords[i][target_fields])
    return np.array(full_target_fields), np.array(full_target_coords), coords_dict, coords
  

class ToyEnv_v1(gym.Env):
    def __init__(self, coords_dict, max_visits, target_fields):
        # instantiate static attributes
        self.coords_dict = coords_dict # field_id: (x,y)
        self.nfields = len(coords_dict)
        self.max_visits = max_visits
        self.zenith = np.array([0,0])
        self.target_sequence = target_fields

        # Initialize variable attributes - will be set in reset()
        self._init_to_nonstate()
       
        # Define observation space
        self.obs_size = 2 + self.nfields * self.max_visits
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1e5,
            shape=(self.obs_size,),
            dtype=np.float32,
        )
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "field_id": Discrete(n=self.nfields, start=-1),
        #         "nvisits": Box(0, max_visits, shape=(self.nfields,), dtype=np.int32),
        #         "step_count": Discrete(n=self.nfields, start=-1),
                # "ra": Box(ra_range[0], ra_range[1], shape=(self.nfields), dtype=np.float32),
                # "dec": Box(dec_range[0], dec_range[1], shape=(self.nfields), dtype=np.float32),
            # }
        # )

        # Define action space        
        self.action_space = gym.spaces.Discrete(self.nfields)

    def _get_obs(self):
        """Convert internal state to observation format.
    
        Returns:
            dict: Observation with agent and target positions
        """
        # return {
        #     "field_id": self._field_id,
        #     "nvisits": self._nvisits,
        #     "step_count": self._step_count
        # }
        obs = np.concatenate([np.array([self._step_count]), np.array([self._field_id]), self._nvisits])
        return obs.astype(np.float32)

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            
        """
        return {'action_mask': self._action_mask}
    
    def _update_action_mask(self):
        """Update mask for cutting invalid actions.
        Must update self._field and self._nvisits before updating actions
        """
        self._prev_action_mask = self._action_mask.copy()
        if self._nvisits[self._field_id] == self.max_visits:
            self._action_mask[self._field_id] = False

    def _update_obs(self):
        self._step_count += 1
        self._nvisits[self._field_id] += 1
        self._visited.append(self._field_id)
        self._coord = self.coords_dict[self._field_id] #TODO need to change for closest distance learning
        self._update_action_mask()

    def _init_to_nonstate(self):
        self._field_id = -1
        self._nvisits = np.full(shape=(self.nfields,), fill_value=0, dtype=np.int32)
        self._step_count = -1
        self._visited = []
        self._coord = np.array([None, None])
        self._action_mask = np.ones(self.nfields, dtype=bool)

    def reset(self, seed = None, options=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # initialize into a non-state.
        # this allows first field choice to be learned
        self._init_to_nonstate()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action: int):
        """Execute one timestep within the environment.

        Args:

        Returns:
        """
        self._field_id = action
        self._update_obs()
        # if self._step_count == 0:
        #     last_field_coord = self.zenith
        # else:
        #     last_field_coord = self.coords_dict[self._visited[-2]]

        # separation = get_distance(self._coord, last_field_coord)
        # if separation <= self.nfields//5*2:
        #     reward = 1
        # elif separation <= self.nfields//5*3:
        #     reward = .5
        # elif separation <= self.nfields//5*4:
        #     reward = .1
        # else:
        #     reward = 0

        off_by_val = np.abs(self.target_sequence[self._step_count] - self._field_id) ==  3
        if self.target_sequence[self._step_count] == self._field_id:
            reward = 1
        elif off_by_val:
            print()
            reward = .25
        else:
            reward = 0    
        total_visits = int(self.nfields * self.max_visits)
        
        all_objects_visited = total_visits == self._step_count + 1

        # end condition
        terminated = all_objects_visited
        truncated = False

        # get obs and info
        next_obs = self._get_obs()
        info = self._get_info()

        return next_obs, reward, terminated, truncated, info

class DQN(nn.Module):
    '''Sends observations to action-value space'''
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Experience stores experience steps gathered in training
# essentially maps (current state, action) to (next state, reward)
Experience = namedtuple(
    "Experience",
    field_names=["obs", "action", "reward", "next_obs", "done", "action_mask", "next_action_mask"],
)

# Stores experiences
class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.buffer = deque([], maxlen=capacity)
        self.full = False
        self.device = device
        self.pos = 0

    def __len__(self):
        return len(self.buffer) 
    
    def append(self, *args):
        """Save a transition"""
        self.buffer.append(Experience(*args))
        self.pos += 1
        if self.pos == self.buffer.maxlen:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones, action_masks, next_action_masks = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=bool),
            np.array(action_masks),
            np.array(next_action_masks)
        )
    
    def reset(self):
        # self.buffer.clear()
        self.full = False
        self.pos = 0

from typing import Tuple

class DQNAgent:
    def __init__(
            self,
            env: gym.Env, 
            replay_buffer: ReplayBuffer, 
            # net: nn.Module,
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

        # set next_obs to current obs for next step
        self.obs = next_obs
        self.info = info

        # if finished survey, reset
        if terminated or truncated:
            self.reset()
        return reward, terminated
    
    def predict(self, nsteps):
        obs = []
        for t_i in nsteps:
            reward, terminated = self.play_step()
          
    
def linear_schedule(eps_start: float, eps_end: float, duration: int, t: int):
    try:
        slope = (eps_end - eps_start) / duration
        return max(slope * t + eps_start, eps_end)
    except:
        raise Exception("Error in linear_schedule")

def exponential_schedule(eps_start: float, eps_end: float, decay_rate: float, t: int):
    try:
        return eps_end + (eps_start - eps_end) * np.exp(-1. * t / decay_rate)
    except:
        raise Exception("Error in exponential_schedule")

from collections.abc import Callable

def train_agent(
        agent: DQNAgent,
        total_timesteps: int,
        # num_episodes, # or total timesteps?
        lr: float,
        batch_size: int,
        gamma: float,
        eps_scheduler_kwargs: dict[str, int | str],
        tau: float,
        # device: torch.device, # take from agent
        eps_scheduler: Callable,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: Dict,
        learn_start: int,
        train_freq: int, #4
        target_freq: int,
        ):
    """
    Trains a DQN agent.
    
    Args
    -----
    agent: DQNAgent
    total_timesteps: int
        Total number of timesteps through which to step agent through environment.
        Number of episodes is total_timesteps // episode_steps
    lr: float
    batch_size: int
    gamma: float
    eps_scheduler_kwargs: dict[str, int | str]
        arguments for epsilon scheduling method
    tau: float
    eps_scheduler: Callable,
        Function that calculates epsilon at each time step
    optimizer: torch.optim.Optimizer,
        Optimizer for neural network. Adamw is recommended.
    optimizer_kwargs: Dict,
        Kwargs for chosen optimizer
    learn_start: int
        Time step at which updates to policy network and target network 
    train_freq: int, #4
        Number of time steps between policy network updates  
    target_freq: int
        Number of time steps between target network updates

    Returns
    -------
    None
    """
    start_time = time.time()
    wandb.log({'train_start_time': start_time})
    obs, info = agent.env.reset()
    optimizer = optimizer(agent.policy_net.parameters(), lr=lr, amsgrad=False, **optimizer_kwargs)

    for t_i in range(total_timesteps):
        # set epsilon according to scheduler
        epsilon = eps_scheduler(t=t_i, **eps_scheduler_kwargs)

        # agent performs step in environment and sees next observation
        _, _ = agent.play_step(epsilon)
        
        # use temporal difference between new obs and last obs to update Q-values
        if t_i > learn_start and t_i % train_freq == 0 and batch_size < len(agent.replay_buffer):
            # sample from experiences
            obs, actions, rewards, next_obs, dones, _, next_action_masks = agent.replay_buffer.sample(batch_size)
            current_q = agent.policy_net(torch.tensor(obs)).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze()

            with torch.no_grad():
            # gets maximally valued action for each observation in batch
                # next_q = agent.target_net(torch.tensor(next_obs, device='cpu'))
                next_q = agent.target_net(torch.tensor(np.array(next_obs), device='cpu'))

                # mask invalid actions
                for i, mask in enumerate(next_action_masks):
                    next_q[i][~torch.tensor(mask)] = float('-inf')

                max_next_q = next_q.max(dim=1)[0]
                rewards = torch.tensor(rewards)
                td_target = rewards + GAMMA * max_next_q * torch.tensor(1 - dones, dtype=torch.float32)

            loss = F.mse_loss(td_target, current_q)
            wandb.log({'loss': loss.item(), 'epsilon': epsilon}, step=t_i)

            # optimize w/ backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update target network
            if t_i % target_freq == 0:
                for target_param, param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
    end_time = time.time()
    wandb.log({'train_end_time': end_time, 'train_duration': end_time - start_time})
    env.close()

        

def cli_args():
        # Make parser object
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("-Nf", "--Nfields", type=int)
    parser.add_argument("-vmax", "--max-visits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("-v", "--verbosity", type=bool)
    parser.add_argument("--timesteps", type=int, default=1e4)
                   
    return(parser.parse_args())

# -------------------------------------------------------------- #
# -------------------------------------------------------------- #

# python run_simple_sequence_imitation.py -Nf 100 -vmax 3 --seed 20 -v True --timesteps 1e5

if __name__ == '__main__':
    args = cli_args()

    env_name = 'ToyEnv_v1'
    OUTDIR = f'results/{env_name}/'
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    SEED = 10
    seed_everything(SEED)
    torch.set_default_dtype(torch.float32)

    # os.environ['WANDB_NOTEBOOK_NAME'] = 'DQN_closest_distance_path.ipynb'
    run = wandb.init(project=f"DQN-{env_name}", dir=OUTDIR)

    # Set parameters from command line args
    target_fields, target_coords, coords_dict, coords = generate_dataset(100)

    # -------------------------------------------------------------- #

    # Register the environment so we can create it with gym.make()
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=SimpleTelEnv,
        max_episode_steps=300,  # Prevent infinite episodes. Here just set to 300 even though episode will terminate when stepping to last element of sequence
    )

    # Register the environment so we can create it with gym.make()
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=ToyEnv_v1,
        max_episode_steps=300,  # Prevent infinite episodes. Here just set to 300 even though episode will terminate when stepping to last element of sequence
    )
    env = gym.make(f"gymnasium_env/{env_name}", coords_dict=coords_dict['eps0'], max_visits=1, target_fields=target_fields[0])
    # Create multiple environments for parallel training
    # vec_env = gym.make_vec("gymnasium_env/SimpleTel-v0", num_envs=5, vectorization_mode='sync', Nf=Nf, target_sequence=true_sequence, nv_max=nv_max)

    from gymnasium.utils.env_checker import check_env

    # This will catch many common issues
    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # -------------------------------------------------------------- #

    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
    )

    BATCH_SIZE = 16
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4
    NUM_EPISODES = 100

    eps_scheduler_kwargs = {'eps_start': EPS_START, 'eps_end': EPS_END, 'decay_rate': EPS_DECAY}
    agent = DQNAgent(
        env=env,
        replay_buffer=ReplayBuffer(capacity=100000, device=device),
        device=device,
    )

    train_agent(
        agent=agent,
        total_timesteps=1000,
        lr=LR,
        batch_size=5,
        gamma=GAMMA,
        eps_scheduler_kwargs={
            'eps_start': EPS_START,
            'eps_end': EPS_END,
            'decay_rate': EPS_DECAY,
        },
        tau=TAU,
        eps_scheduler=exponential_schedule,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={},
        learn_start=100,
        train_freq=2,
        target_freq=2,
    )

    action_masks = []
    actions = []
    observations = []

    agent.reset()
    for i in range(10):
        action_mask = agent.info['action_mask']
        action = agent.select_action(0)
        actions.append(action)

        next_obs, reward, terminated, truncated, next_info = agent.env.step(action)
        next_action_mask = next_info['action_mask']
        action_masks.append(next_action_mask)
        observations.append(next_obs)

        agent.obs = next_obs
        agent.info = next_info
        if terminated:
            break
