from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np


from typing import Optional
import numpy as np

class TelescopeEnv_v0(gym.Env):
    """
    Environment compatible with TelescopeDatasetv0
    """
    def __init__(self, dataset):
        # super().__init__()
        # instantiate static attributes
        self.nfields = dataset._nfields
        self.id2pos = dataset._id2pos
        self.max_visits = dataset._max_visits
        self._n_obs_per_night = dataset._n_obs_per_night
        self.target_field_ids = dataset._schedule_field_ids[0]
        # Initialize variable attributes - will be set in reset()
        self.reward_func = dataset.reward_func
        self.normalize_obs = dataset.normalize_obs
        self.norm = np.array([1,1])
        if self.normalize_obs:
            self.norm = dataset.norm.flatten()
       
        self._init_to_first_state()
        self.obs_dim = dataset.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-1, #np.min(dataset.obs),
            high=1e5,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Define action space        
        self.action_space = gym.spaces.Discrete(self.nfields)
    
    # ------------------------------------------------------------ #
    # -----------------------Gymnasium API ----------------------- #
    # ------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
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
        self._init_to_first_state()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    # TODO
    def _get_rewards(self, last_field, current_field):
        if self.reward_func is None:
            return 1
        return self._reward_func(last_field, current_field)

    def step(self, action: int):
        """Execute one timestep within the environment.

        Args:

        Returns:
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        # last_coord = self._coord
        last_field_id = np.int32(self._field_id)
        self._update_obs(action)
        
        # ------------------- Calculate reward ------------------- #
        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)
        # ------------------------------------------ #

        # end condition
        truncated = False
        terminated = self._obs_idx + 1 >= self._n_obs_per_night

        # get obs and info
        next_obs = self._get_obs()
        info = self._get_info()

        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self):
        self._obs_idx = 0
        self._field_id = self.target_field_ids[0]
        self._action_mask = np.ones(self.nfields, dtype=bool)
        self._visited = [self.target_field_ids[0]]
        self._update_action_mask(int(self.target_field_ids[0]))

    def _update_action_mask(self, action):
        """Update mask for cutting invalid actions.
        Must update self._field and self._nvisits before updating actions
        """
        if self._visited.count(action) == self.max_visits:
            self._action_mask[action] = False

    def _update_obs(self, action):
        self._obs_idx += 1
        self._field_id = action
        # self._coord = np.array(self.id2pos[action], dtype=np.float32)
        self._visited.append(action)
        self._update_action_mask(action)

    def _get_obs(self):
        """Convert internal state to observation format.
    
        Returns:
            dict: Observation with agent and target positions
        """
        obs = np.array([self._field_id, self._obs_idx], dtype=np.float32)
        # obs = np.concatenate((np.array([self._field_id]), self._coord.flatten()), dtype=np.float32)
        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            
        """
        return {'action_mask': self._action_mask.copy()}
