from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np


from typing import Optional
import numpy as np

class TelescopeEnv_v0(gym.Env):
    """
    Environment compatible with state input = (field id)
    """
    def __init__(self, dataset, use_separation_reward=True, use_field_id_reward=True):
        super().__init__()
        # instantiate static attributes
        self.nfields = dataset._nfields
        self.id2pos = dataset._id2pos
        self.max_visits = dataset._max_visits
        self._n_obs_per_night = dataset._n_obs_per_night
        self.use_separation_reward = use_separation_reward
        self.use_field_id_reward = use_field_id_reward
        self.target_field_ids = dataset._schedule_field_ids[0]

        # Initialize variable attributes - will be set in reset()
        self._init_to_nonstate()
       
        self.obs_dim = dataset.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-10, #np.min(dataset.obs),
            high=np.max(dataset.obs)+1,
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
        self._init_to_nonstate()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    # TODO
    def _get_reward(self, last_obs, current_obs):
        raise NotImplementedError
    
    def step(self, action: int):
        """Execute one timestep within the environment.

        Args:

        Returns:
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        # last_coord = self._coord
        last_field_id = self._field_id
        self._update_obs(action)
        
        # ------------------- Calculate reward ------------------- #
        reward = 0
        # 1. Separation
        if self.use_separation_reward:
            pass
        #     separation = np.linalg.norm(self._coord - last_coord)
        #     low_sep_lim = 3
        #     mid_sep_lim = 6
        #     max_sep_lim = 10

        #     if separation < low_sep_lim:
        #         reward += .3
        #     elif separation < mid_sep_lim:
        #         reward += .2
        #     elif separation < max_sep_lim:
        #         reward += .1
        
        # 2. Field ID
        if self.use_field_id_reward:
            target_field = self.target_field_ids[self._obs_idx]
            field_id_diff = np.abs(target_field - self._field_id)
            if field_id_diff == 0:
                reward += 1
            elif field_id_diff <= 1:
                reward += .3
            elif field_id_diff <= 3:
                reward += .1
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

    def _init_to_nonstate(self):
        self._obs_idx = -1
        self._field_id = -1
        self._action_mask = np.ones(self.nfields, dtype=bool)
        self._visited = []

    def _update_action_mask(self, action):
        """Update mask for cutting invalid actions.
        Must update self._field and self._nvisits before updating actions
        """
        if self._visited.count(action) == self.max_visits:
            self._action_mask[action] = False

    def _update_obs(self, action):
        self._obs_idx += 1
        self._field_id = action
        self._coord = np.array(self.id2pos[action], dtype=np.float32)
        self._visited.append(action)
        self._update_action_mask(action)

    def _get_obs(self):
        """Convert internal state to observation format.
    
        Returns:
            dict: Observation with agent and target positions
        """
        obs = np.array([self._obs_idx/self._n_obs_per_night, self._field_id/self.nfields], dtype=np.float32)
        # obs = np.concatenate((np.array([self._field_id]), self._coord.flatten()), dtype=np.float32)
        return obs

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            
        """
        return {'action_mask': self._action_mask.copy()}
