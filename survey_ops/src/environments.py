from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd

from survey_ops.utils.ephemerides import get_source_ra_dec, equatorial_to_topographic, topographic_to_equatorial, HealpixGrid
from survey_ops.utils.interpolate import interpolate_on_sphere

#TODO
# interpolate_on_sphere(az, el, az_data, el_data, values)

class BaseTelescope(gym.Env):
    """
    Base class providing for a Gymnasium environment simulating sequential observation scheduling.

    This class provides core Gym Env structure (reset, step) and common logic.

    Subclasses must define the following methods:
        - _init_to_first_state(): Initializes the environment state for a new episode
        - _update_action_mask(): Updates action masks
        - _update_obs(action): Updates the internal state based on teh action taken
        - _get_obs(): Converts internal state into the formal observation
        - _get_info(): Computes auxiliary information dictionary
        - _get_termination_status(): Checks if episode has terminated
    """
    def __init__(self):
        '''
        Subclasses must instantiate the following attributes here *before* calling self._init_to_first_state()):
            - reward_func (Callable): Function to calculate rewards for transitions
            - observation_space (gym.spaces)
            - action_space (gym.spaces)
            - obs_dim (int): Size of observation input vector
            - normalize_obs (bool): Whether observations should be normalized
            - norm (nd.array): Normalization factors for observations. If normalize_obs is False, it is simply an array of ones.
        Subclasses must call self._init_to_first_state() after assigning attributes
        '''

        super().__init__()

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args
        ----
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns
        -------
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
    
    def step(self, action: int):
        """Execute one timestep within the environment.

        Args
        ----
            action (int): The field ID to observe next.

        Returns
        -------
            tuple: (next_obs, reward, terminated, truncated, info)
                - next_obs (np.ndarray): The observation after the action.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has ended (e.g., reached observation limit).
                - truncated (bool): Whether the episode was truncated (always False here).
                - info (dict): Auxiliary diagnostic information.
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
        terminated = self._get_termination_status()
        # get obs and info
        next_obs = self._get_obs()
        info = self._get_info()

        return next_obs, reward, terminated, truncated, info
    
    def _get_rewards(self, last_field, next_field):
        '''
        Calculates the reward for a single state transition.

        Uses self._reward_func() if available, otherwise returns 1.

        Args
        ----
            last_field (int): Field ID before taking the action.
            next_field (int): Field ID after taking the action.

        Returns
        -------
            float: The calculated reward value.
        '''
        if getattr(self, "reward_func", None) is None:
            return 1
        return self._reward_func(last_field, next_field)

class ToyEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with TelescopeDatasetv0.

    This environment models a single night of observation scheduling. The agent's
    goal is to sequentially select fields to observe, constrained by a maximum
    number of total observations for the night and a maximum number of visits
    per individual field.

    The observation space is a 1D Box representing the current state (Field ID and time index).
    The action space is a Discrete space over all possible field IDs.
    """
    def __init__(self, dataset):
        """
        Initializes the ToyEnv with parameters from the dataset.

        Args
        ----
            dataset: An object (assumed to be TelescopeDatasetv0) containing
                     static environment parameters and observation data.
        """
        # instantiate static attributes
        self.nfields = dataset._nfields
        self.id2pos = dataset._id2pos
        self.max_visits = dataset._max_visits
        self._n_obs_per_night = dataset._n_obs_per_night
        self.target_field_ids = dataset._schedule_field_ids[0]
        self.obs_dim = dataset.obs_dim
        self.reward_func = dataset.reward_func
        self.normalize_obs = dataset.normalize_obs
        self.norm = np.ones(shape=self.obs_dim)
        if self.normalize_obs:
            self.norm = dataset.norm.flatten()
        self.observation_space = gym.spaces.Box(
            low=-1, #np.min(dataset.obs),
            high=1e5,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        # Define action space        
        self.action_space = gym.spaces.Discrete(self.nfields)
        super().__init__()
    
    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.

        The episode starts at the beginning of the observation window (index 0)
        and with the telescope pointing at the first target field.
        """
        self._obs_idx = 0
        self._field_id = self.target_field_ids[0]
        self._action_mask = np.ones(self.nfields, dtype=bool)
        self._visited = [self.target_field_ids[0]]
        self._update_action_mask(int(self.target_field_ids[0]))

    def _update_action_mask(self, action):
        """
        Updates the boolean mask that tracks valid actions (field IDs).

        An action becomes invalid if the target field has already been visited
        the maximum allowed number of times (`self.max_visits`).

        Args:
            action (int): The field ID to check and potentially mask.
        """
        if self._visited.count(action) == self.max_visits:
            self._action_mask[action] = False

    def _update_obs(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._obs_idx += 1
        self._field_id = action
        # self._coord = np.array(self.id2pos[action], dtype=np.float32)
        self._visited.append(action)
        self._update_action_mask(action)

    def _get_obs(self):
        """Converts the current internal state into the formal observation format.

        The observation is a vector containing the current field ID and the
        current observation index (time step).

        Returns
        -------
            np.ndarray: The observation vector, potentially normalized.
        """
        obs = np.array([self._field_id, self._obs_idx], dtype=np.float32)
        # obs = np.concatenate((np.array([self._field_id]), self._coord.flatten()), dtype=np.float32)
        return obs

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy()}
    
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met or exceeded.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        terminated = self._obs_idx + 1 >= self._n_obs_per_night
        return terminated

class OfflineEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, dataset):
        """
        Args
        ----
            dataset: An object (assumed to be TelescopeDatasetv0) containing
                     static environment parameters and observation data.
        """
        # instantiate static attributes
        self.dataset = dataset

        self.unique_radec, field_counts = np.unique([(ra, dec) for ra, dec in zip(dataset._df['ra'].values, dataset._df['dec'].values)], axis=0, return_counts=True)
        self.idx2radec = dataset.idx2radec
        self.timestamps = self.dataset.timestamps
        self.dones = dataset.dones
        self.reward_func = dataset.reward_func
        self.normalize_obs = dataset.normalize_obs
        self.norm = np.ones(shape=self.obs_dim)

        if self.normalize_obs:
            self.means = dataset.means
            self.stds = dataset.stds

        self.obs_dim = dataset.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-10, #np.min(dataset.obs),
            high=1e5,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        # Define action space        
        self.action_space = gym.spaces.Discrete(self.num_actions)
        super().__init__()
    
    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.

        The episode starts at the beginning of the observation window (index 0)
        and with the telescope pointing at the first target field.
        """
        first_state = np.zeros(self.obs_dim, dtype=np.float32)
        
        self._field_id = self.target_field_ids[0]
        self._action_mask = np.ones(self.nfields, dtype=bool)
        self._visited = [self.target_field_ids[0]]
        self._update_action_mask(int(self.target_field_ids[0]))

    def _update_action_mask(self, action):
        """
        Updates the boolean mask that tracks valid actions (field IDs).

        An action becomes invalid if the target field has already been visited
        the maximum allowed number of times (`self.max_visits`).

        Args:
            action (int): The field ID to check and potentially mask.
        """
        if self._visited.count(action) == self.max_visits:
            self._action_mask[action] = False

    def _update_obs(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._bin_id = action
        # self._coord = np.array(self.id2pos[action], dtype=np.float32)
        self._visited.append(action)
        self._update_action_mask(action)

    def _get_obs(self):
        """Converts the current internal state into the formal observation format.

        The observation is a vector containing the current field ID and the
        current observation index (time step).

        Returns
        -------
            np.ndarray: The observation vector, potentially normalized.
        """

        obs = np.array([
            self._az, 
            self._el,
            self._sun_az,
            self._sun_el,
            self._moon_az,
            self._moon_el,
            self._airmass,
            self._ha,
            self._timestamp
            ], 
            dtype=np.float32)
        # obs = np.concatenate((np.array([self._field_id]), self._coord.flatten()), dtype=np.float32)
        return obs

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy()}
    
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met or exceeded.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        terminated = self._obs_idx + 1 >= self.num_transitions
        return terminated
