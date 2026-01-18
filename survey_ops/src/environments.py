from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from survey_ops.utils import ephemerides, units
from survey_ops.utils.interpolate import interpolate_on_sphere
import random
from survey_ops.utils.geometry import angular_separation


class BaseTelescope(gym.Env):
    """
    Base class providing for a Gymnasium environment simulating sequential observation scheduling.

    This class provides core Gym Env structure (reset, step) and common logic.

    Subclasses must define the following methods:
        - _init_to_first_state(): Initializes the environment state for a new episode
        - _update_action_mask(): Updates action masks
        - _update_obs(action): Updates the internal state based on teh action taken
        - _get_state(): Converts internal state into the formal observation
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
        self._init_to_first_state(options=options)
        obs = self._get_state()
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
        next_obs = self._get_state()
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

    def _update_state(self, action):
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

    def _get_state(self):
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
    def __init__(self, test_dataset, max_nights=None, exp_time=90., slew_time=30.):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        # Instantiate static attributes
        self.test_dataset = test_dataset
        self.exp_time = exp_time
        self.slew_time = slew_time
        self.time_between_obs = exp_time + slew_time
        self.time_dependent_feature_substrs = ['az', 'el', 'ha', 'time_fraction_since_start']
        self.cyclical_feature_names = test_dataset.cyclical_feature_names
        self.z_score_feature_names = test_dataset.z_score_feature_names
        self.do_z_score_norm = test_dataset.do_z_score_norm
        self.z_score_feature_names = test_dataset.z_score_feature_names
        self.do_cyclical_norm = test_dataset.do_cyclical_norm
        self.do_max_norm = test_dataset.do_max_norm
        self.max_norm_feature_names = test_dataset.max_norm_feature_names
        self.do_inverse_airmass = test_dataset.do_inverse_airmass
        self.include_bin_features = len(test_dataset.bin_feature_names) > 0

        # Dataset-wide mappings
        self.field2radec = test_dataset.field2radec
        self.field_ids = test_dataset.field_ids
        self.field_radecs = test_dataset.field_radecs

        # Bin-space dependent function to get fields in bin
        self.get_fields_in_bin = test_dataset.get_fields_in_bin
        if not test_dataset.hpGrid.is_azel:
            self.bin_space = 'radec'
        else:
            self.bin_space = 'azel'

        self.field2nvisits = {int(fid): int(count) for fid, count in test_dataset.field2nvisits.items()}
        self.nfields = len(self.field2nvisits)

        self.hpGrid = test_dataset.hpGrid
        self.nbins = len(self.hpGrid.lon)
        
        self.base_state_feature_names = test_dataset.base_feature_names
        self.base_pointing_feature_names = test_dataset.base_pointing_feature_names
        self.base_bin_feature_names = test_dataset.base_bin_feature_names
        self.state_feature_names = test_dataset.state_feature_names
        self.pointing_feature_names = test_dataset.pointing_feature_names
        self.bin_feature_names = test_dataset.bin_feature_names

        if self.do_z_score_norm:
            self.zscore_means = test_dataset.means.detach().numpy()
            self.zscore_stds = test_dataset.stds.detach().numpy()

        self.pointing_pd_nightgroup = test_dataset._df.groupby('night')
        if self.include_bin_features:
            self.bin_pd_nightgroup = test_dataset._bin_df.groupby('night')

        self.max_nights = max_nights
        if max_nights is None:
            self.max_nights = self.pointing_pd_nightgroup.ngroups
        if hasattr(test_dataset, 'reward_func'):
            self._reward_func = test_dataset.reward_func
        else:
            self._reward_func = lambda x_prev, x_cur: angular_separation(pos1=x_prev, pos2=x_cur)

        self.obs_dim = test_dataset.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-100, #np.min(dataset.obs),
            high=1e8,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        # Define action space        
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.nbins, len(self.field2radec)]), dtype=np.int32)

        self._state = np.zeros(self.obs_dim, dtype=np.float32)
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
        # self._night_idx = -1
        # self._visited = []

        self._init_to_first_state()
        state = self._get_state()
        info = self._get_info()
        return state, info

    def step(self, actions: np.ndarray):
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
        assert self.action_space.contains(actions), f"Invalid action {actions}"
        action, field_id = np.int32(actions[0]), int(actions[1])
        # assert field_id in self.get_fields_in_bin(action), f"Field ID {field_id} not in bin {action}"
        last_field_id = np.int32(self._field_id)

        # ------------------- Advance state ------------------- #
        self._update_state((action, field_id))
        
        # ------------------- Calculate reward ------------------- #

        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)

        # -------------------- Start new night if last transition -----------------------#

        # is_new_night = self._state[-1] >= 1.0 # time_fraction_since_start > = 1.0
        is_new_night = self._timestamp >= self._night_final_timestamp
        self._is_new_night = is_new_night
        if is_new_night:
            self._start_new_night()
        
        # -------------------- Terminate condition -----------------------#
        truncated = False
        terminated = self._get_termination_status()

        # get obs and info
        next_state = self._get_state()
        info = self._get_info()

        return next_state, reward, terminated, truncated, info

    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self, init_state=None, options=None):
        """
        Initializes the internal state variables for the start of a new episode.

        The episode starts at the beginning of the observation window (index 0)
        and with the telescope pointing at the first target field.
        """
        self._action_mask = np.zeros(self.nbins, dtype=bool)
        valid_bins = np.arange(len(self.hpGrid.idx_lookup))
        self._action_mask[valid_bins] = True

        if init_state is None:
            self._visited = []
            self._night_idx = -1
            self._is_new_night = True
            self._start_new_night()
        else:
            self._state = options['init_state']
            self._timestamp = options['init_timestamp']

            field_id = options['field_id']
            self._field_id = field_id
            self._visited = [field_id]
            raise NotImplementedError
        
    
    def _start_new_night(self):
        self._night_idx +=1
        self._is_new_night = True
        if self._night_idx >= self.max_nights:
            return

        # Pointing features
        first_row_in_night_pointing = self.pointing_pd_nightgroup.head(1).iloc[self._night_idx]
        self._night_final_timestamp = self.pointing_pd_nightgroup.tail(1).iloc[self._night_idx]['timestamp']
        self._night_first_timestamp = first_row_in_night_pointing['timestamp']
        self._field_id = first_row_in_night_pointing['field_id']
        self._bin_num = first_row_in_night_pointing['bin']
        # print(first_row_in_night_pointing)
        self._timestamp = first_row_in_night_pointing['timestamp']
        self._visited.append(self._field_id)

        # Bin features
        
        # first_feature_state_in_night = self.test_dataset._get_pointing_features_from_row(row=first_row_in_night)
        self._pointing_state_features = [first_row_in_night_pointing[feat_name] for feat_name in self.pointing_feature_names]
        if self.include_bin_features:
            first_row_in_night_bin = self.bin_pd_nightgroup.head(1).iloc[self._night_idx]
            self._bin_state_features = [first_row_in_night_bin[feat_name] for feat_name in self.bin_feature_names]
        else:
            self._bin_state_features = []
        self._state = self._pointing_state_features + self._bin_state_features

    def _get_azel_action_mask(self, timestamp):
        # At timestamp, get bins with fields in them
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=self.field_radecs[:, 0], dec=self.field_radecs[:, 1], time=timestamp)
        field_bins = self.hpGrid.ang2idx(lon=fields_az, lat=fields_el)
        field_bins = field_bins[field_bins != None]
        bins_with_fields = np.unique(field_bins).astype(np.int32)
        print(bins_with_fields)
        mask = np.zeros(self.nbins, dtype=bool)
        mask[bins_with_fields] = True
        return mask

    def _update_radec_action_mask(self, action, timestamp=None): #DONE
        """
        Updates the boolean mask that tracks valid actions (field IDs).

        An action becomes invalid if the target field has already been visited
        the maximum allowed number of times (`self.max_visits`).

        Args:
            action (int): The field ID to check and potentially mask.
        """
        # If all fields in bin are fully visited, bin is no longer valid action
        for field_id in self.get_fields_in_bin(action, timestamp):
            max_nvisits = self.field2nvisits[field_id]
            if self._visited.count(field_id) < max_nvisits:
                return
        self._action_mask[action] = False
            
    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._timestamp += self.time_between_obs
        action, field_id = int(action[0]), int(action[1])

        self._bin_num = action
        self._field_id = field_id
        self._visited.append(field_id)

        new_features = {}
        new_features['ra'], new_features['dec'] = self.field2radec[field_id]
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=self._timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=self._timestamp)
        
        cos_zenith = np.cos(90 * units.deg - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=self._timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=self._timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=self._timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=self._timestamp)

        if self._night_final_timestamp == self._night_first_timestamp:
            new_features['time_fraction_since_start'] = 0
        else:
            new_features['time_fraction_since_start'] = (self._timestamp - self._night_first_timestamp) / (self._night_final_timestamp - self._night_first_timestamp)
        
        for feat_name in self.base_pointing_feature_names:
            if any(string in feat_name and 'bin' not in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                new_features.update({f'{feat_name}_cos': np.cos(new_features[feat_name])})
                new_features.update({f'{feat_name}_sin': np.sin(new_features[feat_name])})



        self._state = np.array([new_features.get(feat, 0.0) for feat in self.state_feature_names], dtype=np.float32)
        self._update_action_mask(action, self._timestamp)
        
    def _get_state(self):
        """Converts the current internal state into the formal observation format.

        The observation is a vector containing the current field ID and the
        current observation index (time step).

        Returns
        -------
            np.ndarray: The observation vector, potentially normalized.
        """

        self._state = np.array(self._state, dtype=np.float32)
        state_copy = np.array(self._state, dtype=np.float32).copy()
        # state_norm = state_copy
        if not self._is_new_night:
            state_copy = self._do_noncyclic_normalizations(state=state_copy)
        self._state_normed = state_copy.copy()
        return state_copy

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy(), 
                'visited': self._visited,
                'timestamp': int(self._timestamp),
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id)
                } # 'night_idx': self._night_idx, 'timestamp': self._timestamp, 'field_id': self._field_id}
    
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met or exceeded.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        terminated = self._night_idx >= self.max_nights
        return terminated

    def _update_action_mask(self, action, time):
        if self.hpGrid.is_azel:
            self._action_mask = self._get_azel_action_mask(time)
        if not self.hpGrid.is_azel:
            self._update_radec_action_mask(action, time)

    def _do_noncyclic_normalizations(self, state):
        """Performs z-score normalization on any non-periodic features, including bin-specific features"""
        if self.do_inverse_airmass:
            airmass_mask = np.array(['airmass' in feat_name for feat_name in self.state_feature_names], dtype=bool)
            normalized_airmasses = 1.0 / state[airmass_mask]
            airmasses_fixed_nans = np.where(np.isnan(normalized_airmasses), 10.0, normalized_airmasses)
            state[airmass_mask] = airmasses_fixed_nans
        
        if self.do_max_norm:
            max_norm_mask = np.array([
                any(max_feat in feat_name for max_feat in self.max_norm_feature_names)
                for feat_name in self.state_feature_names
                ], dtype=bool)
            state[max_norm_mask] = state[max_norm_mask] / (np.pi/2)

        # z-score normalization for any non-periodic features
        if self.do_z_score_norm:
            z_score_mask = np.array([
            any(z_feat in feat_name for z_feat in self.z_score_feature_names) 
            for feat_name in self.state_feature_names
            ])
            state[z_score_mask] = ((state[z_score_mask] - self.zscore_means) / self.zscore_stds)
        state[np.isnan(state)] = 10 # for airmass nan values -- only airmass should be high

        return state