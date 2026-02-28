from collections import defaultdict
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd
from survey_ops.coreRL.data_processing import get_nautical_twilight

from survey_ops.coreRL.data_processing import normalize_noncyclic_features
from survey_ops.utils import ephemerides, units
from survey_ops.utils.interpolate import interpolate_on_sphere
import random
from survey_ops.utils.geometry import angular_separation
from survey_ops.coreRL.survey_logic import get_fields_in_bin
from survey_ops.coreRL.offline_dataset import setup_feature_names
from survey_ops.coreRL.data_processing import *

from astropy.time import Time
from datetime import datetime, timezone
import pickle
import json

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

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
        self._init_to_first_state()
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
    

class OfflineDECamTestingEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, max_nights=None, exp_time=90., slew_time=30., global_pd_nightgroup=None, bin_pd_nightgroup=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        
        # Assign static attributes
        self.exp_time = exp_time
        self.slew_time = slew_time
        self.time_between_obs = exp_time + slew_time
        self.time_dependent_feature_substrs = gcfg['features']['TIME_DEPENDENT_FEATURE_NAMES']
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES']
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES']
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.bin_space = cfg['data']['bin_space']
        nside = cfg['data']['nside']
        self.hpGrid = None if cfg['data']['bin_method'] != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=(self.bin_space == 'azel'))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        if any(f in cfg['data']['bin_features'] for f in ['night_num_visits', 'night_num_unvisited_fields', 'night_num_incomplete_fields']):
            self._has_historical_features = True
        else:
            self._has_historical_features = False

        # Dataset-wide mappings        
        # with open(gcfg['paths']['LOOKUP_DIR'] + '/' + gcfg['files']['FIELD2NAME'], 'r') as f:
        #     field2name = json.load(f)
        #     self.field2name
        with open(gcfg['paths']['LOOKUP_DIR'] + '/' + gcfg['files']['FIELD2RADEC'], 'r') as f:
            field2radec = json.load(f)
            self.field2radec = {int(k): v for k, v in field2radec.items()}
        with open(gcfg['paths']['LOOKUP_DIR'] + '/' + gcfg['files']['FIELD2MAXVISITS_EVAL'], 'r') as f:
            field2maxvisits = json.load(f)
            self.field2maxvisits = {int(fid): int(count) for fid, count in field2maxvisits.items()}
        
        # Field to index mapping for sparse field ids; unused fields maps to -1
        self.nfields = len(self.field2maxvisits)
        self._fids = np.array(list(self.field2maxvisits.keys()))
        self._ra_arr = np.zeros(self.nfields)
        self._dec_arr = np.zeros(self.nfields)
        self._max_visits_arr = np.zeros(self.nfields, dtype=np.int32)
        for idx, fid in enumerate(self._fids):
            self._ra_arr[idx], self._dec_arr[idx] = self.field2radec[fid]
            self._max_visits_arr[idx] = self.field2maxvisits[fid]

        max_fid = self._fids[-1]
        fid2idx = np.full(max_fid + 1, -1, dtype=np.int32)
        for idx, fid in enumerate(self._fids):
            fid2idx[fid] = idx
 
        with open(gcfg['paths']['LOOKUP_DIR'] + gcfg['files']['DELVE_NIGHT2FIELDVISITS'], 'rb') as f:
            self.night2visithistory = pickle.load(f)

        # Bin-space dependent function to get fields in bin
        if not self.hpGrid.is_azel:
            with open(gcfg['paths']['LOOKUP_DIR'] + '/' + f'nside{nside}_bin2fields_in_bin.json', 'r') as f:
                self.bin2fields_in_bin = json.load(f)
                self.bin2fields_in_bin = {int(k): v for k, v in self.bin2fields_in_bin.items()}
        else:
            self.bin2fields_in_bin = None

        self.base_global_feature_names = cfg['data']['global_features'].copy()
        self.base_bin_feature_names = cfg['data']['bin_features'].copy()
        self.global_feature_names, self.bin_feature_names, self.prenorm_expanded_bin_feature_names =\
            setup_feature_names(base_global_feature_names=cfg['data']['global_features'],
                                base_bin_feature_names=cfg['data']['bin_features'],
                                cyclical_feature_names=self.cyclical_feature_names,
                                nbins=self.nbins,
                                do_cyclical_norm=self.do_cyclical_norm,
                                grid_network=self._grid_network
                                )
        
        if self._grid_network is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network == 'single_bin_scorer':
            self.state_feature_names = self.global_feature_names
        
        self.global_pd_nightgroup = global_pd_nightgroup
        self.bin_pd_nightgroup = bin_pd_nightgroup

        self.max_nights = max_nights
        if max_nights is None:
            self.max_nights = self.global_pd_nightgroup.ngroups


        self.state_dim = cfg['data']['state_dim']
        self.bins_state_dim = cfg['data']['bin_state_dim']

        if self.include_bin_features:
            bins_state_shape = (self.nbins, self.bins_state_dim, )
        else:
            bins_state_shape = (0,)

        self.observation_space = gym.spaces.Dict(
            {
                "global_state": gym.spaces.Box(-2, 2, shape=(self.state_dim,), dtype=np.float32),
                "bins_state": gym.spaces.Box(-2, 2, shape=bins_state_shape, dtype=np.float32),
            }
        )

        # Define action space 
        self.action_space = gym.spaces.Dict(
            {
                "bin": gym.spaces.Discrete(self.nbins),
                "field_id": gym.spaces.Discrete(len(self.field2radec)),
                "filter": gym.spaces.Box(0., 1., dtype=np.float32)
            }
        )       
        # self.action_space = gym.spaces.Box(low=np.array([0, 0, 0.]), high=np.array([self.nbins, len(self.field2radec), 1.]), dtype=np.int32)

        self._global_state = np.zeros(self.state_dim, dtype=np.float32)
        self._bins_state = np.zeros(self.bins_state_dim, dtype=np.float32)
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

        self._init_to_first_state()
        state = self._get_state()
        info = self._get_info()
        return state, info

    def step(self, actions: dict):
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
        last_field_id = np.int32(self._field_id)

        # ------------------- Advance state ------------------- #
        self._update_state(actions)
        
        # ------------------- Calculate reward ------------------- #

        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)

        # -------------------- Start new night if last transition -----------------------#

        is_new_night = self._timestamp >= np.min([self._sunrise_time, self._night_final_timestamp])
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

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._night_idx = -1
        self._is_new_night = True
        self._start_new_night()
        self._update_action_mask(time=self._timestamp)
    
    def _start_new_night(self):
        self._night_idx += 1
        if self._night_idx >= self.max_nights:
            return

        # global features
        global_first_row = self.global_pd_nightgroup.head(1).iloc[self._night_idx]
        night = global_first_row['night']
        self._timestamp = global_first_row['timestamp']
        self._sunset_time = get_nautical_twilight(self._timestamp+10, 'set') # add 10 seconds just in case timestamp is exactly at twilight
        self._sunrise_time = get_nautical_twilight(self._timestamp+10, 'rise')
        self._night_final_timestamp = self.global_pd_nightgroup.tail(1).iloc[self._night_idx]['timestamp']
        self._night_first_timestamp = global_first_row['timestamp']
        self._field_id = global_first_row['field_id']
        if self._field_id != -1:
            logger.debug('FIRST FIELD IS NOT -1!!!')
        self._bin_num = global_first_row['bin']

        # Get field visit counts at start of night
        self._s_visits_cur = self.night2visithistory[night][self._fids].copy().astype(np.int32)
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        self._global_state = [global_first_row[feat_name] for feat_name in self.global_feature_names]

        if self.include_bin_features:
            # bin_feature_names = expand_feature_names_for_cyclic_norm(self.base_global_feature_names.copy(), self.cyclical_feature_names)
            bingroup = self.bin_pd_nightgroup
            first_row_in_night_bin = bingroup.head(1).iloc[self._night_idx]
            self._bins_state = np.array([first_row_in_night_bin[feat_name] for feat_name in self.bin_feature_names])

            # I think this is all wrapped up in the bin_df f
            if self._has_historical_features:
                global_night_df = self.global_pd_nightgroup.get_group(night)
                if self.hpGrid.is_azel:
                    # I think these features are already included in the offline dataset bin_df
                    pass
            #         az, el = ephemerides.equatorial_to_topographic(ra=self.field_radecs[:, 0], dec=self.field_radecs[:, 1], time=self._timestamp)
            #         bins = self.hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
            #         valid_mask = bins != None

            #         # Mask quantities whose associated field is below horizon
            #         v_bins = bins[valid_mask].astype(np.int32)
            #         v_survey_counts = self._s_visits_cur[valid_mask].astype(np.int32)
            #         v_night_counts = self._n_visits_cur[valid_mask].astype(np.int32)
            #         v_max_v = self._max_visits_arr[valid_mask].astype(np.int32)

            #         # Count total visible fields in each bin
            #         bin_count = np.bincount(v_bins, minlength=self.nbins)
            #         active_bins = bin_count > 0

            #         # Num Unvisited fields
            #         s_unvisited = np.bincount(v_bins, weights=(v_survey_counts == 0), minlength=self.nbins)
            #         n_unvisited = np.bincount(v_bins, weights=(v_night_counts == 0), minlength=self.nbins)
                    
            #         # Num Incomplete fields
            #         s_incomplete_mask = v_survey_counts < v_max_v
            #         s_incomplete = np.bincount(v_bins, weights=s_incomplete_mask, minlength=self.nbins)
            #         n_incomplete_mask = v_night_counts < v_max_v
            #         n_incomplete = np.bincount(v_bins, weights=n_incomplete_mask, minlength=self.nbins)

            #         # Save 
            #         self._s_unvisited = np.divide(s_unvisited, bin_count, where=active_bins)
            #         self._n_unvisited = np.divide(n_unvisited, bin_count, where=active_bins)
            #         self._s_incomplete = np.divide(s_incomplete, bin_count, where=active_bins)
            #         self._n_incomplete = np.divide(n_incomplete, bin_count, where=active_bins)

            #         # Min tiling
            #         unique_bins = np.where(active_bins)[0]
            #         s_tiling_all = v_survey_counts / v_max_v
            #         n_tiling_all = v_night_counts / v_max_v

            #         self._s_min_tiling = -.1 * np.ones(self.nbins)
            #         self._n_min_tiling = -.1 * np.ones(self.nbins)
            #         for b in unique_bins:
            #             mask = v_bins == b
            #             self._s_min_tiling[b] = np.min(s_tiling_all[mask])
            #             self._n_min_tiling[b] = np.min(n_tiling_all[mask])

                else: #radec action space
                    unique_field_ids, unique_field_counts = np.unique(global_night_df['field_id'][global_night_df['object'] != 'zenith'].to_numpy().astype(np.int32), return_counts=True)
                    unique_bin_ids, unique_bin_counts = np.unique(global_night_df['bin'][global_night_df['object'] != 'zenith'], return_counts=True)

                    self._night_field2nvisits = {int(fid): int(c) for fid, c in zip(unique_field_ids, unique_field_counts)}
                    self._night_bin2nvisits = {int(bid): int(c) for bid, c in zip(unique_bin_ids, unique_bin_counts)}

                    # Build bin2fields map
                    self._night_field_visits_counter = np.zeros(self.nfields)
                    # self._night_num_visits_tracking = np.zeros(self.nbins)
                    self._night_num_unvisited_fields_tracking = np.zeros(self.nbins)
                    self._night_num_incomplete_fields_tracking = np.zeros(self.nbins)
                    self._min_tiling_tracking = np.zeros(self.nbins)
                    
                    bins_arr = global_night_df['bin'].to_numpy(dtype=np.int32)
                    fields_arr = global_night_df['field_id'].to_numpy(dtype=np.int32)

                    bin2fields_in_bin = {}
                    for b, f in zip(bins_arr, fields_arr):
                        if f != -1:
                            bin2fields_in_bin.setdefault(int(b), set()).add(int(f))
                    
                    for bid, flist in bin2fields_in_bin.items():
                        self._night_num_unvisited_fields_tracking[bid] = len(flist)
                        self._night_num_incomplete_fields_tracking[bid] = len(flist)

            if self._grid_network == 'single_bin_scorer':
                A, B = self.nbins, self.bins_state_dim
                self._bins_state = np.array(self._bins_state).reshape((A, B))
        else:
            self._bins_state = np.array([])

        self._update_action_mask(self._timestamp)

    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._timestamp += self.time_between_obs
        bin_num, field_id, filter_wave = int(action['bin']), int(action['field_id']), float(action['filter'])

        self._bin_num = bin_num
        self._field_id = field_id
        self._n_visits_cur[field_id] += 1
        self._s_visits_cur[field_id] += 1

        self._global_state = self._update_global_features(field_id=self._field_id, filter_wave=filter_wave, timestamp=self._timestamp,
                                                          sunset_time=self._sunset_time, sunrise_time=self._sunrise_time
                                                          )
        self._bins_state = self._update_bin_features(timestamp=self._timestamp) if self.include_bin_features else np.array([])

        self._update_action_mask(self._timestamp)

    def _update_global_features(self, field_id, filter_wave, timestamp, sunset_time, sunrise_time):
        new_features = {}
        astro_time = Time(timestamp, format='unix', scale='utc')
        lst = astro_time.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        new_features['lst'] = lst.radian
        new_features['ra'], new_features['dec'] = self.field2radec[field_id]
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        
        cos_zenith = np.cos(90 * units.deg - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=timestamp)

        if sunrise_time == sunset_time:
            new_features['time_fraction_since_start'] = 0
        else:
            new_features['time_fraction_since_start'] = (timestamp - sunset_time) / (sunrise_time - sunset_time)

        new_features['filter_wave'] = filter_wave

        for feat_name in self.base_global_feature_names:
            if any(string in feat_name and 'bin' not in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                new_features.update({f'{feat_name}_cos': np.cos(new_features[feat_name])})
                new_features.update({f'{feat_name}_sin': np.sin(new_features[feat_name])})

        global_state_features = [new_features.get(feat, np.nan) for feat in self.global_feature_names]
        nan_feats = np.isnan(global_state_features)
        if any(nan_feats):
            nan_idxs = np.where(nan_feats == True)[0]
            for idx in nan_idxs:
                raise ValueError(f"Calculated nan value for global feature {self.global_feature_names[idx]}")


        return global_state_features
    
    def _update_bin_features(self, timestamp):
        
        features = {}
        current_ra, current_dec = self.field2radec[self._field_id]
        if self.hpGrid.is_azel:
            lons, lats = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=timestamp)
            features['az'], features['el'] = self.hpGrid.lon, self.hpGrid.lat
            features['ra'], features['dec'] = lons, lats
            current_lon, current_lat = ephemerides.equatorial_to_topographic(ra=current_ra, dec=current_dec, time=timestamp)
        else:
            lons, lats = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=timestamp)
            features['ra'], features['dec'] = self.hpGrid.lon, self.hpGrid.lat
            features['az'], features['el'] = lons, lats
            current_lon, current_lat = current_ra, current_dec
            
        features['angular_distance_to_pointing'] = self.hpGrid.get_angular_separations(lon=current_lon, lat=current_lat)
        features['ha'] = self.hpGrid.get_hour_angle(time=timestamp)
        features['airmass'] = self.hpGrid.get_airmass(timestamp)
        features['moon_distance'] = self.hpGrid.get_source_angular_separations('moon', time=timestamp)
        
        if self._has_historical_features:
            self._n_visits_cur[self._field_id] += 1
            self._s_visits_cur[self._field_id] += 1 
            # self._night_num_visits_tracking[self._bin_num] += 1
            # features['night_num_visits'] = self._night_num_visits_tracking.copy() 

            if not self.hpGrid.is_azel:
                nfields_in_bin = len(self.bin2fields_in_bin[self._bin_num])
                if self._n_visits_cur[self._field_id] == 1:
                    self._night_num_unvisited_fields_tracking[self._bin_num] -= 1
                    features['night_num_unvisited_fields'] = self._night_num_unvisited_fields_tracking.copy() / nfields_in_bin
                
                if self._night_field_visits_counter[self._field_id] == self.field2maxvisits[self._field_id]:
                    self._night_num_incomplete_fields_tracking[self._bin_num] -= 1
                    features['night_num_incomplete_fields'] = self._night_num_incomplete_fields_tracking.copy() / nfields_in_bin
                
                for bid in self.bin2fields_in_bin:

                    min_tiling = np.array([self._n_visits_cur[fid] for fid in self.bin2fields_in_bin[bid]]).min()
                    self._min_tiling_tracking[bid] = min_tiling
                features['night_min_tiling'] = self._min_tiling_tracking.copy() / self.max_tiling
            
            else:
                # Reset at each timestep since fields' bin memberships change over time
                az, el = ephemerides.equatorial_to_topographic(ra=self._ra_arr, dec=self._dec_arr, time=self._timestamp)
                bins = self.hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
                valid_mask = bins != None

                # Mask quantities whose associated field is below horizon
                v_bins = bins[valid_mask].astype(np.int32)
                v_survey_counts = self._s_visits_cur[valid_mask].astype(np.int32)
                v_night_counts = self._n_visits_cur[valid_mask].astype(np.int32)
                v_max_v = self._max_visits_arr[valid_mask].astype(np.int32)

                # Count total visible fields in each bin
                bin_count = np.bincount(v_bins, minlength=self.nbins)
                active_bins = bin_count > 0

                # Num Unvisited fields
                s_unvisited = np.bincount(v_bins, weights=(v_survey_counts == 0), minlength=self.nbins)
                n_unvisited = np.bincount(v_bins, weights=(v_night_counts == 0), minlength=self.nbins)
                
                # Num Incomplete fields
                s_incomplete_mask = v_survey_counts < v_max_v
                s_incomplete = np.bincount(v_bins, weights=s_incomplete_mask, minlength=self.nbins)
                n_incomplete_mask = v_night_counts < v_max_v
                n_incomplete = np.bincount(v_bins, weights=n_incomplete_mask, minlength=self.nbins)

                # Save 
                features['survey_num_unvisited_fields'] = np.divide(s_unvisited, bin_count, where=active_bins)
                features['night_num_unvisited_fields'] = np.divide(n_unvisited, bin_count, where=active_bins)
                features['survey_num_incomplete_fields'] = np.divide(s_incomplete, bin_count, where=active_bins)
                features['night_num_incomplete_fields'] = np.divide(n_incomplete, bin_count, where=active_bins)

                # Min tiling
                unique_bins = np.where(active_bins)[0]
                s_tiling_all = v_survey_counts / v_max_v
                n_tiling_all = v_night_counts / v_max_v

                s_min_tiling = -.1 * np.ones(self.nbins)
                n_min_tiling = -.1 * np.ones(self.nbins)
                for b in unique_bins:
                    mask = v_bins == b
                    s_min_tiling[b] = np.min(s_tiling_all[mask])
                    n_min_tiling[b] = np.min(n_tiling_all[mask])
                features['survey_min_tiling'] = s_min_tiling
                features['night_min_tiling'] = n_min_tiling
            # -------------------------------- old ---------------------------------- #
            # self._min_tiling_tracking = -1. * np.ones(self.nbins)
            # self._night_num_unvisited_fields_tracking = np.zeros(self.nbins)
            # self._night_num_incomplete_fields_tracking = np.zeros(self.nbins)

            # # Get bin membership of all fields at this time
            # fieldradecs = np.array([[ra, dec] for ra, dec in self.field2radec.values()])
            # _az, _el = ephemerides.equatorial_to_topographic(ra=fieldradecs[:, 0], dec=fieldradecs[:, 1], time=timestamp)
            # fieldazels = np.array([_az, _el]).T
            # field_bins = self.hpGrid.ang2idx(lon=fieldazels[:, 0], lat=fieldazels[:, 1])
            # above_horizon_fields_mask = field_bins != None
            # field_bins = field_bins[above_horizon_fields_mask]
            # bins_with_fields = np.unique(field_bins)

            # for bid in bins_with_fields:
            #     bin_mask = field_bins == bid
            #     # fids_in_bin = 
            #     fids = np.where(field_bins == bid)[0]
            #     num_fields_in_bin = len(fids_in_bin)
            #     f_visit_counts = np.array(([self._field_visit_counter[fid] for fid in fids_in_bin]))

            #     # Get min tiling in bin
            #     min_tiling = np.array([self._field_visit_counts[fid] for fid in fids]).min()
            #     self._min_tiling_tracking[bid] = min_tiling

            #     # Get number of unvisited fields
            #     n_unvisited = np.sum(self._field_visit_counts[fids] == 0)
            #     self._night_num_unvisited_fields_tracking[bid] = n_unvisited

            #     # Get number of incomplete fields
            #     n_incomplete = np.sum(self._field_visit_counts[fids] != np.array([self.field2maxvisits[fid] for fid in fids]))
            #     self._night_num_incomplete_fields_tracking[bid] = n_incomplete

            # features['min_tiling'] = self._min_tiling_tracking.copy()
            # features['night_num_unvisited_fields'] = self._night_num_unvisited_fields_tracking.copy()
            # features['night_num_incomplete_fields'] = self._night_num_incomplete_fields_tracking.copy()
            # -------------------------------- old ---------------------------------- #

        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    features[f'{feat_name}_cos'] = np.cos(features[feat_name])
                    features[f'{feat_name}_sin'] = np.sin(features[feat_name])

        for feat_name in expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names):
            f_row = features.get(feat_name, np.nan)
        bins_state = np.vstack([features.get(feat_name, np.nan) 
                                for feat_name in expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names)]) \
                                .T

        return bins_state

    def _get_state(self):
        global_state, bins_state = self._global_state, self._bins_state
        global_state_copy = global_state.copy()
        bins_state_copy = bins_state.copy()
        # state_normed = self._do_noncyclic_normalizations(state=state_copy)
        global_state_normed = normalize_noncyclic_features(
                            state=np.array(global_state_copy),
                            state_feature_names=self.state_feature_names,
                            max_norm_feature_names=self.max_norm_feature_names,
                            ang_distance_norm_feature_names=self.ang_distance_feature_names,
                            do_inverse_norm=self.do_inverse_norm,
                            do_max_norm=self.do_max_norm,
                            do_ang_distance_norm=self.do_ang_distance_norm,
                            fix_nans=True
                        )
        if self.include_bin_features:
            bins_state_normed = normalize_noncyclic_features(
                state=np.array(bins_state_copy),
                state_feature_names=expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names),
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True
            )
        else:
            bins_state_normed = np.array([])
        self._global_state = global_state_normed.astype(np.float32)
        self._bins_state = bins_state_normed.astype(np.float32)

        # logger.debug(f"Global state max, min {self._global_state.max()}, {self._global_state.min()}")
        # logger.debug(f"State above max: {np.where(self._global_state > 2)}")            
        # logger.debug(f"State below min: {np.where(self._global_state <-2)}")
        # if self.include_bin_features:
        #     logger.debug(f"Bins state max, min {self._bins_state.max()}, {self._bins_state.min()}")
        #     logger.debug(f"State above max: {np.where(self._bins_state > 2)}")            
        #     logger.debug(f"State below min: {np.where(self._bins_state <-2)}")
        return {"global_state": self._global_state, "bins_state": self._bins_state}

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy(), 
                'visited': self._n_visits_cur.copy(),
                'timestamp': self._timestamp,
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id),
                'valid_fields_per_bin': self._valid_fields_per_bin 
        }
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met, or, when all fields have been completely
        visited.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        all_nights_completed = self._night_idx >= self.max_nights
        all_fields_visited = all(np.array([self._s_visits_cur[fid] >= self.field2maxvisits[fid] for fid in self._fids]))
        terminated = all_nights_completed or all_fields_visited
        return terminated
    
    def _update_fields_in_bin(self, field_ids, mask_invalid_fields):
        fields_in_bin = field_ids[mask_invalid_fields]
        self._fields_in_bin = fields_in_bin

    def _get_action_mask(self, timestamp, field2nvisits, field_ids, ras, decs, hpGrid, visited):
        # Mask fields which are completed 
        mask_completed_fields = np.array([visited[fid] < field2nvisits[fid] for fid in field_ids], dtype=bool) #TODO can probably track visits without repeating this operation
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=ras, dec=decs, time=timestamp)
        # Mask fields below horizon
        mask_fields_below_horizon = fields_el > 0
        sel_valid_fields = mask_completed_fields & mask_fields_below_horizon
        self._sel_valid_fields = sel_valid_fields
        # Get bins which are below horizon, masking completed bins
        valid_fids = field_ids[sel_valid_fields]
        if hpGrid.is_azel:
            valid_field_bins = hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
        else:
            valid_field_bins = hpGrid.ang2idx(lon=ras[sel_valid_fields], lat=decs[sel_valid_fields])
        
        self._valid_fields_per_bin = defaultdict(list)
        action_mask = np.zeros(shape=self.nbins, dtype=bool)
        for fid, bin_idx in zip(valid_fids, valid_field_bins):
            if bin_idx is not None:
                b = int(bin_idx)
                action_mask[b] = True
                self._valid_fields_per_bin[b].append(fid)
        if 'filter' in self.bin_space:
            action_mask = np.repeat(action_mask[:, np.newaxis], NUM_FILTERS, axis=1).flatten()
        return action_mask

    def _get_slew_time(self, slew_time=None):
        if slew_time is not None:
            return slew_time
        else:
            raise NotImplementedError

    def _get_exposure_time(self, exp_time=None):
        if exp_time is not None:
            return exp_time
        else:
            raise NotImplementedError

    def _update_action_mask(self, time):
        # action mask is updated via self._visited (whether or not field is complete) and whether or not bin is above horizon
        self._action_mask = self._get_action_mask(timestamp=time, field2nvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)


class OfflineDECamEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, fields_list_path, dt, gcfg, cfg, max_nights=None, exp_time=90., slew_time=30., visit_history_path=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        
        # Assign static attributes
        self.first_night_dt = dt
        self.exp_time = exp_time
        self.slew_time = slew_time
        self.time_between_obs = exp_time + slew_time
        self.time_dependent_feature_substrs = gcfg['features']['TIME_DEPENDENT_FEATURE_NAMES']
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES']
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES']
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.include_bin_features = len(cfg['data']['additional_bin_features']) > 0
        self.bin_space = cfg['data']['bin_space']
        nside = cfg['data']['nside']
        self.hpGrid = None if cfg['data']['bin_method'] != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=(self.bin_space == 'azel'))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        if any(f in cfg['data']['additional_bin_features'] for f in ['night_num_visits', 'night_num_unvisited_fields', 'night_num_incomplete_fields']):
            self._has_historical_features = True
        else:
            self._has_historical_features = False

        with open(fields_list_path, 'r') as f:
            fields_list = json.load(f)

        fields_df = pd.DataFrame(fields_list)
        fields_df['field_id'] = pd.factorize(fields_df['object'])[0]

        if visit_history_path is None:
            self.visit_history = [0 for fid in fields_df['field_id'].values]
        else:
            with open(visit_history_path, 'rb') as f:
                self.visit_history = pickle.load(f)

        self.field2nvisits = {int(fid): seqtot for fid, seqtot in zip(fields_df['field_id'].values, fields_df['seqtot'])}
        self.field2radec = {int(fid): np.array([ra, dec]) for fid, ra, dec in zip(fields_df['field_id'].values, fields_df['RA'], fields_df['dec'])}
        self.field_ids = np.array(list(self.field2radec.keys()), dtype=np.int32)
        self.field_radecs = np.array(list(self.field2radec.values()))
        self.nfields = len(self.field2nvisits)

        # Bin-space dependent function to get fields in bin
        if not self.hpGrid.is_azel:
            lon, lat = fields_df['RA'], fields_df['dec']
            fields_df['bin'] = self.hpGrid.ang2idx(lon=lon, lat=lat)
            self.bin2fields_in_bin = {int(bin_id): g['field_id'].values.tolist() for bin_id, g in fields_df.groupby('bin')}
        else:
            self.bin2fields_in_bin = None

        self.base_global_feature_names, self.base_bin_feature_names, self.base_feature_names, self.global_feature_names, \
            self.bin_feature_names, self.state_feature_names, self.prenorm_bin_feature_names \
            = setup_feature_names(include_default_features=True,
                                  additional_global_features=cfg['data']['additional_global_features'],
                                  bin_feature_names=cfg['data']['additional_bin_features'],
                                  global_feature_names=gcfg['features']['DEFAULT_GLOBAL_FEATURE_NAMES'], 
                                  cyclical_feature_names=self.cyclical_feature_names,
                                  hpGrid=self.hpGrid, do_cyclical_norm=self.do_cyclical_norm,
                                  grid_network=self._grid_network
                                  )

        self.max_nights = max_nights if max_nights is not None else 1
            
        self.state_dim = cfg['data']['state_dim']
        self.bins_state_dim = cfg['data']['bin_state_dim']

        self.observation_space = gym.spaces.Dict(
            {
                "global_state": gym.spaces.Box(-2, 2, shape=(self.state_dim,), dtype=np.float32),
                "bins_state": gym.spaces.Box(-2, 2, shape=(self.nbins, self.bins_state_dim,), dtype=np.float32),
            }
        )
        # Define action space        
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.nbins, len(self.field2radec)]), dtype=np.int32)

        self._global_state = np.zeros(self.state_dim, dtype=np.float32)
        self._bins_state = np.zeros(self.bins_state_dim, dtype=np.float32)
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
        last_field_id = np.int32(self._field_id)

        # ------------------- Advance state ------------------- #
        self._update_state((action, field_id))
        
        # ------------------- Calculate reward ------------------- #

        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)

        # -------------------- Start new night if last transition -----------------------#

        is_new_night = self._timestamp >= np.min([self._sunrise_time, self._night_final_timestamp])
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

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._visited = np.zeros(np.max(self.field_ids) + 1)
        self._bins_visited = np.zeros(self.nbins)
        self._night_idx = -1
        self._is_new_night = True
        self._start_new_night()
        self._update_action_mask(time=self._timestamp)
    
    def _start_new_night(self):
        pass

    def get_zenith_state(year, month, day, hour, minute, sec, t_sunrise, t_sunset):
        datetime = datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
        night = datetime(year, month, day, tzinfo=timezone.utc)
        t0 = datetime.timestamp()
        blanco = ephemerides.blanco_observer(time=t0)

        zenith = {}
        zenith['ra'], zenith['dec'] = blanco.lat, blanco.lon
        zenith['az'], zenith['el'] = 0, np.pi/2
        zenith['airmass'] = 1
        zenith['ha'] = 0
        zenith['object'] = 'zenith'
        zenith['field_id'] = -1
        zenith['bin'] = -1
        zenith['datetime'] = datetime
        zenith['night'] = night
        zenith['sun_ra'], zenith['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=t0)
        zenith['sun_az'], zenith['sun_el'] = ephemerides.equatorial_to_topographic(ra=zenith['sun_ra'], dec=zenith['sun_dec'], time=t0)
        zenith['moon_ra'], zenith['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=t0)
        zenith['moon_az'], zenith['moon_el'] = ephemerides.equatorial_to_topographic(ra=zenith['moon_ra'], dec=zenith['moon_dec'], time=t0)
        zenith['time_fraction_since_start'] = (t0 - t_sunset) / (t_sunrise - t_sunset)

    def _update_global_features(self, field_id, timestamp, sunset_time, sunrise_time):
        new_features = {}
        astro_time = Time(timestamp, format='unix', scale='utc')
        lst = astro_time.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        new_features['lst'] = lst.radian
        new_features['ra'], new_features['dec'] = self.field2radec[field_id]
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        
        cos_zenith = np.cos(90 * units.deg - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=timestamp)

        if sunrise_time == sunset_time:
            new_features['time_fraction_since_start'] = 0
        else:
            new_features['time_fraction_since_start'] = (timestamp - sunset_time) / (sunrise_time - sunset_time)

        new_features['bins_visited_in_night'] = len(set(self._bins_visited))

        for feat_name in self.base_global_feature_names:
            if any(string in feat_name and 'bin' not in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                new_features.update({f'{feat_name}_cos': np.cos(new_features[feat_name])})
                new_features.update({f'{feat_name}_sin': np.sin(new_features[feat_name])})
        global_state_features = [new_features.get(feat, np.nan) for feat in self.global_feature_names]
        return global_state_features
    
    def _update_bin_features(self, timestamp):
        
        features = {}
        if self.hpGrid.is_azel:
            lons, lats = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=timestamp)
            features['az'], features['el'] = self.hpGrid.lon, self.hpGrid.lat
            features['ra'], features['dec'] = lons, lats
        else:
            lons, lats = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=timestamp)
            features['ra'], features['dec'] = self.hpGrid.lon, self.hpGrid.lat
            features['az'], features['el'] = lons, lats
            
        features['ha'] = self.hpGrid.get_hour_angle(time=timestamp)
        features['airmass'] = self.hpGrid.get_airmass(timestamp)
        features['moon_distance'] = self.hpGrid.get_source_angular_separations('moon', time=timestamp)
        current_ra, current_dec = self.field2radec[self._field_id]
        features['angular_distance_to_pointing'] = self.hpGrid.get_angular_separations(lon=current_ra, lat=current_dec)
        
        if self._has_historical_features:
            self._night_field_visits_counter[self._field_id] += 1
            self._night_num_visits_tracking[self._bin_num] += 1

            if self._night_field_visits_counter[self._field_id] == 1:
                self._night_num_unvisited_fields_tracking[self._bin_num] -= 1
            if self._night_field_visits_counter[self._field_id] == self.field2nvisits[self._field_id]:
                self._night_num_incomplete_fields_tracking[self._bin_num] -= 1

        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    features[f'{feat_name}_cos'] = np.cos(features[feat_name])
                    features[f'{feat_name}_sin'] = np.sin(features[feat_name])

        bins_state = np.vstack([features.get(feat_name, np.nan) 
                                for feat_name in expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names)]) \
                                .T

        return bins_state

    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._timestamp += self.time_between_obs
        bin_num, field_id, filter = int(action['bin']), int(action['field_id']), np.float32(action['filter'][0])

        self._bin_num = bin_num
        self._field_id = field_id
        self._visited[field_id] += 1
        self._bins_visited[self._bin_num] += 1

        self._global_state = self._update_global_features(field_id=self._field_id, timestamp=self._timestamp,
                                                          sunset_time=self._sunset_time, sunrise_time=self._sunrise_time
                                                          )
        self._bins_state = self._update_bin_features(timestamp=self._timestamp) if self.include_bin_features else []

        self._update_action_mask(self._timestamp)

    def _get_state(self):
        global_state, bins_state = self._global_state, self._bins_state
        global_state_copy = global_state.copy()
        bins_state_copy = bins_state.copy()
        # state_normed = self._do_noncyclic_normalizations(state=state_copy)
        global_state_normed = normalize_noncyclic_features(
                            state=np.array(global_state_copy),
                            state_feature_names=self.state_feature_names,
                            max_norm_feature_names=self.max_norm_feature_names,
                            ang_distance_norm_feature_names=self.ang_distance_feature_names,
                            do_inverse_norm=self.do_inverse_norm,
                            do_max_norm=self.do_max_norm,
                            do_ang_distance_norm=self.do_ang_distance_norm,
                            fix_nans=True
                        )
        if self.include_bin_features:
            bins_state_normed = normalize_noncyclic_features(
                state=np.array(bins_state_copy),
                state_feature_names=expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names),
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True
            )
        else:
            bins_state_normed = np.array([])
        self._global_state = global_state_normed.astype(np.float32)
        self._bins_state = bins_state_normed.astype(np.float32)
        
        return {"global_state": self._global_state, "bins_state": self._bins_state}

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy(), 
                'visited': self._visited.copy(),
                'bins_visited': self._bins_visited.copy(),
                'timestamp': self._timestamp,
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id),
                'valid_fields_per_bin': self._valid_fields_per_bin 
        }
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met, or, when all fields have been completely
        visited.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        all_nights_completed = self._night_idx >= self.max_nights
        all_fields_visited = all(np.array([self._visited[fid] >= self.field2nvisits[fid] for fid in self.field_ids]))
        terminated = all_nights_completed or all_fields_visited
        return terminated
    
    def _update_fields_in_bin(self, field_ids, mask_invalid_fields):
        fields_in_bin = field_ids[mask_invalid_fields]
        self._fields_in_bin = fields_in_bin

    def _get_action_mask(self, timestamp, field2nvisits, field_ids, field_radecs, hpGrid, visited):
        # Mask fields which are completed 
        mask_completed_fields = np.array([visited[fid] < field2nvisits[fid] for fid in field_ids], dtype=bool) #TODO can probably track visits without repeating this operation
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=field_radecs[:, 0], dec=field_radecs[:, 1], time=timestamp)
        # Mask fields below horizon
        mask_fields_below_horizon = fields_el > 0
        sel_valid_fields = mask_completed_fields & mask_fields_below_horizon
        self._sel_valid_fields = sel_valid_fields
        # Get bins which are below horizon, masking completed bins
        valid_fids = field_ids[sel_valid_fields]
        if hpGrid.is_azel:
            valid_field_bins = hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
        else:
            valid_field_bins = hpGrid.ang2idx(lon=field_radecs[sel_valid_fields, 0], lat=field_radecs[sel_valid_fields, 1])
        
        self._valid_fields_per_bin = defaultdict(list)
        action_mask = np.zeros(shape=self.nbins, dtype=bool)
        for fid, bin_idx in zip(valid_fids, valid_field_bins):
            if bin_idx is not None:
                b = int(bin_idx)
                action_mask[b] = True
                self._valid_fields_per_bin[b].append(fid)
                
        return action_mask

    def _get_slew_time(self, slew_time=None):
        if slew_time is not None:
            return slew_time
        else:
            raise NotImplementedError

    def _get_exposure_time(self, exp_time=None):
        if exp_time is not None:
            return exp_time
        else:
            raise NotImplementedError

    def _update_action_mask(self, time):
        # action mask is updated via self._visited (whether or not field is complete) and whether or not bin is above horizon
        self._action_mask = self._get_action_mask(timestamp=time, field2nvisits=self.field2nvisits, field_ids=self.field_ids, field_radecs=self.field_radecs, 
                                                  hpGrid=self.hpGrid, visited=self._visited)

