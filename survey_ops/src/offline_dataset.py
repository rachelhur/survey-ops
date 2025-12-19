import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from collections import defaultdict

from survey_ops.utils import units
from survey_ops.utils import ephemerides
import healpy as hp

import pandas as pd
import json

def reward_func_v0():
    raise NotImplementedError

def save_schedule_for_video(outdir, df, return_outputs=False):
    # def produce_schedule_for_video(outdir, id2radec):
    field_filepath = outdir + 'fields2radec.json' # field id to ra_dec
    schedule_filepath = outdir + 'true_schedule.csv' # keys time and field_id

    timestamps = df['timestamp'].values.tolist()
    ra = df['ra'].values
    dec = df['dec'].values

    id2radec = {i: np.float64([ra, dec]).tolist() for i, (ra, dec) in enumerate(zip(ra, dec))}
    radec2id = {tuple(v): k for k, v in id2radec.items()}  
    # save field_to_radec
    with open(field_filepath, 'w') as f:
        json.dump(id2radec, f, indent=2)
    
    # save time, field_id, next_field_id
    field_ids = [radec2id[(ra_, dec_)] for ra_, dec_ in zip(ra, dec)]
    data = {'time': timestamps, 'field_id': field_ids}
    schedule_df = pd.DataFrame(data)
    schedule_df.to_csv(schedule_filepath, index=False)

    if return_outputs:
        return id2radec, schedule_df


class OfflineDECamDataset(torch.utils.data.Dataset):
    def __init__(self, 
                df: pd.DataFrame, 
                specific_years: list = None,
                specific_months: list = None,
                specific_days: list = None,
                specific_filters: list = None,
                binning_method = 'healpix',
                bin_space = 'radec',
                nside=None,
                num_bins_1d = None,
                additional_pointing_features = [],
                additional_bin_features=[],
                include_default_features=True,
                include_bin_features=True,
                ):
        assert binning_method in ['uniform_grid', 'healpix'], 'bining_method must be uniform_grid or healpix'
        assert (binning_method == 'uniform_grid' and num_bins_1d is not None) or (binning_method == 'healpix' and nside is not None), 'num_bins_1d must be specified for uniform_grid and nside must be specified for healpix'

        # Initialize healpix grid if binning_method is healpix
        self.hpGrid = None if binning_method != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=(bin_space == 'azel'))

        # Any experiment will likely have these state features
        required_point_features = ['ra', 'dec', 'az', 'el', 'airmass', 'ha', 'sun_ra', 'sun_dec', 'sun_az', 'sun_el', 'moon_ra', 'moon_dec', 'moon_az', 'moon_el', 'time_fraction_since_start', 'time_seconds_till_end']
        required_bin_features = ['ha', 'airmass', 'ang_dist_to_moon', 'ra', 'dec', 'az', 'el']
        self.periodic_feature_names = ['ra', 'az', 'ha']

        if not include_default_features:
            required_point_features = []
            required_bin_features = []

        # State features names and lookup table for features --> state index
        self.pointing_feature_names = required_point_features + additional_pointing_features
        if include_bin_features:
            bin_feature_names = required_bin_features + additional_bin_features
            bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}' for bin_feat in bin_feature_names] for bin_num in range(self.hpGrid.npix)])
            self.bin_feature_names = bin_feature_names.flatten().tolist()
        else:
            self.bin_feature_names = []

        self.base_state_feature_names = self.pointing_feature_names + self.bin_feature_names
        self.base_featname2idx = {name: i for i, name in enumerate(self.base_state_feature_names)}

        self.state_feature_names = self._expand_feature_names_for_normalization(feature_names=self.base_state_feature_names)
        self.featname2idx = {name: i for i, name in enumerate(self.state_feature_names)}

        # Process dataframe
        df = self._process_dataframe(df, specific_years=specific_years, specific_months=specific_months, specific_days=specific_days, specific_filters=specific_filters)
        self._df = df # Save for diagnostics

        # Set dataset-wide (across observation nights) attributes
        self.num_transitions = len(df) # size of dataset
        if binning_method == 'uniform_grid':
            self.num_actions = int(num_bins_1d**2)
        elif binning_method == 'healpix':
            self.num_actions = self.hpGrid.npix
            
        # Save mappings
        self.field2idx = {field_name: i for i, field_name in enumerate(set(df.object))}

        # Save night dates, total number of nights in dataset, and number of obs per night
        self.unique_nights = df.groupby('night').groups.keys()
        self.n_nights = df.groupby('night').ngroups
        self.n_obs_per_night = df.groupby('night').size() # nights have different numbers of observations

        # Construct Transitions
        states, next_states = self._construct_states(df=df, include_bin_features=include_bin_features)
        actions = self._construct_actions(next_states, num_bins_1d=num_bins_1d, binning_method=binning_method, bin_space=bin_space)
        rewards = self._construct_rewards(df)
        dones = np.zeros(self.num_transitions, dtype=bool) # False unless last observation of the night
        self._done_indices = np.where(states[:, 0] == 0)[0][1:] - 1
        dones[self._done_indices] = True
        dones[-1] = True
        action_masks = self._construct_action_masks(timestamps=df['timestamp'].values)

        # Save Transitions as tensors
        self.states = torch.tensor(states, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.int32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.bool)
        self.action_masks = torch.tensor(action_masks, dtype=torch.bool)
        
        # Set dimension of observation
        self.obs_dim = self.states.shape[-1]

        # Normalize states
        self._normalize_states()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        transition = (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.action_masks[idx]
        )
        return transition
    
    def _expand_feature_names_for_normalization(self, feature_names):
        # periodic vars first
        periodic_names = ['ra', 'az', 'ha']
        feature_names = [
            element 
            for feat_name in feature_names
            for element in ([feat_name + '_cos', feat_name + '_sin'] 
                            if any(string in feat_name and 'frac' not in feat_name for string in periodic_names)
                            else [feat_name])
        ]
        return feature_names

    def _normalize_states(self):
        # periodic features
        # periodic_features_names = ['ra', 'az']
        # cols_with_
        # cos_features = torch.cos(self.states[:, periodic_feature_mask])
        # sin_features = torch.sin(self.states[:, periodic_feature_mask])
        periodic_mask = np.array( [any(string in feat_name and 'frac' not in feat_name for string in self.periodic_feature_names) for feat_name in self.featname2idx.keys()] )

        self.means = torch.mean(self.next_states, axis=0)
        self.stds = torch.std(self.next_states, axis=0)
        self.next_states[:, ~periodic_mask] = ((self.next_states - self.means) / self.stds)[:, ~periodic_mask]

        mask_null = self.states == 0
        new_states = self.states.clone()

        new_states[:, ~periodic_mask] = ((new_states - self.means) / self.stds)[:, ~periodic_mask]
        new_states[mask_null] = 0.
        self.states = new_states

    def _relabel_mislabelled_objects(self, df):
        """Renames object columns with 'object_name (outlier)' if they are outside of a certain cutoff from the median RA/Dec.

        Args
        ----
        df (pd.DataFrame): The dataframe with object names and RA/Dec positions.

        Returns
        -------
        df_relabelled (pd.DataFrame): The dataframe with relabelled objects.
        """

        object_radec_df = df[['object', 'ra', 'dec']]
        object_radec_groups = object_radec_df.groupby('object')
        df_relabelled = df.copy(deep=True)

        outlier_indices = []
        for obj_name, g in object_radec_groups:
            # print(g.index)

            cutoff_deg = 3
            median_ra = g.ra.median()
            delta_ra = g.ra - median_ra
            delta_ra_shifted = np.remainder(delta_ra + 180, 360) - 180
            mask_outlier_ra = np.abs(delta_ra_shifted) > cutoff_deg

            median_dec = g.dec.median()
            delta_dec = g.dec - median_dec
            delta_dec_shifted = np.remainder(delta_dec + 180, 360) - 180
            mask_outlier_dec = np.abs(delta_dec_shifted) > cutoff_deg

            mask_outlier = mask_outlier_ra | mask_outlier_dec

            if np.count_nonzero(mask_outlier) > 0:
                indices = g.index[mask_outlier].values
                outlier_indices.extend(indices)

        df_relabelled.loc[outlier_indices, 'object'] = [f'{obj_name} (outlier)' for obj_name in df.loc[outlier_indices, 'object'].values]
        return df_relabelled

    def _process_dataframe(self, df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
        """Processes and filters the dataframe. Adds columns that we want to include in current pointing state features"""
        # Add column which indicates observing night (noon to noon)
        df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()

        # Get observations for specific years, days, filters, etc.
        if specific_years is not None:
            df = df[df['night'].dt.year.isin(specific_years)]
        if specific_months is not None:
            df = df[df['night'].dt.month.isin(specific_months)]
        if specific_days is not None:
            df = df[df['night'].dt.day.isin(specific_days)]
        if specific_filters is not None:
            df = df[df['filter'].isin(specific_filters)]

        # Remove observations in 1970 - what are these?
        df = df[df['night'].dt.year != 1970]
        assert len(df) > 0, "No observations found for the specified year/month/day/filter selections."
        
        # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
        df = self._relabel_mislabelled_objects(df)

        # Add timestamp col
        utc = pd.to_datetime(df['datetime'], utc=True)
        timestamps = (utc.astype('int64') // 10**9).values
        df['timestamp'] = timestamps
        
        # Sort df by timestamp
        df = df.sort_values(by='timestamp')
        
        # Get time dependent features
        timestamps = df['timestamp'].values
        for idx, time in enumerate(timestamps):
            sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
            df.loc[idx, ['sun_ra', 'sun_dec']] = sun_ra, sun_dec
            df.loc[idx, ['sun_az', 'sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec)

            moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
            df.loc[idx, ['sun_ra', 'moon_dec']] = moon_ra, moon_dec
            df.loc[idx, ['moon_az', 'moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

        # Add features with special calculations 
        df['el'] = 90 - df['zd']
        
        # Add 
        for feat_name in self.pointing_feature_names:
            if feat_name in df.columns:
                continue
            
        # Get 
        df['time_fraction_since_start'] = df.groupby('night')['timestamp'].transform(lambda x: (x - x.values[0]) / (x.values[-1] - x.values[0] if len(x) > 1 else 0))
        df['time_seconds_since_start'] = df.groupby('night')['timestamp'].transform(lambda x: x - x.min())



        # Add bin column to dataframe
        self.field2bin = {field_name: self.hpGrid.ang2idx(lon=ra * units.deg, lat=dec * units.deg) for field_name, (ra, dec) in zip(df.object, zip(df['ra'].values, df['dec'].values))}
        df['bin'] = df['object'].map(self.field2bin)

        # Ensure all data are 32-bit precision before training
        for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
            cols = df.select_dtypes(include=[str_bit]).columns
            df[cols] = df[cols].astype(np_bit)

        return df

    def _get_point_features(self, df):
        """
        Constructs state and next_states for all transitions.
        Inserts a "null" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # Pointing features already in DECam data
        # pointing_states = np.zeros(shape=(self.num_transitions, len(self.pointing_feature_names)))
        # feats_in_df = ['ra', 'dec', 'az', 'airmass', 'ha']

        # # periodic_feat_names = ['ra', 'az', 'ha']
        # pointing_states[:, :len(feats_in_df)] = df[feats_in_df].to_numpy()
        # pointing_states[:, self.base_featname2idx['el']] = 90 - pointing_states[:, self.base_featname2idx['zd']]

        # # Get sun and moon positions
        # timestamps = df['timestamp'].values
        # for idx, time in enumerate(timestamps):
        #     sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
        #     moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
        #     pointing_states[idx, self.base_featname2idx['sun_ra']] = sun_ra
        #     pointing_states[idx, self.base_featname2idx['sun_dec']] = sun_dec
        #     pointing_states[idx, self.base_featname2idx['moon_ra']] = moon_ra
        #     pointing_states[idx, self.base_featname2idx['moon_dec']] = moon_dec
        #     pointing_states[idx, self.base_featname2idx['sun_az']], pointing_states[idx, self.base_featname2idx['sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)
        #     pointing_states[idx, self.base_featname2idx['moon_az']], pointing_states[idx, self.base_featname2idx['moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

        # # Get time related features
        # groups = df.groupby('night')
        # idx0 = df.head(1).index # indices do not necessarily start at 0
        # for dt, g_df in groups:
        #     indices = g_df.index - idx0
        #     times = g_df['timestamp'].values
        #     t_start = times[0]
        #     t_end = times[-1]
        #     t_tot = t_end - t_start
        #     if t_end == t_start: # only one observation that night
        #         pointing_states[indices, self.base_featname2idx['time_fraction_since_start']] = 0
        #         pointing_states[indices, self.base_featname2idx['time_seconds_till_end']] = 0
        #     else:
        #         pointing_states[indices, self.base_featname2idx['time_fraction_since_start']] = (times - t_start) / t_tot
        #         pointing_states[indices, self.base_featname2idx['time_seconds_till_end']] = t_tot - (times - t_start)

        # Normalize states
        new_cols = {}
        for feat_name, column_vals in zip(self.base_featname2idx, pointing_states.T):
            # Periodic vars
            if any(string in feat_name and 'frac' not in feat_name for string in self.periodic_feature_names):
                new_cols.update({feat_name: np.array([np.cos(column_vals * units.deg),  np.sin(column_vals * units.deg)]).T})
        
        for i, (col_name, cs_cols) in enumerate(new_cols.items()):
            i_col_orig = self.base_featname2idx[col_name]
            left_arr = pointing_states[:, :i_col_orig + i]
            right_arr = pointing_states[:, i_col_orig + i + 1:]
            pointing_states = np.concatenate([left_arr, cs_cols, right_arr], axis=1)

        # "Next States" are just all observations
        next_pointing_states = pointing_states.copy()
        
        # "States" require inserting rows of 0's before first observation of each night, and deleting last observation
        night_end_indices = groups.tail(1).index - idx0
        pointing_states[night_end_indices[:-1]] = 0 # Replace last observation of each night with 0's, except last observation of entire self
        pointing_states = pointing_states[:-1, :] # remove last observation of entire self
        zero_row = np.zeros_like(pointing_states[0]) # insert row of 0s in front of first observation
        pointing_states = np.vstack([zero_row, pointing_states])

        return pointing_states, next_pointing_states
        
    def _get_bin_features(self, df):
        timestamps = df['timestamp'].values
        hour_angles = np.empty(shape=(len(timestamps), self.hpGrid.npix))
        airmasses = np.empty_like(hour_angles)
        moon_dists = np.empty_like(hour_angles)
        
        for i, time in enumerate(timestamps):
            hour_angles[i] = self.hpGrid.get_hour_angle(time=time)
            airmasses[i] = self.hpGrid.get_airmass(time)
            moon_dists[i] = self.hpGrid.get_source_angular_separations('moon', time=time)
            
        stacked = np.stack([hour_angles, airmasses, moon_dists], axis=2)
        bin_states = stacked.reshape(len(hour_angles), -1)

        # "Next States" are just all observations
        next_bin_states = bin_states.copy()
        
        groups = df.groupby('night')
        idx0 = df.head(1).index # indices do not necessarily start at 0

        # "States" require inserting rows of 0's before first observation of each night, and deleting last observation
        night_end_indices = groups.tail(1).index - idx0
        bin_states[night_end_indices[:-1]] = 0 # Replace last observation of each night with 0's, except last observation of entire self
        bin_states = bin_states[:-1, :] # remove last observation of entire self
        zero_row = np.zeros_like(bin_states[0]) # insert row of 0s in front of first observation
        bin_states = np.vstack([zero_row, bin_states])
        
        return bin_states, next_bin_states
    
    def _construct_states(self, df, include_bin_features):
        pointing_states, next_pointing_states = self._get_point_features(df=df)
        if include_bin_features:
            bin_states, bin_next_states = self._get_bin_features(df=df)
            states = np.concatenate((pointing_states, bin_states), axis=1)
            next_states = np.concatenate((next_pointing_states, bin_next_states), axis=1)
            return states, next_states
        return pointing_states, next_pointing_states
    
    def _construct_actions(self, next_states, df, bin_space='radec', binning_method='healpix', num_bins_1d=None):
        assert bin_space in ['radec', 'azel'], 'bin_space must be radec or azel'
        assert binning_method in ['uniform_grid', 'healpix'], 'bining_method must be uniform_grid or healpix'

        if binning_method == 'healpix':
            if self.hpGrid.is_azel:
                lon, lat = df.az.values, df.el.values
            else:
                lon, lat = df.ra.values, df.dec.values
            indices = self.hpGrid.ang2idx(lon=lon*units.deg, lat=lat*units.deg)
            return indices
        
        elif binning_method == 'uniform_grid' and bin_space == 'azel':
            az_idx = self.featname2idx['az']
            el_idx = self.featname2idx['el']
            az_edges = np.linspace(0, 360, num_bins_1d + 1, dtype=np.float32)
            # az_centers = az_edges[:-1] + (az_edges[1] - az_edges[0])/2
            el_edges = np.linspace(0, 90, num_bins_1d + 1, dtype=np.float32)
            # az_centers = el_edges[:-1] + (el_edges[1] - el_edges[0])/2

            i_x = np.digitize(next_states[:, az_idx], az_edges).astype(np.int32) - 1
            i_y = np.digitize(next_states[:, el_idx], el_edges).astype(np.int32) - 1
            bin_ids = i_x + i_y * (num_bins_1d)
            self.az_edges = az_edges
            self.el_edges = el_edges
            
            id2azel = defaultdict(list)
            for az, el, bin_id in zip(next_states[:, az_idx], next_states[:, el_idx], bin_ids):
                id2azel[bin_id].append((az, el))            
            self.id2azel = dict(sorted(id2azel.items()))

            id2radec = defaultdict(list)
            for ra, dec, bin_id in zip(df['ra'].values, df['dec'].values, bin_ids):
                id2radec[bin_id].append((ra, dec))
            self.id2radec = dict(sorted(id2radec.items()))

            return bin_ids
        
        else:
            raise NotImplementedError
    
    def _get_unique_fields_dict(self, df, radec_width=.05):
        # """Constructs uniform binning across RA/Dec, then decides that all observations within this width are observing the same field.

        # Args
        # ----
        # radec_width (float): the bin width in degrees
        # """
        # radec_width = .05
        # ra_edges = np.arange(np.min(df['ra'].values), np.max(df['ra'].values), step=radec_width, dtype=np.float32)
        # dec_edges = np.arange(np.min(df['dec'].values), np.max(df['dec'].values), step=radec_width, dtype=np.float32)

        # i_x = np.digitize(df['ra'].values, ra_edges).astype(np.int32) - 1
        # i_y = np.digitize(df['dec'].values, dec_edges).astype(np.int32) - 1
        # bin_ids = i_x + i_y * (num_bins_1d)
        raise NotImplementedError

    def _construct_rewards(self, df):
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        rewards = df['teff'].values
        rewards -= np.min(rewards)
        rewards /= np.max(rewards)
        return rewards

    def _construct_action_masks(self, timestamps=None):
        # given timestamp, determine bins which are outside of observable range
        return np.ones((self.num_transitions, self.num_actions), dtype=bool)
    
    def get_dataloader(self, batch_size, num_workers, pin_memory):
        loader = DataLoader(
            self,
            batch_size,
            sampler=RandomSampler(
                self,
                replacement=True,
                num_samples=10**10,
            ),
            drop_last=True, # drops last non-full batch
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return loader

class TelescopeDatasetv0:
    """
    A dataset wrapper converting a nightly telescope schedule into a structured transition dataset.
    First version of dataset class for telescope data. Designed for behavior cloning where the original
     schedule is mimicked exactly (no generalization).
    
    Attributes
    ----------
        obs (ndarray): The full set of observations; has shape (obs_dim, n_nights, n_transitions) and dtype float32
            to be input into the q-network.
        actions (ndarray): The array of field ids of shape (n_nights, n_transitions) which indicate the
            next field id to be observed
        next_obs (ndarray): The next observations given obs and action; shape (obs_dim, n_nights, n_transitions)
        rewards (ndarray): The reward of doing next_obs given obs and action; shape (n_nights, n_transitions)
        dones (ndarray): Boolean array indicating whether or not this action terminates the episode; 
            shape (n_nights, n_transitions)
        action_masks (ndarray): Boolean array indicating valid possible actions

        device (str): The device to place tensors on ('cpu' or 'cuda')
        normalize_obs (bool): Whether to normalize observations.
        norm (ndarray): Normalization constants applied to observations
        obs_dim (int): Size of observations for a single transition
        num_actions (int): Size of action space
        reward_func (Callable | None): User-provided reward_function

        _radec (ndarray): RA/Dec positions for each scheduled field
        _obs_indices (ndarray): A time index for each observation made in a single night (starts at 0)
        _schedule_field_id (ndarray): The actual, time-ordered schedule; has shape (n_nights, n_obs_in_night) where
            n_obs_in_night = n_transitions + 1
        _unique_field_ids (ndarray): An array of the unique field ids in increasing order
        _n_obs_per_night (int): Number of observations in each night
        _n_transitions (int): Number of transitions in dataset
        _id2pos (dict): A dictionary of field_id as keys and values as coordinates

        _

    """
    def __init__(self, schedule, id2pos, reward_func=None, normalize_obs=True, device='cpu'):
        """
        Args
        ---- 
            schedule (pd.Dataframe): Pandas dataframe with columns 'field_id' and 'next_field_id'
            id2pos (dict): Mapping from field ID to its (RA, Dec) coordinates
            reward_func (Callable | None): Optional custom reward ufnction. Should have signature: ``reward_func(obs, action, next_obs) -> float or array``
            normalize_obs (bool): Whether or not to normalize observations
            device (str): Device where tensors will be moved ('cuda' or 'CPU')
        """
        self._schedule_field_ids = schedule.field_id.values.astype(np.float32)
        if len(self._schedule_field_ids.shape) == 1:
            self._schedule_field_ids = self._schedule_field_ids[np.newaxis, :]
        assert len(self._schedule_field_ids.shape) == 2

        # vars used internally to help calculate transition variables
        self._radec = np.array([id2pos[field_id] for field_id in schedule.field_id.values], dtype=np.float32)
        self._unique_field_ids, counts = np.unique(self._schedule_field_ids, return_counts=True)
        self._max_visits = np.int32(np.max(counts))
        self._nfields = np.int32(len(self._unique_field_ids))
        self._n_obs_per_night = np.int32(self._schedule_field_ids.shape[-1])
        self._n_transitions = np.int32(len(self._radec))
        self._id2pos = id2pos
        self.device = device
        self.normalize_obs = normalize_obs
        self.norm = 1
        if self.normalize_obs:
            self.norm = np.array([self._n_obs_per_night, self._nfields])[:, np.newaxis]

        # 
        self.reward_func = reward_func

        self._obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, :-1]
        self._field_ids = schedule.field_id.values[:-1][np.newaxis, np.newaxis, :]
        self._next_obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, 1:]
        self._next_field_ids = schedule.field_id.values[1:][np.newaxis, np.newaxis, :]

        self.obs = np.concatenate((self._field_ids, self._obs_indices), axis=0)
        self.next_obs = np.concatenate((self._next_field_ids, self._next_obs_indices), axis=0)

        self.obs_dim = self.obs.shape[0]
        self._n_nights = np.int32(self.obs.shape[1])
        self.num_actions = self._nfields 

        self.actions = schedule.field_id.values[1:][np.newaxis, :]
        self.rewards = self._get_rewards()
        self.dones = np.zeros_like(self.obs[0], dtype=np.bool_)
        self.dones[:, -1] = True
        self.action_masks = self._get_action_masks()

    def _get_rewards(self):
        """Compute rewards for each transition
        
        Returns
        -------
            Array of shape (n_nights, n_obs_per_night - 1) containing rewards. If ```reward_func``` is None, returns an array of ones
        """
        if self.reward_func is None:
            return np.ones_like(self.obs[0])
        return self._reward_func(self.obs, self.actions, self.next_obs)
    
    def _get_action_masks(self):
        """Computes action masks based on per-field visit limits.

        Tracks cumulative number of visits per field and disallows actions
        (fields) that exceed the maximum visit count observed in the schedule.

        Returns
        --------
            Boolean array of shape (n_nights, n_obs_per_night, n_fields) where ``True`` indicates the action is allowed.
        """
        nvisits_base = np.zeros(shape=(self._nfields), dtype=np.int32)
        full_nvisits = np.zeros(shape=(self._n_nights, self._n_obs_per_night, self._nfields), dtype=np.int32)

        for i, night_ids in enumerate(self._schedule_field_ids):
            for j, field_id in enumerate(night_ids):
                nvisits_base[np.int32(field_id)] += 1
                full_nvisits[i, j] = nvisits_base.copy()
        action_masks = full_nvisits != self._max_visits
        return action_masks

    def __len__(self):
        """Number of nights available in the dataset.

        Returns:
            int: Number of nights.
        """
        return self._n_nights

    def sample(self, batch_size):
        """Randomly samples a batch of transitions. Sampling is performed by uniformly selecting a random night and a random observation step
            within that night
        """
        #TODO should flatten transition dataset first, then sample uniformly
        night_indices = np.random.choice(self._n_nights, batch_size, replace=True)
        obs_indices = np.random.choice(self._n_obs_per_night - 1, batch_size)
        return (
            np.array(self.obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T, # needs to be float for network
            np.array(self.actions[night_indices, obs_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, obs_indices], dtype=np.float32),
            np.array(self.next_obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T,
            np.array(self.dones[night_indices, obs_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, obs_indices], dtype=bool),
        )