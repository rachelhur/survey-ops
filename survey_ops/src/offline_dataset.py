import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from collections import defaultdict

from survey_ops.utils.units import *
from survey_ops.utils.ephemerides import *
import healpy as hp

import pandas as pd
import json

def reward_func_v0():
    raise NotImplementedError

def produce_schedule_for_video(outdir, id2radec):
    field_filepath = outdir + 'fields2radec.json' # field id to ra_dec
    schedule_filepath = outdir + 'schedule.csv' # keys time and field_id

    # save field_to_radec
    with open(field_filepath, 'w') as f:
        json.dump(id2radec, f, indent=2)
    
    # save time, field_id, next_field_id
    df = pd.DataFrame(list(data.items()), columns=['timestamp', 'field_id', 'next_field_id'])
    df.to_csv(schedule_filepath, index=False)

class OfflineDECamDataset(torch.utils.data.Dataset):
    def __init__(self, 
                df: pd.DataFrame, 
                num_bins_1d = 10,
                normalize_state: bool = True, 
                specific_years: list = None,
                specific_months: list = None,
                specific_days: list = None,
                specific_filters: list = None,
                binning_method = 'healpix',
                bin_space = 'radec',
                nside=None
                ):
        self.stateidx2name = {0: 'ra', 1: 'dec', 2: 'az', 3: 'el', 
                                4: 'sun_ra', 5: 'sun_dec', 6: 'sun_az', 7: 'sun_el',
                                8: 'moon_ra', 9: 'moon_dec', 10: 'moon_az', 11: 'moon_el',
                                12: 'airmass', 13: 'ha', 14: 'T_night_fraction'
                                }
        self.statename2idx = {v: k for k, v in self.stateidx2name.items()}
        self.normalize_state = normalize_state
        assert binning_method in ['uniform_grid', 'healpix']
        assert (binning_method == 'uniform_grid' and num_bins_1d is not None) or (binning_method == 'healpix' and nside is not None)
        if binning_method == 'healpix':
            self.hpGrid = HealpixGrid(nside=nside, hemisphere=False)
        else:
            self.hpGrid = None

        # self.fov = 2.2 # degrees
        # self.arcsec_per_pix = .26 # https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        # self.dither_offset = 100 #arcsec

        # Add timestamps column to df and sort df by timestamp (increasing)
        utc = pd.to_datetime(df['datetime'], utc=True)
        timestamps = (utc.astype('int64') // 10**9).values
        df['timestamp'] = timestamps
        df = df.sort_values(by='timestamp')

        # Ensure all data are 32-bit precision before training
        for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
            cols = df.select_dtypes(include=[str_bit]).columns
            df[cols] = df[cols].astype(np_bit)
        
        # Set the DataFrame index as its datetime
        # df = df.set_index('datetime')
        # df.index = ((df.index) - pd.Timedelta(hours=np.max(np.unique(df.index.hour))))
        # df = df[df.index.year != 1970]
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
        # remove observations in 1970 - what are these?
        df = df[df['night'].dt.year != 1970]
        self._df = df # save for diagnostics - #TODO remove when all is tested
        
        # Group observations by observation night
        groups = df.groupby('night')
        self._groups = groups
        self.unique_nights = groups.groups.keys()
        self.n_nights = groups.ngroups
        self.n_obs_per_night = groups.size() # nights have different numbers of observations
        
        # Set dataset-wide (across observation nights) attributes
        self.n_obs_tot = len(df)
        self.num_transitions = len(df) # size of dataset
        if binning_method == 'uniform_grid':
            self.num_actions = int(num_bins_1d**2)
        elif binning_method == 'healpix':
            self.num_actions = hp.nside2npix(nside)
        
        # Get transition variables
        states, next_states = self._construct_states(groups)
        actions = self._construct_actions(next_states, num_bins_1d=num_bins_1d, binning_method=binning_method, bin_space=bin_space)
        rewards = self._construct_rewards(groups)
        dones = np.zeros(self.num_transitions, dtype=bool) # False unless last observation of the night
        self._done_indices = np.where(states[:, 0] == 0)[0][1:] - 1
        dones[self._done_indices] = True
        dones[-1] = True
        action_masks = self._construct_action_masks()

        # Save transitions as tensors and instatiate as attributes
        self.states = torch.tensor(states, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.int32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.bool)
        self.action_masks = torch.tensor(action_masks, dtype=torch.bool)
        
        # Set dimension of observation
        self.obs_dim = self.states.shape[-1]

        # Construct lookup table for scheduling
        self.idx2radec = {self.hpGrid.ang2idx(lon=lon * deg, lat=lat * deg): np.array([lon, lat]) for (lon, lat) in zip(df['ra'].values, df['dec'].values)}
        # Normalize states if specified
        if self.normalize_state:
            self.means = torch.mean(self.next_states, axis=0)
            self.stds = torch.std(self.next_states, axis=0)
            self.next_states = (self.next_states - self.means) / self.stds

            mask_null = self.states == 0
            new_states = self.states.clone()
            new_states = (new_states - self.means) / self.stds
            new_states[mask_null] = 0.
            self.states = new_states
        else:
            self.means = 0.
            self.stds = 1.

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

    def _construct_states(self, groups):
        """
        Constructs state and next_states for all transitions.
        Inserts a "null" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # State vars
        all_states = np.zeros(shape=(self.n_obs_tot + self.n_nights, len(self.stateidx2name)))
        null_obs_indices = []
        
        for i, ((day, subdf), (_, idxs)) in enumerate(zip(groups, groups.indices.items())):
            indices = idxs + i + 1
            null_obs_indices.append(idxs[0] + i)
            timestamps = subdf['timestamp'].values
            all_states[indices, self.statename2idx['ra']] = subdf['ra'].values
            all_states[indices, self.statename2idx['dec']] = subdf['dec'].values
            all_states[indices, self.statename2idx['az']] = subdf['az'].values
            all_states[indices, self.statename2idx['el']] = 90.0 - subdf['zd'].values
            all_states[indices, self.statename2idx['airmass']] = subdf['airmass'].values
            all_states[indices, self.statename2idx['ha']] = subdf['ha'].values

            T_night = subdf['timestamp'].values[-1] - subdf['timestamp'].values[0]
            # if T_night == 0:
            #     print(f'BAD (T_obs) at {i}, indices {indices}')
            #     print(len(subdf))
            
            # Get sun and moon az,el
            for idx, time in zip(indices, timestamps):
                sun_ra, sun_dec = get_source_ra_dec('sun', time=time)
                moon_ra, moon_dec = get_source_ra_dec('moon', time=time)
                all_states[idx, self.statename2idx['sun_ra']] = sun_ra
                all_states[idx, self.statename2idx['sun_dec']] = sun_dec
                all_states[idx, self.statename2idx['moon_ra']] = moon_ra
                all_states[idx, self.statename2idx['moon_dec']] = moon_dec
                all_states[idx, self.statename2idx['sun_az']], all_states[idx, self.statename2idx['sun_el']] = equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)
                all_states[idx, self.statename2idx['moon_az']], all_states[idx, self.statename2idx['moon_el']] = equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)
                if T_night == 0 and time - timestamps[0] == 0:
                    all_states[idx, self.statename2idx['T_night_fraction']] = 0.
                else:   
                    all_states[idx, self.statename2idx['T_night_fraction']] = (time - timestamps[0]) / T_night
         
        null_obs_indices = np.array(null_obs_indices, dtype=np.int32)
        self.null_mask = np.ones(shape=all_states.shape[0], dtype=bool)
        self.null_mask[null_obs_indices] = False
        self._all_states = all_states
        states = np.delete(all_states, null_obs_indices[1:] - 1, axis=0)[:-1]
        next_states = np.delete(all_states, null_obs_indices, axis=0)

        return states, next_states

    def _construct_actions(self, next_states, bin_space='radec', binning_method='healpix', num_bins_1d=None):
        assert bin_space in ['radec', 'azel']
        assert binning_method in ['uniform_grid', 'healpix']
        if binning_method == 'uniform_grid' and bin_space == 'azel':
            az_idx = self.statename2idx['az']
            el_idx = self.statename2idx['el']
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
            for ra, dec, bin_id in zip(self._df['ra'].values, self._df['dec'].values, bin_ids):
                id2radec[bin_id].append((ra, dec))
            self.id2radec = dict(sorted(id2radec.items()))

            return bin_ids
        
        elif binning_method=='healpix' and bin_space=='radec':
            ra_idx = self.statename2idx['ra']
            dec_idx = self.statename2idx['dec']
            indices = self.hpGrid.ang2idx(lon=next_states[:, ra_idx]*deg, lat=next_states[:, dec_idx]*deg)
            return indices
        else:
            raise NotImplementedError
    
    def _get_unique_fields_dict(self, radec_width=.05):
        """Constructs uniform binning across RA/Dec, then decides that all observations within this width are observing the same field.

        Args
        ----
        radec_width (float): the bin width in degrees
        """
        radec_width = .05
        ra_edges = np.arange(np.min(self._df['ra'].values), np.max(self._df['ra'].values), step=radec_width, dtype=np.float32)
        dec_edges = np.arange(np.min(self._df['dec'].values), np.max(self._df['dec'].values), step=radec_width, dtype=np.float32)

        i_x = np.digitize(self._df['ra'].values, ra_edges).astype(np.int32) - 1
        i_y = np.digitize(self._df['dec'].values, dec_edges).astype(np.int32) - 1
        bin_ids = i_x + i_y * (num_bins_1d)
        raise NotImplementedError

    def _construct_rewards(self, groups):
        rewards = np.ones(self.num_transitions)
        for ((day, subdf), (_, idxs)) in zip(groups, groups.indices.items()):
            rewards[idxs] = subdf['teff']
        return rewards

    def _construct_action_masks(self):
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
    
    def save_schedule_for_video(self, outdir):
        # def produce_schedule_for_video(outdir, id2radec):
        field_filepath = outdir + 'fields2radec.json' # field id to ra_dec
        schedule_filepath = outdir + 'true_schedule.csv' # keys time and field_id

        idx2radec = {key: np.float64(items*deg).tolist() for key, items in self.idx2radec.items()}
        # save field_to_radec
        with open(field_filepath, 'w') as f:
            json.dump(idx2radec, f, indent=2)
        
        # save time, field_id, next_field_id
        timestamps = self._df['timestamp'].values.tolist()
        all_field_ids = self.hpGrid.ang2idx(lon=self._df['ra'].values*deg, lat=self._df['dec'].values*deg).tolist()
        field_ids = all_field_ids
        next_field_ids = all_field_ids[1:] + [None]
        data = {'time': timestamps, 'field_id': field_ids, 'next_field_id': next_field_ids}
        df = pd.DataFrame(data)
        df.to_csv(schedule_filepath, index=False)



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