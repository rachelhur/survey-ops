import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from collections import defaultdict

from tqdm import tqdm

from survey_ops.utils import ephemerides
from survey_ops.coreRL.data_processing import calculate_and_add_pointing_features, calculate_and_add_bin_features, drop_rows_in_DECam_data, remove_dates, normalize_noncyclic_features, get_inst_teff_rate
import pandas as pd
import json
from torch.utils.data import random_split, RandomSampler

import astropy


# Get the logger associated with this module's name (e.g., 'my_module')
import logging
logger = logging.getLogger(__name__)


def get_lst(timestamp):
    t = astropy.time.Time(timestamp, format='unix', scale='utc')
    lst = t.sidereal_time('apparent', longitude=-1.2358069931456779) # blanco dec
    return lst.to(astropy.units.rad).value

def reward_func_v0():
    raise NotImplementedError

            
def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    # periodic vars first
    feature_names = [
        element 
        for feat_name in feature_names
        for element in ([feat_name + '_cos', feat_name + '_sin'] 
                        if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names)
                        else [feat_name])
]
    return feature_names

def setup_feature_names(include_default_features, include_bin_features, additional_pointing_features, additional_bin_features,
                        default_pntg_feature_names, default_bin_feature_names, cyclical_feature_names, hpGrid, do_cyclical_norm):

    # Any experiment will likely have at least these state features
    required_point_features = default_pntg_feature_names \
                                if include_default_features else []
    required_bin_features = default_bin_feature_names \
                                if (include_default_features and include_bin_features) else []
    
    # Include additional features not in default features above
    pointing_feature_names = required_point_features + additional_pointing_features
    if include_bin_features:
        bin_feature_names = required_bin_features + np.unique(np.array(additional_bin_features)).tolist()
        bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}' for bin_feat in bin_feature_names] for bin_num in range(len(hpGrid.idx_lookup))])
        bin_feature_names = bin_feature_names.flatten().tolist()
    else:
        bin_feature_names = []

    base_pointing_feature_names = pointing_feature_names.copy()
    base_bin_feature_names = bin_feature_names.copy()
    base_feature_names = base_pointing_feature_names + base_bin_feature_names

    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        pointing_feature_names = expand_feature_names_for_cyclic_norm(pointing_feature_names, cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(bin_feature_names, cyclical_feature_names)
    
    state_feature_names = pointing_feature_names + bin_feature_names
    return base_pointing_feature_names, base_bin_feature_names, base_feature_names, pointing_feature_names, bin_feature_names, state_feature_names

class OfflineDECamDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, cfg = None, glob_cfg=None,
                 specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
        assert cfg is not None, "Must pass Config object"

        # Assign static attributes
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec"]
        self.reward_choice = cfg['data']['reward_choice']
        self._calculate_action_mask = cfg['model']['algorithm'] != 'BC' # should be False if using bc (to minimize data processing time), otherwise True

        # Get global feature names
        self.cyclical_feature_names = glob_cfg['features']['CYCLICAL_FEATURE_NAMES'] if self.do_cyclical_norm else []
        self.max_norm_feature_names = glob_cfg['features']['MAX_NORM_FEATURE_NAMES'] if self.do_max_norm else []

        # Get other configurations
        bin_space = cfg['data']['bin_space']
        binning_method = cfg['data']['bin_method']
        nside = cfg['data']['nside']
        remove_large_time_diffs = cfg['data']['remove_large_time_diffs']
        include_bin_features = cfg['data']['include_bin_features']
        num_bins_1d = cfg['data']['num_bins_1d']

        # Initialize healpix grid if binning_method is healpix
        self.hpGrid = None if binning_method != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=(bin_space == 'azel'))

        # Set number of actions based on binning method
        if binning_method == 'uniform':
            self.num_actions = int(num_bins_1d**2)
        elif binning_method == 'healpix':
            self.num_actions = len(self.hpGrid.lon)

        # Save list of all feature names
        self.base_pointing_feature_names, self.base_bin_feature_names, self.base_feature_names, self.pointing_feature_names, self.bin_feature_names, self.state_feature_names \
            = setup_feature_names(include_default_features=True, include_bin_features=include_bin_features,
                                  additional_bin_features=cfg['data']['additional_bin_features'], additional_pointing_features=cfg['data']['additional_pointing_features'],
                                  default_pntg_feature_names=glob_cfg['features']['DEFAULT_PNTG_FEATURE_NAMES'], cyclical_feature_names=self.cyclical_feature_names,
                                  default_bin_feature_names=glob_cfg['features']['DEFAULT_BIN_FEATURE_NAMES'],
                                  hpGrid=self.hpGrid, do_cyclical_norm=self.do_cyclical_norm)
        
        # Get lookup tables
        with open(glob_cfg['paths']['LOOKUP_DIR'] + glob_cfg['files']['FIELD2NAME'], 'r') as f:
            self.field2name = json.load(f)

        # Process dataframe to add columns for pointing features
        df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
        df = drop_rows_in_DECam_data(df, objects_to_remove=self.objects_to_remove)
        df = remove_dates(df,
                          specific_years=cfg['data']['specific_years'] if specific_years is None else specific_years, 
                          specific_months=cfg['data']['specific_months'] if specific_months is None else specific_months, 
                          specific_days=cfg['data']['specific_days'] if specific_days is None else specific_days,
                          specific_filters=cfg['data']['specific_filters'] if specific_filters is None else specific_filters
                          )
        df = calculate_and_add_pointing_features(df=df, field2name=self.field2name, hpGrid=self.hpGrid, pointing_feature_names=self.pointing_feature_names, base_pointing_feature_names=self.base_pointing_feature_names,
                               cyclical_feature_names=self.cyclical_feature_names, do_cyclical_norm=self.do_cyclical_norm)
        bin_df = calculate_and_add_bin_features(pt_df=df, datetimes=df['datetime'], hpGrid=self.hpGrid, base_bin_feature_names=self.base_bin_feature_names, bin_feature_names=self.bin_feature_names, cyclical_feature_names=self.cyclical_feature_names, do_cyclical_norm=self.do_cyclical_norm)
        self._df = df # Save for diagnostics
        self._bin_df = bin_df # Save for diagnostics
            
        # Save night dates, total number of nights in dataset, and number of obs per night
        groups_by_night = df.groupby('night')
        self.unique_nights = groups_by_night.groups.keys()
        self.n_nights = groups_by_night.ngroups
        self.n_obs_per_night = groups_by_night.size() # nights have different numbers of observations

        # Construct Transitions
        states, next_states, actions, rewards, dones, action_masks, num_transitions = self._construct_transitions(
            df=df, 
            bin_df=bin_df,  
            include_bin_features=include_bin_features, 
            num_bins_1d=num_bins_1d, 
            binning_method=binning_method, 
            bin_space=bin_space,
            # timestamps=df.groupby('night').tail(-1)['timestamp'], # all but zenith timestamps,
            remove_large_time_diffs=remove_large_time_diffs
            )

        self.num_transitions = num_transitions

        # # Save Transitions as tensors
        self.states = torch.tensor(states, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.int32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.dones = torch.tensor(dones, dtype=torch.bool)
        self.action_masks = torch.tensor(action_masks, dtype=torch.bool)
        
        # Set dimension of observation
        self.obs_dim = self.states.shape[-1]

        # Normalize states and next_states
        self.states = normalize_noncyclic_features(
            self.states,
            self.state_feature_names,
            self.max_norm_feature_names,
            self.do_inverse_norm,
            self.do_max_norm,
            fix_nans=True
        )
        self.next_states = normalize_noncyclic_features(
            self.next_states,
            self.state_feature_names,
            self.max_norm_feature_names,
            self.do_inverse_norm,
            self.do_max_norm,
            fix_nans=True
        )
        assert self.states.shape[0] == self.actions.shape[0] == self.rewards.shape[0] == self.next_states.shape[0] == self.dones.shape[0] == self.action_masks.shape[0], f"Shape mismatch: states {self.states.shape}, actions {self.actions.shape}, rewards {self.rewards.shape}, next_states {self.next_states.shape}, dones {self.dones.shape}, action_masks {self.action_masks.shape}"
        # self._do_noncyclic_normalizations()

    def _construct_transitions(self, df, bin_df, include_bin_features, num_bins_1d, binning_method, bin_space, remove_large_time_diffs):
        if remove_large_time_diffs:
            next_state_idxs = self._get_next_state_indices(df)
        else:
            next_state_idxs = None
        states, next_states = self._construct_states(df=df, bin_df=bin_df, include_bin_features=include_bin_features, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        num_transitions = states.shape[0]
        actions = self._construct_actions(df, next_states=next_states, bin_space=bin_space, binning_method=binning_method, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs, num_bins_1d=num_bins_1d)
        rewards = self._construct_rewards(df, next_state_idxs=next_state_idxs, remove_large_time_diffs=remove_large_time_diffs, reward_choice=self.reward_choice)
        dones = np.zeros(num_transitions, dtype=bool) # False unless last observation of the night
        dones[-1] = True
        action_masks = self._construct_action_masks(df=df, num_transitions=num_transitions, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        return states, next_states, actions, rewards, dones, action_masks, num_transitions
        
    def _get_next_state_indices(self, df, max_time_diff_min=10):
        time_diffs = df['timestamp'].diff().values
        keep = time_diffs < max_time_diff_min * 60 + 90
        next_state_idxs = np.where(keep)[0]
        self.next_state_idxs = next_state_idxs  # Save for diagnostics
        logger.debug(f'Removing {np.sum(~keep)} transitions with large time diffs > {max_time_diff_min} min. Total transitions: {len(keep)}')
        return next_state_idxs

    def _construct_pointing_features(self, df, remove_large_time_diffs, next_state_idxs=None):
        """
        Constructs state and next_states for all transitions.
        Inserts a "null" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # Pointing features already in DECam data
        missing_cols = set(self.pointing_feature_names) - set(df.columns) == 0
        assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. Must be implemented in method self._process_dataframe()'
        if remove_large_time_diffs:
            next_state_df = df.iloc[next_state_idxs]
            current_state_df = df.iloc[next_state_idxs - 1]
            pointing_features = current_state_df[self.pointing_feature_names].to_numpy()
            next_pointing_features = next_state_df[self.pointing_feature_names].to_numpy()
        else:
            pointing_features = df.groupby('night')[self.pointing_feature_names].apply(lambda group: group.iloc[:-1, :]).to_numpy()
            next_pointing_features = df.groupby('night')[self.pointing_feature_names].apply(lambda group: group.iloc[1:, :]).to_numpy()
        return pointing_features, next_pointing_features
        
    def _construct_bin_features(self, bin_df, remove_large_time_diffs, next_state_idxs):
        # Get bin_features and next_bin_features
        if remove_large_time_diffs:
            next_state_df = bin_df.iloc[next_state_idxs]
            current_state_df = bin_df.iloc[next_state_idxs - 1]
            bin_features = current_state_df[self.bin_feature_names].to_numpy()
            next_bin_features = next_state_df[self.bin_feature_names].to_numpy()
        else:
            bin_features = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[:-1, :]).to_numpy()
            next_bin_features = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[1:, :]).to_numpy()

        self._bin_df = bin_df
        return bin_features, next_bin_features
    
    def _construct_states(self, df, bin_df, include_bin_features, remove_large_time_diffs, next_state_idxs):
        if remove_large_time_diffs:
            pointing_features, next_pointing_features = self._construct_pointing_features(df=df, remove_large_time_diffs=True, next_state_idxs=next_state_idxs)
            if include_bin_features:
                bin_features, next_bin_features = self._construct_bin_features(bin_df=bin_df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
                self.bin_features = bin_features
                self.next_bin_features = next_bin_features
                states = np.concatenate((pointing_features, bin_features), axis=1)
                next_states = np.concatenate((next_pointing_features, next_bin_features), axis=1)
                return states, next_states
            return pointing_features, next_pointing_features
        else:
            pointing_features, next_pointing_features = self._construct_pointing_features(df=df, remove_large_time_diffs=remove_large_time_diffs)
            if include_bin_features:
                bin_features, next_bin_features = self._construct_bin_features(bin_df=bin_df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=None)
                self.bin_features = bin_features
                self.next_bin_features = next_bin_features
                states = np.concatenate((pointing_features, bin_features), axis=1)
                next_states = np.concatenate((next_pointing_features, next_bin_features), axis=1)
                return states, next_states
            return pointing_features, next_pointing_features
    
    def _construct_actions(self, df, next_states, bin_space, binning_method, remove_large_time_diffs, next_state_idxs, num_bins_1d=None):
        assert bin_space in ['radec', 'azel'], 'bin_space must be radec or azel'
        assert binning_method in ['uniform', 'healpix'], 'bining_method must be uniform or healpix'

        if binning_method == 'healpix':
            if remove_large_time_diffs:
                if self.hpGrid.is_azel:
                    lonlat = df.iloc[next_state_idxs][['az', 'el']].values
                else:
                    lonlat = df.iloc[next_state_idxs][['ra', 'dec']].values
                indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])
            else:
                if self.hpGrid.is_azel:
                    lonlat_no_zen = df.groupby('night').tail(-1)[['az', 'el']].values
                else:
                    # lon, lat = df.ra.values, df.dec.values
                    lonlat_no_zen = df.groupby('night').tail(-1)[['ra', 'dec']].values
                indices = self.hpGrid.ang2idx(lon=lonlat_no_zen[:, 0], lat=lonlat_no_zen[:, 1])
            return indices
        
        elif binning_method == 'uniform' and bin_space == 'azel':
            raise NotImplementedError
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

    def _construct_rewards(self, df, remove_large_time_diffs, next_state_idxs, reward_choice='teff_rate'):
        assert reward_choice in ['teff_rate', 'expert_actions'], 'reward_choice must be teff_rate or expert_actions'
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        if reward_choice == 'teff_rate':
            rewards = get_inst_teff_rate(df=df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        elif reward_choice == 'expert_actions':
            if remove_large_time_diffs:
                next_state_df = df.iloc[next_state_idxs]
                rewards = np.ones(len(next_state_df), dtype=np.float32)
        return rewards

    def _construct_action_masks(self, df, num_transitions, remove_large_time_diffs, next_state_idxs):
        """
        Constructs action masks only with the condition that bins outside of horizon are masked
        """
        if remove_large_time_diffs:
            df = df.iloc[next_state_idxs-1]
        # given timestamp, determine bins which are outside of observable range
        els = np.empty((num_transitions, self.num_actions))
        if self._calculate_action_mask:
            print("Calculating action masks based on horizon. This may take a few minutes...")
            if not self.hpGrid.is_azel:
                lon, lat = self.hpGrid.lon, self.hpGrid.lat
                for i, time in tqdm(enumerate(df['timestamp'].values), total=len(df['timestamp'].values), desc="Calculating action mask"):
                    _, els[i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
                self._els = els
                action_mask = els > 0
            else:
                els = np.tile(self.hpGrid.lon[:, np.newaxis], reps=len(df['timestamp'].values)).T
                action_mask = els > 0
        else:
            action_mask = np.ones((num_transitions, self.num_actions))
        return action_mask
        
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
    
    def get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True, val_split=.1, return_train_and_val=True):
        generator = torch.Generator().manual_seed(random_seed)
    
        # Split dataset
        train_size = int(len(self) * (1 - val_split))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        
        # Train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=10**10),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator
        )
        if return_train_and_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader
        return train_loader

        # loader = DataLoader(
        #     self,
        #     batch_size,
        #     sampler=RandomSampler(
        #         self,
        #         replacement=True,
        #         num_samples=10**10,
        #     ),
        #     drop_last=drop_last, # drops last non-full batch
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        #     generator=generator
        # )
 
class ToyDatasetv0:
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