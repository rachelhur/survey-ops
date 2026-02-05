import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from collections import defaultdict

from tqdm import tqdm

from survey_ops.utils import units
from survey_ops.utils import ephemerides
# from survey_ops.coreRL.survey_logic import get_fields_in_azel_bin, get_fields_in_radec_bin

import pandas as pd
import json
from torch.utils.data import random_split, RandomSampler

import astropy
import logging
from survey_ops.coreRL.survey_logic import do_noncyclic_normalizations, add_bin_visits_to_dataframe


# Get the logger associated with this module's name (e.g., 'my_module')
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
        bin_feature_names = required_bin_features + list(set(additional_bin_features))
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

        # Save list of all feature names
        self.base_pointing_feature_names, self.base_bin_feature_names, self.base_feature_names, self.pointing_feature_names, self.bin_feature_names, self.state_feature_names \
            = setup_feature_names(include_default_features=True, include_bin_features=include_bin_features,
                                  additional_bin_features=cfg['data']['additional_bin_features'], additional_pointing_features=cfg['data']['additional_pointing_features'],
                                  default_pntg_feature_names=glob_cfg['features']['DEFAULT_PNTG_FEATURE_NAMES'], cyclical_feature_names=self.cyclical_feature_names,
                                  default_bin_feature_names=glob_cfg['features']['DEFAULT_BIN_FEATURE_NAMES'],
                                  hpGrid=self.hpGrid, do_cyclical_norm=self.do_cyclical_norm)
        
        # Get lookup tables
        with open(glob_cfg['paths']['LOOKUP_DIR'] + '/' +  glob_cfg['files']['FIELD2NAME'], 'r') as f:
            self.field2name = json.load(f)

        # Process dataframe to add columns for pointing features
        df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
        df = self._drop_rows(df)
        df = self._process_dataframe(
            df, 
            specific_years= cfg['data']['specific_years'] if specific_years is None else specific_years, 
            specific_months= cfg['data']['specific_months'] if specific_months is None else specific_months, 
            specific_days= cfg['data']['specific_days'] if specific_days is None else specific_days, 
            specific_filters= cfg['data']['specific_filters'] if specific_filters is None else specific_filters
        )
        self._df = df # Save for diagnostics

        # Set dataset-wide (across observation nights) attributes
        if binning_method == 'uniform':
            self.num_actions = int(num_bins_1d**2)
        elif binning_method == 'healpix':
            self.num_actions = len(self.hpGrid.lon)
            
        # Save night dates, total number of nights in dataset, and number of obs per night
        groups_by_night = df.groupby('night')
        self.unique_nights = groups_by_night.groups.keys()
        self.n_nights = groups_by_night.ngroups
        self.n_obs_per_night = groups_by_night.size() # nights have different numbers of observations

        # Construct Transitions
        states, next_states, actions, rewards, dones, action_masks, num_transitions = self._construct_transitions(
            df=df, 
            include_bin_features=include_bin_features, 
            num_bins_1d=num_bins_1d, 
            binning_method=binning_method, 
            bin_space=bin_space,
            timestamps=df.groupby('night').tail(-1)['timestamp'], # all but zenith timestamps,
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
        self.states = do_noncyclic_normalizations(
            self.states,
            self.state_feature_names,
            self.max_norm_feature_names,
            self.do_inverse_norm,
            self.do_max_norm,
            fix_nans=True
        )
        self.next_states = do_noncyclic_normalizations(
            self.next_states,
            self.state_feature_names,
            self.max_norm_feature_names,
            self.do_inverse_norm,
            self.do_max_norm,
            fix_nans=True
        )      
        # self._do_noncyclic_normalizations()

        
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
        for _, g in object_radec_groups:
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

    def _remove_specific_objects(self, objects_to_remove, df):
        nights_with_special_fields = set()
        for i, spec_obj in enumerate(objects_to_remove):
            for night, subdf in df.groupby('night'):
                if any(spec_obj in obj_name for obj_name in subdf['object'].values):
                    nights_with_special_fields.add(night)
        
        nights_to_remove_mask = df['night'].isin(nights_with_special_fields)
        df = df[~nights_to_remove_mask]
        return df

    def _drop_rows(self, df):
        # Remove observations in 1970 - what are these?
        df = df[df['datetime'].dt.year != 1970]
        assert len(df) > 0, "No observations found for the specified year/month/day/filter selections."

        # Remove specific nights according to object name
        df = self._remove_specific_objects(objects_to_remove=self.objects_to_remove, df=df)
        
        # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
        df = self._relabel_mislabelled_objects(df)
        return df

    def _process_dataframe(self, df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
        """Processes and filters the dataframe to return a new dataframe with added columns for current pointing state features"""
        # Add column which indicates observing night (noon to noon)
        # Get observations for specific years, days, filters, etc.
        if specific_years is not None and specific_years is not []:
            df = df[df['night'].dt.year.isin(specific_years)]
            assert not df.empty, f"Years {specific_years} do not exist in dataset"
        if specific_months is not None and specific_months is not []:
            df = df[df['night'].dt.month.isin(specific_months)]
            assert not df.empty, f"Months {specific_months} do not exist in years {specific_years}"
        if specific_days is not None and specific_days is not []:
            df = df[df['night'].dt.day.isin(specific_days)]
            assert not df.empty, f"Days {specific_days} do not exist in months {specific_months}, and years {specific_years}"
        if specific_filters is not None and specific_filters is not []:
            df = df[df['filter'].isin(specific_filters)]
            assert not df.empty, f"Filters {specific_filters} do not exist in days {specific_days}, months {specific_months}, and years {specific_years}"

        # Sort df by timestamp
        df = df.sort_values(by='timestamp')

        # Get time dependent features (sun and moon pos)
        for idx, time in tqdm(zip(df.index, df['timestamp'].values), total=len(df['timestamp']), desc='Calculating sun and moon ra/dec and az/el'):
            sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
            df.loc[idx, ['sun_ra', 'sun_dec']] = sun_ra, sun_dec
            df.loc[idx, ['sun_az', 'sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)

            moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
            df.loc[idx, ['moon_ra', 'moon_dec']] = moon_ra, moon_dec
            df.loc[idx, ['moon_az', 'moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

        df['time_fraction_since_start'] = df.groupby('night')['timestamp'].transform(lambda x: (x - x.values[0]) / (x.values[-1] - x.values[0]) if len(x) > 1 else 0)
        df.loc[:, ['ra', 'dec', 'az', 'zd', 'ha']] *= units.deg
        df['el'] = np.pi/2 - df['zd'].values

        # Add bin and field id columns to dataframe
        df['field_id'] = df['object'].map({v: k for k, v in self.field2name.items()})
        if self.hpGrid is not None:
            if self.hpGrid.is_azel:
                lon = df['az']
                lat = df['el']
            else:
                lon = df['ra']
                lat = df['dec']
            df['bin'] = self.hpGrid.ang2idx(lon=lon, lat=lat)

        # Add other feature columns for those not present in dataframe
        for feat_name in self.base_pointing_feature_names:
            if feat_name in df.columns:
                continue
            else:
                if 'bins_visited_in_night' in self.pointing_feature_names:
                    df = add_bin_visits_to_dataframe(df)
        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_pointing_feature_names:
                if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in self.cyclical_feature_names):
                    df[f'{feat_name}_cos'] = np.cos(df[feat_name].values)
                    df[f'{feat_name}_sin'] = np.sin(df[feat_name].values)

        # Insert zenith states in dataframe (needed for gym.environment to use zenith state as first state)

        zenith_df = self._get_zenith_states(original_df=df, is_pointing=True)
        df = pd.concat([df, zenith_df], ignore_index=True)
        df = df.sort_values(by='timestamp')

        # Ensure all data are 32-bit precision before training
        for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
            cols = df.select_dtypes(include=[str_bit]).columns
            df[cols] = df[cols].astype(np_bit)

        return df
    
    def _get_next_state_indices(self, df, max_time_diff_min=10):
        time_diffs = df['timestamp'].diff().values
        keep = time_diffs < max_time_diff_min * 60 + 90
        next_state_idxs = np.where(keep)[0]
        return next_state_idxs

    def _construct_transitions(self, df, include_bin_features, num_bins_1d, binning_method, bin_space, timestamps, remove_large_time_diffs=False):
        if remove_large_time_diffs:
            next_state_idxs = self._get_next_state_indices(df)
        else:
            next_state_idxs = None
        states, next_states = self._construct_states(df=df, include_bin_features=include_bin_features, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        actions = self._construct_actions(df, num_bins_1d=num_bins_1d, binning_method=binning_method, bin_space=bin_space, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        rewards = self._construct_rewards(df)
        num_transitions = states.shape[0]
        dones = np.zeros(num_transitions, dtype=bool) # False unless last observation of the night
        dones[-1] = True
        action_masks = self._construct_action_masks(timestamps=timestamps, num_transitions=num_transitions, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        return states, next_states, actions, rewards, dones, action_masks, num_transitions
    
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
        
    def _construct_bin_features(self, df, datetimes, remove_large_time_diffs, next_state_idxs=None):
        # These timestamps already have zenith -- they were already constructed when calculating zeniths for pointing features
        # Create empty arrays 
        timestamps=df['timestamp'].values
        hour_angles = np.empty(shape=(len(timestamps), len(self.hpGrid.idx_lookup)))
        airmasses = np.empty_like(hour_angles)
        moon_dists = np.empty_like(hour_angles)
        xs = np.empty_like(hour_angles) # xs = az if actions are in ra, dec
        ys = np.empty_like(hour_angles) # ys = dec if actions are in az, el
        num_visits_hist = np.zeros_like(hour_angles, dtype=np.int32)
        num_visits_tracking = np.zeros_like(hour_angles[0])

        lon, lat = self.hpGrid.lon, self.hpGrid.lat
        for i, time in tqdm(enumerate(timestamps), total=len(timestamps), desc='Calculating bin features for all healpix bins and timestamps'):
            hour_angles[i] = self.hpGrid.get_hour_angle(time=time)
            airmasses[i] = self.hpGrid.get_airmass(time)
            moon_dists[i] = self.hpGrid.get_source_angular_separations('moon', time=time)
            if self.hpGrid.is_azel:
                xs[i], ys[i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
            else:
                xs[i], ys[i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
            if df.iloc[i]['object'] != 'zenith':
                bin_num = df.iloc[i]['bin']
                num_visits_tracking[bin_num] += 1
            num_visits_hist[i] = num_visits_tracking.copy()

        stacked = np.stack([hour_angles, airmasses, moon_dists, xs, ys, num_visits_hist], axis=2) # Order must be exactly same as base_bin_feature_names
        bin_states = stacked.reshape(len(hour_angles), -1)
        bin_df = pd.DataFrame(data=bin_states, columns=self.base_bin_feature_names)
        bin_df['night'] = (datetimes - pd.Timedelta(hours=12)).dt.normalize()
        bin_df['timestamp'] = timestamps

        # Normalize periodic features here and add as df cols
        new_cols = {}
        if self.do_cyclical_norm:
            for feat_name in tqdm(self.base_bin_feature_names, total=len(self.base_bin_feature_names), desc='Normalizing bin features'):
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    new_cols[f'{feat_name}_cos'] = np.cos(bin_df[feat_name].values)
                    new_cols[f'{feat_name}_sin'] = np.sin(bin_df[feat_name].values)
        
        new_cols_df = pd.DataFrame(data=new_cols)
        bin_df = pd.concat([bin_df, new_cols_df], axis=1)

        bin_df = bin_df.reset_index(drop=True, inplace=False)
        # zenith_df = self._get_zenith_states(original_df=bin_df, is_pointing=False)
        # bin_df = pd.concat([bin_df, zenith_df], ignore_index=True)
        bin_df = bin_df.sort_values(by='timestamp')
        
        # Pointing features already in DECam data
        missing_cols = set(self.bin_feature_names) - set(bin_df.columns) == 0
        assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'

       # Ensure all data are 32-bit precision before training
        for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
            cols = bin_df.select_dtypes(include=[str_bit]).columns
            bin_df[cols] = bin_df[cols].astype(np_bit)

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
    
    def _construct_states(self, df, include_bin_features, remove_large_time_diffs, next_state_idxs):
        if remove_large_time_diffs:
            pointing_features, next_pointing_features = self._construct_pointing_features(df=df, remove_large_time_diffs=True, next_state_idxs=next_state_idxs)
            if include_bin_features:
                bin_features, next_bin_features = self._construct_bin_features(df=df, datetimes=df['datetime'], remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
                self.bin_features = bin_features
                self.next_bin_features = next_bin_features
                states = np.concatenate((pointing_features, bin_features), axis=1)
                next_states = np.concatenate((next_pointing_features, next_bin_features), axis=1)
                return states, next_states
            return pointing_features, next_pointing_features
        else:
            pointing_features, next_pointing_features = self._construct_pointing_features(df=df, remove_large_time_diffs=remove_large_time_diffs)
            if include_bin_features:
                bin_features, next_bin_features = self._construct_bin_features(df=df, datetimes=df['datetime'], remove_large_time_diffs=remove_large_time_diffs)
                self.bin_features = bin_features
                self.next_bin_features = next_bin_features
                states = np.concatenate((pointing_features, bin_features), axis=1)
                next_states = np.concatenate((next_pointing_features, next_bin_features), axis=1)
                return states, next_states
            return pointing_features, next_pointing_features
    
    def _construct_actions(self, df, next_states=None, bin_space='radec', binning_method='healpix', num_bins_1d=None, remove_large_time_diffs=False, next_state_idxs=None):
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

    def _construct_rewards(self, df, remove_large_time_diffs=False, next_state_idxs=None):
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        teff_no_zen = df[df['object'] != 'zenith'][['teff']].values[:, 0]
        t_diff = df.sort_values(['night', 'timestamp']).groupby('night')['timestamp'].diff().dropna().to_numpy()
        teff_inst_rate = teff_no_zen / t_diff
        min_rate = np.min(teff_inst_rate)
        max_rate = np.max(teff_inst_rate)
        rewards = (teff_inst_rate - min_rate)/max_rate
        return rewards

    def _construct_action_masks(self, timestamps, num_transitions, remove_large_time_diffs=False, next_state_idxs=None):
        """
        Constructs action masks only with the condition that bins outside of horizon are masked
        """
        # given timestamp, determine bins which are outside of observable range
        els = np.empty((num_transitions, self.num_actions))
        if self._calculate_action_mask:
            if not self.hpGrid.is_azel:
                lon, lat = self.hpGrid.lon, self.hpGrid.lat
                for i, time in tqdm(enumerate(timestamps), total=len(timestamps), desc="Calculating action mask"):
                    _, els[i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
                self._els = els
                action_mask = els > 0
            else:
                action_mask = self.hpGrid.lat > 0
        else:
            action_mask = np.ones((num_transitions, self.num_actions))
        return action_mask

    def _get_zenith_states(self, original_df, is_pointing=True):
        _df = original_df.reset_index(drop=True, inplace=False)
        zenith_timestamps = _df.groupby('night').head(1).timestamp - 10
        zenith_datetimes = (_df.groupby('night').head(1).datetime - pd.Timedelta(seconds=10)).values
        zenith_rows = []
        nights = original_df.night.unique()
        if is_pointing:
            for i_row, time in tqdm(enumerate(zenith_timestamps), total=len(zenith_timestamps), desc='Calculating zenith states'):
                row_dict = {}
                row_dict['timestamp'] = time
                # row_dict['datetime'] = zenith_datetimes[i_row]
                row_dict['night'] = nights[i_row]
                row_dict['sun_ra'], row_dict['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=time)
                row_dict['moon_ra'], row_dict['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=time)
                row_dict['sun_az'], row_dict['sun_el'] = ephemerides.equatorial_to_topographic(ra=row_dict['sun_ra'], dec=row_dict['sun_dec'], time=time)
                row_dict['moon_az'], row_dict['moon_el'] = ephemerides.equatorial_to_topographic(ra=row_dict['moon_ra'], dec=row_dict['moon_dec'], time=time)
                total_time_in_night = original_df.groupby('night').get_group(row_dict['night'])['timestamp'].values[-1] - original_df.groupby('night').get_group(row_dict['night'])['timestamp'].values[0]
                row_dict['time_fraction_since_start'] = (time - original_df.groupby('night').get_group(row_dict['night'])['timestamp'].values[0]) / total_time_in_night if total_time_in_night > 0 else 0
                # Get zenith bin if radec bin space
                blanco = ephemerides.blanco_observer(time=time)
                zenith_ra, zenith_dec = blanco.radec_of('0', '90')
                row_dict['ra'] = zenith_ra
                row_dict['dec'] = zenith_dec
                if not self.hpGrid.is_azel: 
                    row_dict['bin'] = self.hpGrid.ang2idx(lon=zenith_ra, lat=zenith_dec)
                else:
                    row_dict['bin'] = self.hpGrid.ang2idx(lon=0, lat=np.pi/2)
                zenith_rows.append(row_dict)

            zenith_df = pd.DataFrame(zenith_rows)
            zenith_df['dec'] = blanco.lat
            zenith_df['az'] = 0
            zenith_df['el'] = np.pi/2
            zenith_df['airmass'] = 1
            zenith_df['ha'] = 0
            zenith_df['object'] = 'zenith'
            zenith_df['field_id'] = -1
            zenith_df['datetime'] = zenith_datetimes

            for feat_name in self.base_pointing_feature_names:
                if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in self.cyclical_feature_names):
                    zenith_df[f'{feat_name}_cos'] = np.cos(zenith_df[feat_name].values)
                    zenith_df[f'{feat_name}_sin'] = np.sin(zenith_df[feat_name].values)

        else: # ['ha', 'airmass', 'ang_dist_to_moon']
            for i_row, time in tqdm(enumerate(zenith_timestamps), total=len(zenith_timestamps), desc='Calculating grid-wide zenith states'):
                row_dict = {}
                row_dict['time'] = time
                row_dict['night'] = nights[i_row]
                row_dict['ha'] = self.hpGrid.get_hour_angle(time=time)
                row_dict['airmass'] = self.hpGrid.get_airmass(time)
                row_dict['ang_dist_to_moon'] = self.hpGrid.get_source_angular_separations('moon', time=time)
                if self.hpGrid.is_azel:
                    row_dict['xs'], row_dict['ys'] = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=time)
                else:
                    row_dict['xs'], row_dict['ys'] = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=time)
                zenith_rows.append(row_dict)

            zenith_df = pd.DataFrame(zenith_rows)
            
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in self.cyclical_feature_names):
                    zenith_df[f'{feat_name}_cos'] = np.cos(zenith_df[feat_name].values)
                    zenith_df[f'{feat_name}_sin'] = np.sin(zenith_df[feat_name].values)

        return zenith_df

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