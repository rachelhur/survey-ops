import fitsio
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

import torch

from survey_ops.utils import ephemerides, units
from tqdm import tqdm
logger = logging.getLogger(__name__)

# TODO make into class
def load_raw_data_to_dataframe(fits_path, json_path, save_as_json=True):
    try:
        # --- Load json df ---- #
        df = pd.read_json(json_path)
        logger.info('Loaded data from json')
    except:
        # --- Load fits ---- #
        logger.info(f"Could not find json file {json_path}. Processing data from fits file {fits_path}.")
        d = fitsio.read(fits_path)
        sel = (d['propid'] == '2012B-0001') & (d['exptime'] > 40) & (d['exptime'] < 100) & (~np.isnan(d['teff']))
        selected_d = d[sel]
        column_names = selected_d.dtype.names
        df = pd.DataFrame(selected_d, columns=column_names)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

        # Add timestamp col
        utc = pd.to_datetime(df['datetime'], utc=True)
        timestamps = (utc.astype('int64') // 10**9).values
        df['timestamp'] = timestamps.copy()
        if save_as_json:
            df.to_json(json_path)
    return df

def get_inst_teff_rate(df, remove_large_time_diffs, next_state_idxs):
    if remove_large_time_diffs:
        next_state_df = df.iloc[next_state_idxs]
        current_state_df = df.iloc[next_state_idxs-1]
        t_diff = next_state_df['timestamp'].values - current_state_df['timestamp'].values
        teff_no_zen = next_state_df[['teff']].values[:, 0]
    else:
        t_diff = df.sort_values(['night', 'timestamp']).groupby('night')['timestamp'].diff().dropna().to_numpy()
        teff_no_zen = df[df['object'] != 'zenith'][['teff']].values[:, 0]

    teff_inst_rate = teff_no_zen / t_diff
    min_rate = np.min(teff_inst_rate)
    max_rate = np.max(teff_inst_rate)
    rewards = (teff_inst_rate - min_rate)/max_rate
    return rewards

def drop_rows_in_DECam_data(df, objects_to_remove):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    # Remove observations in 1970 - what are these?
    df = df[df['datetime'].dt.year != 1970]
    assert len(df) > 0, "No observations found for the specified year/month/day/filter selections."

    # Remove specific nights according to object name
    df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    
    # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
    df = relabel_mislabelled_objects(df)
    return df

def remove_specific_objects(df, objects_to_remove):
    nights_with_special_fields = set()
    for i, spec_obj in enumerate(objects_to_remove):
        for night, subdf in df.groupby('night'):
            if any(spec_obj in obj_name for obj_name in subdf['object'].values) or any(subdf['object'] == ""):
                nights_with_special_fields.add(night)

    nights_to_remove_mask = df['night'].isin(nights_with_special_fields)
    df = df[~nights_to_remove_mask]
    return df

def relabel_mislabelled_objects(df):
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


def add_per_night_progress_to_dataframe(df):
    for night, group in df.groupby('night'):
        bins_visited_tonight = set()
        visited_running = 0
        remaining_running = 0

def add_bin_visits_to_dataframe(df):
    bins_visited = []
    for _, group in df.groupby('night'):
        bin_history = set()
        objects = group['object'].values
        bins = group['bin'].values
        for obj, bin_num in zip(objects, bins):
            if obj == 'zenith':
                bins_visited.append(0)
            else:
                bin_history.add(bin_num)
                bins_visited.append(len(bin_history))
        # for _, row in group.iterrows():
        #     if row['object'] == 'zenith':
        #         bins_visited.append(0)
        #     else:
        #         bin_history.add(row['bin'])
        #         bins_visited.append(len(bin_history))
    
    df['bins_visited_in_night'] = bins_visited
    return df

def normalize_noncyclic_features(state, state_feature_names,
                                      max_norm_feature_names,
                                      do_inverse_norm, do_max_norm,
                                      fix_nans=True):
    is_torch = torch.is_tensor(state)
    # build masks (numpy boolean array)
    airmass_mask = np.array(
        ['airmass' in feat for feat in state_feature_names],
        dtype=bool
    )
    max_norm_mask = np.array(
        [any(max_feat in feat for max_feat in max_norm_feature_names)
         for feat in state_feature_names],
        dtype=bool
    )
    if is_torch:
        airmass_mask = torch.tensor(airmass_mask, dtype=torch.bool, device=state.device)
        max_norm_mask = torch.tensor(max_norm_mask, dtype=torch.bool, device=state.device)

    if do_inverse_norm:
        state[..., airmass_mask] = 1.0 / state[..., airmass_mask]

    if do_max_norm:
        state[..., max_norm_mask] = state[..., max_norm_mask] / (np.pi / 2)

    if fix_nans:
        if is_torch:
            state[torch.isnan(state)] = 10.0
        else:
            state[np.isnan(state)] = 10.0
    return state

def remove_dates(df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
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
    return df

def calculate_and_add_pointing_features(df, field2name, hpGrid, pointing_feature_names, 
                      base_pointing_feature_names, cyclical_feature_names, do_cyclical_norm):
    """Processes and filters the dataframe to return a new dataframe with added columns for current pointing state features"""
    # Sort df by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

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
    df['field_id'] = df['object'].map({v: k for k, v in field2name.items()})
    if hpGrid is not None:
        if hpGrid.is_azel:
            lon = df['az']
            lat = df['el']
        else:
            lon = df['ra']
            lat = df['dec']
        df['bin'] = hpGrid.ang2idx(lon=lon, lat=lat)

    # Add other feature columns for those not present in dataframe
    for feat_name in base_pointing_feature_names:
        if feat_name in df.columns:
            continue
        else:
            if 'bins_visited_in_night' in pointing_feature_names:
                df = add_bin_visits_to_dataframe(df)
    # Normalize periodic features here and add as df cols
    if do_cyclical_norm:
        for feat_name in base_pointing_feature_names:
            if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in cyclical_feature_names):
                df[f'{feat_name}_cos'] = np.cos(df[feat_name].values)
                df[f'{feat_name}_sin'] = np.sin(df[feat_name].values)

    # Insert zenith states in dataframe (needed for gym.environment to use zenith state as first state)
    zenith_df = get_zenith_features(original_df=df, is_pointing=True, hpGrid=hpGrid, base_pointing_feature_names=base_pointing_feature_names, cyclical_feature_names=cyclical_feature_names)
    df = pd.concat([df, zenith_df], ignore_index=True)
    df = df.sort_values(by='timestamp')

    # Ensure all data are 32-bit precision before training
    for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = df.select_dtypes(include=[str_bit]).columns
        df[cols] = df[cols].astype(np_bit)
    return df

def get_zenith_features(original_df, is_pointing=True, hpGrid=None, base_pointing_feature_names=None, cyclical_feature_names=None):
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
            row_dict['ra'], row_dict['dec'] = blanco.radec_of('0', '90')
            if not hpGrid.is_azel: 
                row_dict['bin'] = hpGrid.ang2idx(lon=row_dict['ra'], lat=row_dict['dec'])
            else:
                row_dict['bin'] = hpGrid.ang2idx(lon=0, lat=np.pi/2)
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

        for feat_name in base_pointing_feature_names:
            if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in cyclical_feature_names):
                zenith_df[f'{feat_name}_cos'] = np.cos(zenith_df[feat_name].values)
                zenith_df[f'{feat_name}_sin'] = np.sin(zenith_df[feat_name].values)

    return zenith_df

def calculate_and_add_bin_features(pt_df, datetimes, hpGrid, base_bin_feature_names, bin_feature_names, cyclical_feature_names, do_cyclical_norm):

    # NOTE: The timestamps in df are assumed to already have zenith -- they were added when calculating zeniths for pointing features
    # Create empty arrays 
    timestamps=pt_df['timestamp'].values
    hour_angles = np.empty(shape=(len(timestamps), len(hpGrid.idx_lookup)))
    airmasses = np.empty_like(hour_angles)
    moon_dists = np.empty_like(hour_angles)
    xs = np.empty_like(hour_angles) # xs = az if actions are in ra, dec
    ys = np.empty_like(hour_angles) # ys = dec if actions are in az, el
    num_visits_hist = np.zeros_like(hour_angles, dtype=np.int32)
    num_visits_tracking = np.zeros_like(hour_angles[0])

    do_night_num_visits = any("night_num_visits" in name for name in base_bin_feature_names)
    do_night_num_unvisited_fields = any("night_num_unvisited_fields" in name for name in base_bin_feature_names)
    do_night_num_incomplete_fields = any("night_num_incomplete_fields" in name for name in base_bin_feature_names)
    do_survey_num_visits = any("num_visits_hist" in name for name in base_bin_feature_names)
    do_ha = any("hour_angle" in name for name in base_bin_feature_names)
    do_airmass = any("airmass" in name for name in base_bin_feature_names)
    do_moon_dist = any("moon_distance" in name for name in base_bin_feature_names)
    do_coords = any("az" in name or "ra" in name for name in base_bin_feature_names)

    lon, lat = hpGrid.lon, hpGrid.lat
    for i, time in tqdm(enumerate(timestamps), total=len(timestamps), desc='Calculating bin features for all healpix bins and timestamps'):
        if do_ha:
            hour_angles[i] = hpGrid.get_hour_angle(time=time)
        if do_airmass:
            airmasses[i] = hpGrid.get_airmass(time)
        if do_moon_dist:
            moon_dists[i] = hpGrid.get_source_angular_separations('moon', time=time)
        if hpGrid.is_azel and do_coords:
            xs[i], ys[i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
        elif not hpGrid.is_azel and do_coords:
            xs[i], ys[i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
        # Tracks number of bins since start of *survey*
        if do_survey_num_visits:
            if pt_df.iloc[i]['object'] != 'zenith':
                bin_num = pt_df.iloc[i]['bin']
                num_visits_tracking[bin_num] += 1
            num_visits_hist[i] = num_visits_tracking.copy()

    if any([do_night_num_visits, do_night_num_unvisited_fields, do_night_num_incomplete_fields]):
        night_num_visits = np.zeros(shape=(len(pt_df), len(hpGrid.idx_lookup)))
        night_num_unvisited_fields = np.zeros_like(night_num_visits)
        night_num_incomplete_fields = np.zeros_like(night_num_visits)
        index = -1
        for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
            if not hpGrid.is_azel:
                unique_field_ids, unique_field_counts = np.unique(group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32), return_counts=True)
                # unique_bin_ids, unique_bin_counts = np.unique(group['bin'][group['object'] != 'zenith'], return_counts=True)
                field2nvisits = {int(fid): int(c) for fid, c in zip(unique_field_ids, unique_field_counts)}
                # bin2nvisits = {int(bid): int(c) for bid, c in zip(unique_bin_ids, unique_bin_counts)}
                total_num_observations = len(group)

                # Build bin2fields map
                bins_arr = group['bin'].to_numpy(dtype=np.int32)
                fields_arr = group['field_id'].to_numpy(dtype=np.int32)
                bin2fields_in_bin = {}
                for b, f in zip(bins_arr, fields_arr):
                    if f != -1:
                        bin2fields_in_bin.setdefault(int(b), set()).add(int(f))
                
                field_visit_counter = defaultdict(int)
                num_visits_tracking = np.zeros_like(night_num_visits[0])
                num_unvisited_fields_tracking = np.zeros_like(night_num_visits[0])
                num_incomplete_fields_tracking = np.zeros_like(night_num_visits[0])

                for i_row in range(len(group)):
                    index += 1
                    field_id = int(fields_arr[i_row])
                    bin_num = int(bins_arr[i_row])
                    if field_id == -1:
                        # initialize once per night
                        for bid, flist in bin2fields_in_bin.items():
                            num_unvisited_fields_tracking[bid] = len(flist)
                            num_incomplete_fields_tracking[bid] = len(flist)
                    else:
                        field_visit_counter[field_id] += 1
                        
                        # Add one visit to bin
                        num_visits_tracking[bin_num] += 1
            
                        # If this is the first time this field is being visited, subtract one field from tracker
                        if field_visit_counter[field_id] == 1:
                            num_unvisited_fields_tracking[bin_num] -= 1
            
                        # If field has been fully visited, subtract one field from tracker
                        if field_visit_counter[field_id] == field2nvisits[field_id]:
                            num_incomplete_fields_tracking[bin_num] -= 1
                    night_num_visits[index] = num_visits_tracking / total_num_observations
                    night_num_unvisited_fields[index] = num_unvisited_fields_tracking / total_num_observations
                    night_num_incomplete_fields[index] = num_incomplete_fields_tracking / total_num_observations
                
                #TODO: some field dithers are in different bins - need to account for this when calculating unvisited and incomplete fields, currently undercounting (very rarely occurs)
                # sanity check (checks above)
                # if not np.all(num_incomplete_fields_tracking[unique_bin_ids] == 0):
                #     print("Night", night, "incomplete:", 
                #         num_incomplete_fields_tracking[unique_bin_ids])

    # Need to update this to account for choice of bin features in input cfg
    stacked = np.stack([hour_angles, airmasses, moon_dists, xs, ys, num_visits_hist], axis=2) # Order must be exactly same as base_bin_feature_names
    stacked = np.stack([xs, ys, night_num_incomplete_fields, night_num_unvisited_fields, night_num_visits], axis=2) # Order must be exactly same as base_bin_feature_names
    bin_states = stacked.reshape(len(hour_angles), -1)
    bin_df = pd.DataFrame(data=bin_states, columns=base_bin_feature_names)
    bin_df['night'] = (datetimes - pd.Timedelta(hours=12)).dt.normalize()
    bin_df['timestamp'] = timestamps
    # Normalize periodic features here and add as df cols
    new_cols = {}
    if do_cyclical_norm:
        for feat_name in tqdm(base_bin_feature_names, total=len(base_bin_feature_names), desc='Normalizing bin features'):
            if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names):
                new_cols[f'{feat_name}_cos'] = np.cos(bin_df[feat_name].values)
                new_cols[f'{feat_name}_sin'] = np.sin(bin_df[feat_name].values)
    
    new_cols_df = pd.DataFrame(data=new_cols)
    bin_df = pd.concat([bin_df, new_cols_df], axis=1)

    bin_df = bin_df.reset_index(drop=True, inplace=False)
    bin_df = bin_df.sort_values(by='timestamp')
    
    # Make sure there are no missing columns
    missing_cols = set(bin_feature_names) - set(bin_df.columns) == 0
    assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'

    # Ensure all data are 32-bit precision before training
    for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = bin_df.select_dtypes(include=[str_bit]).columns
        bin_df[cols] = bin_df[cols].astype(np_bit)
    
    return bin_df

def calculate_and_add_bin_features(pt_df, datetimes, hpGrid, base_bin_feature_names, bin_feature_names, cyclical_feature_names, do_cyclical_norm):
    """
    Calculate bin features dynamically based on requested feature names.
    
    This version:
    1. Only calculates features that are present in base_bin_feature_names
    2. Dynamically constructs the output arrays based on which features are requested
    """
    
    # Create empty arrays 
    timestamps = pt_df['timestamp'].values
    n_timestamps = len(timestamps)
    n_bins = len(hpGrid.idx_lookup)
    
    # Determine which features to calculate by parsing base_bin_feature_names
    # Extract unique feature types from the bin feature names
    feature_types = set()
    for name in base_bin_feature_names:
        # Extract feature type from names like 'bin_0_hour_angle', 'bin_0_airmass', etc.
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == 'bin':
            feature_type = '_'.join(parts[2:])  # Everything after 'bin_X_'
            feature_types.add(feature_type)
    
    # Determine what to calculate
    do_night_num_visits = "night_num_visits" in feature_types
    do_night_num_unvisited_fields = "night_num_unvisited_fields" in feature_types
    do_night_num_incomplete_fields = "night_num_incomplete_fields" in feature_types
    do_survey_num_visits = "survey_num_visits_hist" in feature_types
    do_ha = "ha" in feature_types
    do_airmass = "airmass" in feature_types
    do_moon_dist = "moon_distance" in feature_types
    do_ra = "ra" in feature_types
    do_dec = "dec" in feature_types
    do_az = "az" in feature_types
    do_el = "el" in feature_types
    
    # Initialize arrays only for features we need
    calculated_features = {}
    
    if do_ha:
        calculated_features['ha'] = np.empty(shape=(n_timestamps, n_bins))
    if do_airmass:
        calculated_features['airmass'] = np.empty(shape=(n_timestamps, n_bins))
    if do_moon_dist:
        calculated_features['moon_distance'] = np.empty(shape=(n_timestamps, n_bins))
    if do_ra or do_az:
        calculated_features['xs'] = np.empty(shape=(n_timestamps, n_bins))  # az or ra
    if do_dec or do_el:
        calculated_features['ys'] = np.empty(shape=(n_timestamps, n_bins))  # el or dec
    if do_survey_num_visits:
        calculated_features['survey_num_visits_hist'] = np.zeros(shape=(n_timestamps, n_bins), dtype=np.int32)
        num_visits_tracking = np.zeros(n_bins, dtype=np.int32)
    
    # Calculate per-timestamp features
    lon, lat = hpGrid.lon, hpGrid.lat
    for i, time in tqdm(enumerate(timestamps), total=n_timestamps, desc='Calculating bin features for all healpix bins and timestamps'):
        if do_ha:
            calculated_features['ha'][i] = hpGrid.get_hour_angle(time=time)
        if do_airmass:
            calculated_features['airmass'][i] = hpGrid.get_airmass(time)
        if do_moon_dist:
            calculated_features['moon_distance'][i] = hpGrid.get_source_angular_separations('moon', time=time)
        
        # Coordinate transformations
        if hpGrid.is_azel and (do_ra or do_dec):
            xs, ys = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
            if do_ra:
                calculated_features['xs'][i] = xs
            if do_dec:
                calculated_features['ys'][i] = ys
        elif not hpGrid.is_azel and (do_az or do_el):
            xs, ys = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
            if do_az:
                calculated_features['xs'][i] = xs
            if do_el:
                calculated_features['ys'][i] = ys
        
        # Track number of visits since start of survey
        if do_survey_num_visits:
            if pt_df.iloc[i]['object'] != 'zenith':
                bin_num = pt_df.iloc[i]['bin']
                num_visits_tracking[bin_num] += 1
            calculated_features['num_visits_hist'][i] = num_visits_tracking.copy()
    
    # Calculate night-based features if needed
    if any([do_night_num_visits, do_night_num_unvisited_fields, do_night_num_incomplete_fields]):
        if do_night_num_visits:
            calculated_features['night_num_visits'] = np.zeros(shape=(n_timestamps, n_bins))
        if do_night_num_unvisited_fields:
            calculated_features['night_num_unvisited_fields'] = np.zeros(shape=(n_timestamps, n_bins))
        if do_night_num_incomplete_fields:
            calculated_features['night_num_incomplete_fields'] = np.zeros(shape=(n_timestamps, n_bins))
        
        index = -1
        for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
            if not hpGrid.is_azel:
                unique_field_ids, unique_field_counts = np.unique(
                    group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32), 
                    return_counts=True
                )
                field2nvisits = {int(fid): int(c) for fid, c in zip(unique_field_ids, unique_field_counts)}
                total_num_observations = len(group)
                
                # Build bin2fields map
                bins_arr = group['bin'].to_numpy(dtype=np.int32)
                fields_arr = group['field_id'].to_numpy(dtype=np.int32)
                bin2fields_in_bin = {}
                for b, f in zip(bins_arr, fields_arr):
                    if f != -1:
                        bin2fields_in_bin.setdefault(int(b), set()).add(int(f))
                
                field_visit_counter = defaultdict(int)
                num_visits_tracking = np.zeros(n_bins)
                num_unvisited_fields_tracking = np.zeros(n_bins)
                num_incomplete_fields_tracking = np.zeros(n_bins)
                
                for i_row in range(len(group)):
                    index += 1
                    field_id = int(fields_arr[i_row])
                    bin_num = int(bins_arr[i_row])
                    
                    if field_id == -1:
                        # Initialize once per night
                        for bid, flist in bin2fields_in_bin.items():
                            num_unvisited_fields_tracking[bid] = len(flist)
                            num_incomplete_fields_tracking[bid] = len(flist)
                    else:
                        field_visit_counter[field_id] += 1
                        num_visits_tracking[bin_num] += 1
                        
                        if field_visit_counter[field_id] == 1:
                            num_unvisited_fields_tracking[bin_num] -= 1
                        
                        if field_visit_counter[field_id] == field2nvisits[field_id]:
                            num_incomplete_fields_tracking[bin_num] -= 1
                    
                    if do_night_num_visits:
                        calculated_features['night_num_visits'][index] = num_visits_tracking / total_num_observations
                    if do_night_num_unvisited_fields:
                        calculated_features['night_num_unvisited_fields'][index] = num_unvisited_fields_tracking / total_num_observations
                    if do_night_num_incomplete_fields:
                        calculated_features['night_num_incomplete_fields'][index] = num_incomplete_fields_tracking / total_num_observations
    
    # Map coordinate features to their proper names based on grid type
    if 'xs' in calculated_features:
        if hpGrid.is_azel:
            calculated_features['ra'] = calculated_features.pop('xs')
        else:
            calculated_features['az'] = calculated_features.pop('xs')
    
    if 'ys' in calculated_features:
        if hpGrid.is_azel:
            calculated_features['dec'] = calculated_features.pop('ys')
        else:
            calculated_features['el'] = calculated_features.pop('ys')
    
    # Dynamically stack features in the order they appear in base_bin_feature_names
    # First, determine the unique feature types and their order
    feature_order = []
    seen_features = set()
    for name in base_bin_feature_names:
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == 'bin':
            feature_type = '_'.join(parts[2:])
            if feature_type not in seen_features:
                feature_order.append(feature_type)
                seen_features.add(feature_type)
    
    # Stack the features in order
    feature_arrays = []
    for feat in feature_order:
        if feat in calculated_features:
            feature_arrays.append(calculated_features[feat])
    
    # Stack and reshape to create the final feature matrix
    if feature_arrays:
        stacked = np.stack(feature_arrays, axis=2)  # shape: (n_timestamps, n_bins, n_features)
        bin_states = stacked.reshape(n_timestamps, -1)  # shape: (n_timestamps, n_bins * n_features)
    else:
        bin_states = np.empty((n_timestamps, 0))
    
    # Create DataFrame
    bin_df = pd.DataFrame(data=bin_states, columns=base_bin_feature_names)
    bin_df['night'] = (datetimes - pd.Timedelta(hours=12)).dt.normalize()
    bin_df['timestamp'] = timestamps
    
    # Normalize periodic features
    new_cols = {}
    if do_cyclical_norm:
        for feat_name in tqdm(base_bin_feature_names, total=len(base_bin_feature_names), desc='Normalizing bin features'):
            if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names):
                new_cols[f'{feat_name}_cos'] = np.cos(bin_df[feat_name].values)
                new_cols[f'{feat_name}_sin'] = np.sin(bin_df[feat_name].values)
    
    new_cols_df = pd.DataFrame(data=new_cols)
    bin_df = pd.concat([bin_df, new_cols_df], axis=1)
    
    bin_df = bin_df.sort_values(by='timestamp')
    bin_df = bin_df.reset_index(drop=True, inplace=False)
    
    # Make sure there are no missing columns
    missing_cols = set(bin_feature_names) - set(bin_df.columns)
    assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'
    
    # Ensure all data are 32-bit precision before training
    for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = bin_df.select_dtypes(include=[str_bit]).columns
        bin_df[cols] = bin_df[cols].astype(np_bit)
    
    return bin_df