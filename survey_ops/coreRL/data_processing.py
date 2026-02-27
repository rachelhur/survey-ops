import fitsio
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from datetime import timezone
import ephem
from astropy.time import Time
import torch

from survey_ops.utils import ephemerides, units
from tqdm import tqdm
logger = logging.getLogger(__name__)

    # Filter wavelengths (nm) according to obztak https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
FILTER2WAVE = {
    'u': 380,
    'g': 480,
    'r': 640,
    'i': 780,
    'z': 920,
    'Y': 990
}
FILTERWAVENORM = 1000.
FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}
NUM_FILTERS = len(FILTER2IDX)


# --- DATAFRAME PROCESSING --- #
# TODO make into class
def load_raw_data_to_dataframe(fits_path):
    d = fitsio.read(fits_path)
    df = pd.DataFrame(d.astype(d.dtype.newbyteorder('='))) # Big-endian/little-endian error

    sel = (df['propid'] == '2012B-0001') & (df['exptime'] > 40) & (df['exptime'] < 100) & (~np.isnan(df['teff']))
    df = df[sel].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
    df = df[df['datetime'].dt.year > 2010] # There are some 1970 rows even after selecting propid

    # Add timestamp col
    utc = pd.to_datetime(df['datetime'], utc=True)
    timestamps = (utc.astype('int64') // 10**9).values
    df['timestamp'] = timestamps.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def drop_rows_in_DECam_data(df, objects_to_remove, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    df = remove_dates(df, specific_years, specific_months, specific_days, specific_filters)
    
    # Remove specific nights according to object name
    # df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    pattern = '|'.join(objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False)

    # Filter the DataFrame
    df = df[mask]

    # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
    df = relabel_mislabelled_objects(df)
    df = remove_outliers(df)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_outliers(df):
    """Removes objects that have (outlier) in its object name"""
    df = df[~df['object'].astype(str).str.contains('(outlier)', regex=False, na=False)]
    return df

def remove_specific_objects(df, objects_to_remove):
    nights_with_special_fields = set()
    for i, spec_obj in enumerate(objects_to_remove):
        for night, subdf in df.groupby('night'):
            if any(spec_obj in obj_name for obj_name in subdf['object'].values) or any(subdf['object'] == ""):
                nights_with_special_fields.add(night)
    nights_to_remove_mask = df['night'].isin(nights_with_special_fields)

    df = df[~nights_to_remove_mask]
    assert not df.empty, "All nights have special fields"
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

def remove_dates(df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Processes and filters the dataframe to return a new dataframe with added columns for current global state features"""
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
    assert not df.empty, "No observations found for the specified year/month/day/filter selections."
    return df

def get_zenith_features(original_df):
    """
    Constructs dataframe with zenith features for each night in the original_df.
    Assumes zenith starts 10 seconds before the first observation.
    """
    zenith_datetimes = (original_df.groupby('night').head(1).datetime - pd.Timedelta(seconds=10)).values
    zenith_timestamps = zenith_datetimes.astype(np.int64) // 10 ** 9
    zenith_rows = []
    nights = original_df.night.unique()
    for i_row, time in tqdm(enumerate(zenith_timestamps), total=len(zenith_timestamps), desc='Calculating zenith states'):
        row_dict = {}
        row_dict['timestamp'] = time
        row_dict['night'] = nights[i_row]
        row_dict['datetime'] = zenith_datetimes[i_row]
        blanco = ephemerides.blanco_observer(time=time)
        row_dict['ra'], row_dict['dec'] = np.array(blanco.radec_of('0',  '90')) / units.deg
        zenith_rows.append(row_dict)

    zenith_df = pd.DataFrame(zenith_rows)
    zenith_df['az'] = 0
    zenith_df['el'] = 90
    zenith_df['airmass'] = 1
    zenith_df['zd'] = 0
    zenith_df['ha'] = 0
    zenith_df['object'] = 'zenith'
    zenith_df['field_id'] = -1
    zenith_df['datetime'] = pd.to_datetime(zenith_df['datetime'], utc=True)
    zenith_df['night'] = pd.to_datetime(zenith_df['night'], utc=True)

    return zenith_df

def calculate_and_add_global_features(df, field2name, hpGrid, global_feature_names, 
                      base_global_feature_names, cyclical_feature_names, do_cyclical_norm):
    """Processes and filters the dataframe to return a new dataframe with added columns for current global state features"""
    # Sort df by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Insert zenith states in dataframe
    zenith_df = get_zenith_features(original_df=df)
    df = pd.concat([df, zenith_df], ignore_index=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 2. Calculate LST for the whole column at once
    if 'lst' in base_global_feature_names:
        t_arr = Time(df['datetime'].values, format='datetime64', scale='utc')
        lst_obj = t_arr.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        df['lst'] = lst_obj.radian
        df['lst_hours'] = lst_obj.hour # for debugging

    # Get time dependent features (sun and moon pos)
    for idx, time in tqdm(zip(df.index, df['timestamp'].values), total=len(df['timestamp']), desc='Calculating sun and moon ra/dec and az/el'):
        sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
        df.loc[idx, ['sun_ra', 'sun_dec']] = sun_ra, sun_dec
        df.loc[idx, ['sun_az', 'sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)

        moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
        df.loc[idx, ['moon_ra', 'moon_dec']] = moon_ra, moon_dec
        df.loc[idx, ['moon_az', 'moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

    # Use first and last observation in night of offline dataset as time start and end
    # df['time_fraction_since_start'] = df.groupby('night')['timestamp'].transform(lambda x: (x - x.values[0]) / (x.values[-1] - x.values[0]) if len(x) > 1 else 0)

    def normalize_times(time_series):
        sunset = get_nautical_twilight(time_series.median(), event_type='set')
        sunrise = get_nautical_twilight(time_series.median(), event_type='rise')
        total_time = sunrise - sunset

        time_series = (time_series - sunset) / total_time
        assert all(time_series.values > 0) and all(time_series.values < 1), "Time fractions should be between 0 and 1"
        return time_series

    # Using nautical twilight for time start and end
    df['time_fraction_since_start'] = df.groupby('night')['timestamp'].transform(normalize_times)

    assert all(df['time_fraction_since_start'].values > 0) and all(df['time_fraction_since_start'].values < 1), "Time fractions should be between 0 and 1"  
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
        # df['bin'][df['object'] == 'zenith'] = -1  # assign zenith bin of -1
        df.loc[df['object'] == 'zenith', "bin"] = -1
        df.loc[df['object'] == 'zenith', "field_id"] = -1 # Need to re-assign zenith field_id bc df['object'].map(...) above will assign zenith the field_id of the field with object name 'zenith', but this field is mis-labelled and not actually the zenith field. #TODO should fix this in field2name

    df['filter_wave'] = df['filter'].map(FILTER2WAVE)
    df['filter_wave'] = df['filter_wave'].fillna(0.) / FILTERWAVENORM. # zenith "filter" set to 0, then normalize

    # Add other feature columns for those not present in dataframe
    for feat_name in base_global_feature_names:
        if feat_name in df.columns:
            continue
        else:
            raise NotImplementedError(f"Feature {feat_name} not found in dataframe columns, and no method to calculate it implemented.")
    # Normalize periodic features here and add as df cols
    if do_cyclical_norm:
        for feat_name in base_global_feature_names:
            if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in cyclical_feature_names):
                logger.info(f'Applying cyclical norm to {feat_name}')
                df[f'{feat_name}_cos'] = np.cos(df[feat_name].values)
                df[f'{feat_name}_sin'] = np.sin(df[feat_name].values)

    # Ensure all data are 32-bit precision before training
    for bin_str, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = df.select_dtypes(include=[bin_str]).columns
        df[cols] = df[cols].astype(np_bit)
    return df

def calculate_and_add_bin_features(pt_df, datetimes, hpGrid, base_bin_feature_names, prenorm_bin_feature_names, 
                                   bin_feature_names, cyclical_feature_names, do_cyclical_norm, night2fieldvisits,
                                   field2radec, field2maxvisits):
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
    
    # History based features
    history_based_features = [
        "num_unvisited_fields",
        "num_incomplete_fields",
        "min_tiling",
    ]
    do_history_based_features = any(hist_feat in base_feat for base_feat in base_bin_feature_names for hist_feat in history_based_features)

    # Instantaenous features
    do_angular_distance_to_pointing = "angular_distance_to_pointing" in base_bin_feature_names
    do_ha = "ha" in base_bin_feature_names
    do_airmass = "airmass" in base_bin_feature_names
    do_moon_dist = "moon_distance" in base_bin_feature_names
    do_ra = "ra" in base_bin_feature_names
    do_dec = "dec" in base_bin_feature_names
    do_az = "az" in base_bin_feature_names
    do_el = "el" in base_bin_feature_names
    
    # Initialize arrays only for features we need
    calculated_features = {}
    
    if do_ha:
        calculated_features['ha'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating ha for {n_timestamps} timestamps and {n_bins} bins")
    if do_airmass:
        calculated_features['airmass'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating airmass for {n_timestamps} timestamps and {n_bins} bins")
    if do_moon_dist:
        calculated_features['moon_distance'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating moon distance for {n_timestamps} timestamps and {n_bins} bins")
    if do_ra or do_az:
        calculated_features['xs'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # az or ra
        logger.debug(f"Calculating xs for {n_timestamps} timestamps and {n_bins} bins")
    if do_dec or do_el:
        calculated_features['ys'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # el or dec
        logger.debug(f"Calculating ys for {n_timestamps} timestamps and {n_bins} bins")
    if do_angular_distance_to_pointing:
        calculated_features['angular_distance_to_pointing'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
    
    # Calculate per-timestamp features
    lon, lat = hpGrid.lon, hpGrid.lat
    for i, time in tqdm(enumerate(timestamps), total=n_timestamps, desc='Calculating bin features for all healpix bins and timestamps'):
        if do_ha:
            calculated_features['ha'][i] = hpGrid.get_hour_angle(time=time)
        if do_airmass:
            calculated_features['airmass'][i] = hpGrid.get_airmass(time)
        if do_moon_dist:
            calculated_features['moon_distance'][i] = hpGrid.get_source_angular_separations('moon', time=time)
        if do_angular_distance_to_pointing:
            if hpGrid.is_azel:
                current_lon, current_lat = pt_df.iloc[i]['az'], pt_df.iloc[i]['el']
            else:
                current_lon, current_lat = pt_df.iloc[i]['ra'], pt_df.iloc[i]['dec']
            calculated_features['angular_distance_to_pointing'][i] = hpGrid.get_angular_separations(lon=current_lon, lat=current_lat)
        
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
        
    # Calculate night-based features if needed
    if do_history_based_features:
        calculated_night_history_features = calculate_night_history_bin_features(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, night2visithistory=night2fieldvisits, field2maxvisits=field2maxvisits)
        calculated_features = calculated_features | calculated_night_history_features
    # Map coordinate features to their proper names based on grid type
    if 'xs' in calculated_features:
        if hpGrid.is_azel:
            calculated_features['ra'] = calculated_features['xs']
        else:
            calculated_features['az'] = calculated_features['xs']
    
    if 'ys' in calculated_features:
        if hpGrid.is_azel:
            calculated_features['dec'] = calculated_features['ys']
        else:
            calculated_features['el'] = calculated_features['ys']
    
    # Dynamically stack features in the order they appear in base_bin_feature_names
    # First, determine the unique feature types and their order
    feature_order = []
    seen_features = set()
    for name in prenorm_bin_feature_names:
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
            feature_arrays.append(calculated_features[feat].astype(np.float32))
        else:
            raise ValueError(f"Feature {feat} is in prenorm_bin_feature_names but was not calculated. Check if it's included in base_bin_feature_names and if the calculation code is implemented.")
    
    # Stack and reshape to create the final feature matrix
    if feature_arrays:
        stacked = np.stack(feature_arrays, axis=2)  # shape: (n_timestamps, n_bins, n_features)
        bin_states = stacked.reshape(n_timestamps, -1)  # shape: (n_timestamps, n_bins * n_features)
    else:
        bin_states = np.empty((n_timestamps, 0))
    
    # ------------- from previous implementation -----------------#
    # # Create DataFrame
    # bin_df = pd.DataFrame(data=bin_states, columns=prenorm_bin_feature_names)
    # bin_df['night'] = (datetimes - pd.Timedelta(hours=12)).dt.normalize()
    # bin_df['timestamp'] = timestamps
    
    # # Normalize periodic features
    # new_cols = {}
    
    # if do_cyclical_norm:
    #     for feat_name in tqdm(prenorm_bin_feature_names, total=len(prenorm_bin_feature_names), desc='Normalizing bin features'):
    #         if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names):
    #             new_cols[f'{feat_name}_cos'] = np.cos(bin_df[feat_name].values)
    #             new_cols[f'{feat_name}_sin'] = np.sin(bin_df[feat_name].values)

    # new_cols_df = pd.DataFrame(data=new_cols)
    # bin_df = pd.concat([bin_df, new_cols_df], axis=1)
    
    # bin_df = bin_df.sort_values(by='timestamp')
    # bin_df = bin_df.reset_index(drop=True, inplace=False)
    
    # # Make sure there are no missing columns
    # missing_cols = set(bin_feature_names) - set(bin_df.columns)
    # assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'
    
    # # Ensure all data are 32-bit precision before training
    # for str_bit, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
    #     cols = bin_df.select_dtypes(include=[str_bit]).columns
    #     bin_df[cols] = bin_df[cols].astype(np_bit)
    # ------------- from previous implementation -----------------#

    final_col_names = prenorm_bin_feature_names.copy()
    if do_cyclical_norm:
        cyclical_cols = []
        cyclical_names = []
        # Identify indices of columns that need cyclic norm
        for i, feat_name in enumerate(prenorm_bin_feature_names):
            if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names):
                cyclical_cols.append(i)
                cyclical_names.extend([f'{feat_name}_cos', f'{feat_name}_sin'])
        
        if cyclical_cols:
            base_cyclical_data = bin_states[:, cyclical_cols]
            
            # Pre-allocate array for the new cos/sin features
            new_data = np.empty((n_timestamps, len(cyclical_cols) * 2), dtype=np.float32)
            new_data[:, 0::2] = np.cos(base_cyclical_data)
            new_data[:, 1::2] = np.sin(base_cyclical_data)
            
            # Horizontally stack the new data
            bin_states = np.hstack([bin_states, new_data])
            final_col_names.extend(cyclical_names)

    sort_idx = np.argsort(timestamps)
    bin_states = bin_states[sort_idx]
    sorted_timestamps = timestamps[sort_idx]
    
    # Map datetimes efficiently using np datetime arrays
    sorted_nights = (datetimes.values[sort_idx] - np.timedelta64(12, 'h')).astype('datetime64[D]')

    bin_df = pd.DataFrame(data=bin_states, columns=final_col_names, copy=False)
    bin_df['night'] = sorted_nights
    bin_df['timestamp'] = sorted_timestamps
    
    # Make sure there are no missing columns
    missing_cols = set(bin_feature_names) - set(bin_df.columns)
    assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'
    
    return bin_df

def normalize_noncyclic_features(state, 
                                state_feature_names,
                                max_norm_feature_names,
                                ang_distance_norm_feature_names,
                                do_inverse_norm, do_max_norm, do_ang_distance_norm,
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

    ang_distance_mask = np.array(
        [any(dist_feat in feat for dist_feat in ang_distance_norm_feature_names)
         for feat in state_feature_names
         ]
    )
    if is_torch:
        airmass_mask = torch.tensor(airmass_mask, dtype=torch.bool, device=state.device)
        max_norm_mask = torch.tensor(max_norm_mask, dtype=torch.bool, device=state.device)
        ang_distance_mask = torch.tensor(ang_distance_mask, dtype=torch.bool, device=state.device)

    do_reshape = False

    if state.ndim == 3: # ie, if is bin states
        do_reshape = True
        nrows, nbins, nfeats_per_bin = state.shape
        state = state.flatten(start_dim=1)
            
    if do_inverse_norm:
        state[..., airmass_mask] = 1.0 / state[..., airmass_mask]

    if do_max_norm:
        state[..., max_norm_mask] = state[..., max_norm_mask] / (np.pi / 2)

    if do_ang_distance_norm:
        state[..., ang_distance_mask] = state[..., ang_distance_mask] / np.pi
    
    if fix_nans:
        if is_torch:
            state[torch.isnan(state)] = 1.2
        else:
            state[np.isnan(state)] = 1.2
    
    if do_reshape:
        state = state.reshape(nrows, nbins, nfeats_per_bin)

    return state

def get_nautical_twilight(timestamp, event_type='set'):
    obs = ephemerides.blanco_observer(time=timestamp)
    obs.horizon = '-10'
    sun = ephem.Sun()
    
    if event_type == 'rise':
        ephem_date = obs.next_rising(sun).datetime()
    elif event_type == 'set':
        ephem_date = obs.previous_setting(sun).datetime()
    else:
        raise NotImplementedError

    dt_utc = ephem_date.replace(tzinfo=timezone.utc)
    return dt_utc.timestamp()

def get_sun_rise_and_set_times(df):
    rise_times = df.groupby('night').apply(get_nautical_twilight, event_type='rise').values
    set_times = df.groupby('night').apply(get_nautical_twilight, event_type='set').values
    return rise_times, set_times
    
def get_sun_rise_and_set_azel(df):
    rise_times, set_times = get_sun_rise_and_set_times(df)
    rise_azels = np.empty(shape=(len(set_times), 2))
    set_azels = np.empty(shape=(len(set_times), 2))
    
    for i, time in enumerate(rise_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        rise_azels[i] = np.array([sun_az, sun_el])
    for i, time in enumerate(set_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        set_azels[i] = np.array([sun_az, sun_el])

    return rise_azels, set_azels

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

def setup_feature_names(include_default_features, additional_global_features, additional_bin_features,
                        default_global_feature_names, cyclical_feature_names, hpGrid, do_cyclical_norm,
                        grid_network):

    # include_bin_features = len(additional_global_features) > 0
    required_point_features = default_global_feature_names \
                                if include_default_features else []
    # required_bin_features = default_bin_feature_names \
                                # if (include_default_features and include_bin_features) else []
    
    # Include additional features not in default features above
    additional_global_features = [] if additional_global_features is None else additional_global_features

    base_global_feature_names = required_point_features + additional_global_features
    base_bin_feature_names = np.unique(np.array(additional_bin_features)).tolist()
    if grid_network == None:
        prenorm_bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}'
                                        for bin_feat in base_bin_feature_names]
                                        for bin_num in range(len(hpGrid.idx_lookup))])
        prenorm_bin_feature_names = prenorm_bin_feature_names.flatten().tolist()
    elif grid_network == 'single_bin_scorer':
        prenorm_bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}'
                            for bin_feat in base_bin_feature_names]
                            for bin_num in range(len(hpGrid.idx_lookup))])
        prenorm_bin_feature_names = prenorm_bin_feature_names.flatten().tolist()
    else:
        raise NotImplementedError(f"grid_network {grid_network} not supported")

    base_feature_names = base_global_feature_names + prenorm_bin_feature_names

    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(prenorm_bin_feature_names, cyclical_feature_names)
    
    if grid_network is None:
        state_feature_names = global_feature_names + bin_feature_names
    elif grid_network == 'single_bin_scorer':
        state_feature_names = global_feature_names

    return base_global_feature_names, base_bin_feature_names, base_feature_names, global_feature_names, \
        bin_feature_names, state_feature_names, prenorm_bin_feature_names

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

# def calculate_night_history_bin_features(pt_df, hpGrid):
#     if hpGrid.is_azel:
#         raise NotImplementedError("Night history features not yet implemented for azel")
#     # Assume global_index tracks the position in the entire dataset (pt_df)
#     global_index = 0 
#     n_bins = len(hpGrid.idx_lookup)
#     calculated_features = {}

#     # Loop through each night
#     for night, group in tqdm(pt_df.groupby('night'), total=pt_df['night'].nunique(), desc='Calculating night history bin features'):
        
#         # 1. PRE-CALCULATE CONSTANTS FOR THE NIGHT
#         n_rows = len(group)
#         total_obs = n_rows
        
#         # Get arrays for fast access
#         field_ids = group['field_id'].to_numpy(dtype=np.int32)
#         bin_ids = group['bin'].to_numpy(dtype=np.int32)
        
#         # Identify valid observations (not Zenith/Slew)
#         non_zenith_mask = field_ids != -1
        
#         # Calculate how many unique fields exist in each bin for this night (Baseline)
#         # Result: Series with index=bin_num, value=count_unique_fields
#         fields_per_bin = group[non_zenith_mask].groupby('bin')['field_id'].nunique()
        
#         # Create a dense array for initial "unvisited" counts per bin
#         # (Assuming n_bins is defined globally)
#         initial_bin_counts = np.zeros(n_bins)
#         initial_bin_counts[fields_per_bin.index] = fields_per_bin.values

#         # 2. IDENTIFY EVENTS (VISITS, NEW FIELDS, COMPLETIONS)
#         # We need to determine the status of every row without a loop
        
#         # A. Calculate visit counts for every row this night
#         field_cum_counts = group.groupby('field_id').cumcount().to_numpy()
#         # 'transform' gives total visits for that field in this night
#         field_total_counts = group.groupby('field_id')['field_id'].transform('count').to_numpy()
        
#         # B. Create Boolean Masks for events
#         # Is this row a valid visit?
#         is_visit = non_zenith_mask
#         # Is this the FIRST time this field is visited tonight? (count == 0)
#         is_first_visit = non_zenith_mask & (field_cum_counts == 0)
#         # Is this the LAST time this field is visited tonight? (count == total - 1)
#         is_completion = non_zenith_mask & (field_cum_counts == (field_total_counts - 1))

#         # 3. BROADCAST TO BINS (VECTORIZED STATE TRACKING)
#         # We create matrices of shape (n_rows, n_bins) and fill 1s where events happen
#         # Then we cumsum down the rows to get the "state" at every timestamp
        
#         # Helper to create cumulative event matrix
#         def get_cumulative_state(mask, bins, rows, n_bins):
#             # Create zero matrix
#             mat = np.zeros((len(rows), n_bins), dtype=np.int32)
#             # Set 1 at the (row, bin) where the event occurred
#             # Only process where mask is True
#             active_rows = rows[mask]
#             active_bins = bins[mask]
#             mat[active_rows, active_bins] = 1
#             # Cumulative sum down the rows
#             return mat.cumsum(axis=0)

#         rows_idx = np.arange(n_rows)
        
#         # State: Total visits per bin so far
#         cum_visits = get_cumulative_state(is_visit, bin_ids, rows_idx, n_bins)
        
#         # State: Number of NEW fields found per bin so far
#         cum_new_fields = get_cumulative_state(is_first_visit, bin_ids, rows_idx, n_bins)
        
#         # State: Number of COMPLETED fields per bin so far
#         cum_completions = get_cumulative_state(is_completion, bin_ids, rows_idx, n_bins)

#         # 4. CALCULATE FINAL FEATURES
#         # Normalize by total observations if required by your logic
#         norm_factor = total_obs if total_obs > 0 else 1.0
        
#         # Broadcasting: (1, n_bins) - (n_rows, n_bins)
#         # Unvisited = Initial_Total - Found_So_Far
#         night_unvisited = (initial_bin_counts[None, :] - cum_new_fields) / norm_factor
        
#         # Incomplete = Initial_Total - Completed_So_Far
#         night_incomplete = (initial_bin_counts[None, :] - cum_completions) / norm_factor
        
#         # Visits = Cumulative    # ---------------------------------------------------------
    # # MEMORY OPTIMIZATION: Create DataFrame ONCE
    # # ---------------------------------------------------------
    # bin_df = pd.DataFrame(data=bin_states, columns=final_col_names, copy=False)
    # bin_df['night'] = sorted_nights
    # bin_df['timestamp'] = sorted_timestamps
    
    # # Make sure there are no missing columns
    # missing_cols = set(bin_feature_names) - set(bin_df.columns)
    # assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'
    #  Visits
#         night_visits = cum_visits / norm_factor

#         # 5. ASSIGN TO OUTPUT (Slice Assignment)
#         # Assign the whole block for this night at once
#         start_idx = global_index
#         end_idx = global_index + n_rows
        
#         calculated_features['night_num_visits'][start_idx:end_idx] = night_visits
#         calculated_features['night_num_unvisited_fields'][start_idx:end_idx] = night_unvisited
#         calculated_features['night_num_incomplete_fields'][start_idx:end_idx] = night_incomplete

#         # Update global index
#         global_index = end_idx
#     return calculated_features

def calculate_night_history_bin_features(pt_df, hpGrid, night2visithistory, field2radec, field2maxvisits):
    n_bins = len(hpGrid.idx_lookup)
    calculated_features = {
        'night_num_visits': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_num_unvisited_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_num_incomplete_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_min_tiling': -.1 * np.ones((len(pt_df), n_bins), dtype=np.float32), # set bin's min tiling to -1 if there are no fields visible
        'survey_num_visits': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_num_unvisited_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_num_incomplete_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_min_tiling': -.1 * np.ones((len(pt_df), n_bins), dtype=np.float32) # set bin's min tiling to -1 if there are no fields visible
    }

    if hpGrid.is_azel:
        calculated_features = calculate_historical_bin_features_azel(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, calculated_features=calculated_features, night2visithistory=night2visithistory, field2maxvisits=field2maxvisits)
    else:
        # Assume global_index tracks the position in the entire dataset (pt_df)
        global_index = 0 

        # Loop through each night
        for night, group in tqdm(pt_df.groupby('night'), total=pt_df['night'].nunique(), desc='Calculating night history bin features'):
            # 1. PRE-CALCULATE CONSTANTS FOR THE NIGHT
            n_rows = len(group)
            total_obs = n_rows
            rows_idx = np.arange(n_rows)
            
            # Get arrays for fast access
            field_ids = group['field_id'].to_numpy(dtype=np.int32)
            bin_ids = group['bin'].to_numpy(dtype=np.int32)
            
            # Identify valid observations (not Zenith/Slew)
            non_zenith_mask = field_ids != -1
            
            # Calculate how many unique fields exist in each bin for this night (Baseline)
            fields_per_bin = group[non_zenith_mask].groupby('bin')['field_id'].nunique()
            initial_bin_counts = np.zeros(n_bins, dtype=np.int32)
            initial_bin_counts[fields_per_bin.index] = fields_per_bin.values

            # 2. IDENTIFY EVENTS (VISITS, NEW FIELDS, COMPLETIONS)
            field_cum_counts = group.groupby('field_id').cumcount().to_numpy()
            field_total_counts = group.groupby('field_id')['field_id'].transform('count').to_numpy()
            
            # Boolean Masks for events
            is_visit = non_zenith_mask
            is_first_visit = non_zenith_mask & (field_cum_counts == 0)
            is_completion = non_zenith_mask & (field_cum_counts == (field_total_counts - 1))

            # 3. BROADCAST TO BINS (VECTORIZED STATE TRACKING)
            def get_cumulative_state(mask, bins, rows, n_bins):
                mat = np.zeros((len(rows), n_bins), dtype=np.int32)
                active_rows = rows[mask]
                active_bins = bins[mask]
                mat[active_rows, active_bins] = 1
                return mat.cumsum(axis=0)

            cum_visits = get_cumulative_state(is_visit, bin_ids, rows_idx, n_bins)
            cum_new_fields = get_cumulative_state(is_first_visit, bin_ids, rows_idx, n_bins)
            cum_completions = get_cumulative_state(is_completion, bin_ids, rows_idx, n_bins)

            # --- NEW: MINIMUM TILING DEPTH VECTORIZATION ---
            min_tiling_increments = np.zeros((n_rows, n_bins), dtype=np.int32)
            valid_rows = rows_idx[non_zenith_mask]
            
            if len(valid_rows) > 0:
                valid_vnums = field_cum_counts[non_zenith_mask]
                valid_bins = bin_ids[non_zenith_mask]
                
                # Temporary DataFrame to quickly find the exact row a bin finishes a tiling level
                tiling_df = pd.DataFrame({
                    'row_idx': valid_rows,
                    'bin': valid_bins,
                    'vnum': valid_vnums
                })
                
                # Group by bin and the visit iteration (0 for 1st visit, 1 for 2nd, etc.)
                agg = tiling_df.groupby(['bin', 'vnum'])['row_idx'].agg(['count', 'max'])
                
                bins_in_agg = agg.index.get_level_values('bin').to_numpy()
                target_counts = initial_bin_counts[bins_in_agg]
                
                # A bin achieves this tiling depth when all its fields have been visited this many times
                mask_fully_tiled = (agg['count'].to_numpy() == target_counts) & (target_counts > 0)
                
                # Get the exact row index where the final field completed the depth
                tiled_rows = agg['max'].to_numpy()[mask_fully_tiled]
                tiled_bins = bins_in_agg[mask_fully_tiled]
                
                # Add +1 to the bin's tiling state at these exact rows
                np.add.at(min_tiling_increments, (tiled_rows, tiled_bins), 1)

            # Cumulative sum yields the active min_tiling state for every row in the night
            night_min_tiling = min_tiling_increments.cumsum(axis=0)
            # ------------------------------------------------

            # 4. CALCULATE FINAL FEATURES
            norm_factor = total_obs if total_obs > 0 else 1.0
            
            night_unvisited = (initial_bin_counts[None, :] - cum_new_fields) / norm_factor
            night_incomplete = (initial_bin_counts[None, :] - cum_completions) / norm_factor
            night_visits = cum_visits / norm_factor

            # 5. ASSIGN TO OUTPUT (Slice Assignment)
            start_idx = global_index
            end_idx = global_index + n_rows
            
            calculated_features['night_num_unvisited_fields'][start_idx:end_idx] = night_unvisited
            calculated_features['night_num_incomplete_fields'][start_idx:end_idx] = night_incomplete
            # Note: I am leaving min_tiling unnormalized (as an integer count) to match the likely intent, 
            # but you can divide by norm_factor here if you prefer.
            calculated_features['night_min_tiling'][start_idx:end_idx] = night_min_tiling

            global_index = end_idx

    for key, arr in calculated_features.items():
        if arr.min() < -.1 and arr.max() > 1.:
            logger.debug(f"{key} is not between 0 and 1. Array max/min={arr.max()}/{arr.min()}. Check normalization factor.")

    return calculated_features

def calculate_historical_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits):
    n_bins = len(hpGrid.idx_lookup)
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    fid2idx = np.full(fids.max() + 1, -1, dtype=np.int32)
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx

    ra_arr = np.array([field2radec[fid][0] for fid in fids])
    dec_arr = np.array([field2radec[fid][1] for fid in fids])
    max_v_arr = np.array([field2maxvisits[fid] for fid in fids], dtype=np.int32)

    # --- TIME CACHING VARIABLES ---
    cache_time = -1e9
    v_bins_cache = None
    active_bins_cache = None
    bin_count_cache = None
    valid_mask_cache = None

    global_idx = 0
    for night, group in tqdm(pt_df.groupby('night'), desc='Calculating AzEl History'):
        cur_survey_visits = night2visithistory[night][fids].copy().astype(np.int32)
        cur_night_visits = np.zeros(nfields, dtype=np.int32)
        
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]

            # 1. TIME CACHING: Refresh every 5 minutes (300s)
            if abs(timestamp - cache_time) > 300:
                az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
                bins_raw = hpGrid.ang2idx(lon=az, lat=el)
                
                # FIX: Explicitly handle None values and convert to numeric sentinel (-1)
                bins = np.array([b if b is not None else -1 for b in bins_raw], dtype=np.int32)
                
                # FIX: Refine horizon mask to exclude elevation <= 0 and invalid bins
                valid_mask = (el > 0) & (bins != -1)
                
                v_bins = bins[valid_mask]
                bin_count = np.bincount(v_bins, minlength=n_bins)
                active_bins = bin_count > 0
                
                cache_time, v_bins_cache, active_bins_cache, bin_count_cache, valid_mask_cache = \
                    timestamp, v_bins, active_bins, bin_count, valid_mask
            else:
                v_bins, active_bins, bin_count, valid_mask = \
                    v_bins_cache, active_bins_cache, bin_count_cache, valid_mask_cache

            # 2. CALCULATE STATE (BEFORE update visit counts)
            # This fixes the Data Leak
            v_survey_counts = cur_survey_visits[valid_mask]
            v_night_counts = cur_night_visits[valid_mask]
            v_max_v = max_v_arr[valid_mask]

            # Vectorized Proportions
            for key_n, key_s, mask_n, mask_s in [
                ('night_num_unvisited_fields', 'survey_num_unvisited_fields', v_night_counts == 0, v_survey_counts == 0),
                ('night_num_incomplete_fields', 'survey_num_incomplete_fields', v_night_counts < v_max_v, v_survey_counts < v_max_v)
            ]:
                res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), bin_count, out=res_n, where=active_bins)
                np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), bin_count, out=res_s, where=active_bins)
                calculated_features[key_n][global_idx] = res_n
                calculated_features[key_s][global_idx] = res_s

            # Vectorized Min Tiling
            s_tiling_all, n_tiling_all = v_survey_counts / v_max_v, v_night_counts / v_max_v
            s_mins, n_mins = np.full(n_bins, 2.0, dtype=np.float32), np.full(n_bins, 2.0, dtype=np.float32)
            
            np.minimum.at(s_mins, v_bins, s_tiling_all)
            np.minimum.at(n_mins, v_bins, n_tiling_all)
            s_mins[~active_bins], n_mins[~active_bins] = -0.1, -0.1
            
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins

            # 3. UPDATE VISITS (At the end of the loop)
            if obs_fid != -1:
                idx = fid2idx[obs_fid]
                if idx != -1:
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1

            global_idx += 1
            
    return calculated_features

def old_calculate_historical_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits):
    n_bins = len(hpGrid.idx_lookup)

    # Save (all) field radecs for quick access during loop
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    max_fid = fids[-1]

    # Field to index mapping for sparse field ids; unused fields maps to -1
    fid2idx = np.full(max_fid + 1, -1, dtype=np.int32)
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx

    # Get compact radec arrays - ie, skip fields not present in field2maxvisits
    ra_arr = np.zeros(nfields)
    dec_arr = np.zeros(nfields)
    max_v_arr = np.zeros(nfields, dtype=np.int32)
    for idx, fid in enumerate(fids):
        ra_arr[idx], dec_arr[idx] = field2radec[fid]
        max_v_arr[idx] = field2maxvisits[fid]

    # Row index
    global_idx = 0

    for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
        
        # Get field visit counts at start of night
        cur_survey_visits = night2visithistory[night][fids].copy().astype(np.int32)
        cur_night_visits = np.zeros(nfields, dtype=np.int32)

        # Speed up loop by extracting dataframe values beforehand
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]
            
            az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)

            bins = hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
            valid_mask = el > 0

            # Mask quantities whose associated field is below horizon
            v_bins = bins[valid_mask].astype(np.int32)
            v_survey_counts = cur_survey_visits[valid_mask].astype(np.int32)
            v_night_counts = cur_night_visits[valid_mask].astype(np.int32)
            v_max_v = max_v_arr[valid_mask].astype(np.int32)

            # Count total visible fields in each bin
            bin_count = np.bincount(v_bins, minlength=n_bins)
            active_bins = bin_count > 0

            # Num Unvisited fields
            s_unvisited = np.bincount(v_bins, weights=(v_survey_counts == 0), minlength=n_bins)
            n_unvisited = np.bincount(v_bins, weights=(v_night_counts == 0), minlength=n_bins)
            
            # Num Incomplete fields
            s_incomplete_mask = v_survey_counts < v_max_v
            s_incomplete = np.bincount(v_bins, weights=s_incomplete_mask, minlength=n_bins)
            n_incomplete_mask = v_night_counts < v_max_v
            n_incomplete = np.bincount(v_bins, weights=n_incomplete_mask, minlength=n_bins)
            
            # Create a zero-filled array for the results
            s_unvisited_frac = np.zeros_like(s_unvisited)
            n_unvisited_frac = np.zeros_like(n_unvisited)
            s_incomplete_frac = np.zeros_like(s_incomplete)
            n_incomplete_frac = np.zeros_like(n_incomplete)

            # Do division in-place (bypasses runtimewarning error )
            np.divide(s_unvisited, bin_count, out=s_unvisited_frac, where=active_bins)
            np.divide(n_unvisited, bin_count, out=n_unvisited_frac, where=active_bins)
            np.divide(s_incomplete, bin_count, out=s_incomplete_frac, where=active_bins)
            np.divide(n_incomplete, bin_count, out=n_incomplete_frac, where=active_bins)

            # Record to dictionary
            calculated_features['survey_num_unvisited_fields'][global_idx] = s_unvisited_frac
            calculated_features['night_num_unvisited_fields'][global_idx] = n_unvisited_frac
            calculated_features['survey_num_incomplete_fields'][global_idx] = s_incomplete_frac
            calculated_features['night_num_incomplete_fields'][global_idx] = n_incomplete_frac

            # # Min Tiling
            # unique_bins = np.where(active_bins)[0]
            # s_tiling_all = v_survey_counts / v_max_v
            # n_tiling_all = v_night_counts / v_max_v

            # for b in unique_bins:
            #     mask = v_bins == b
            #     calculated_features['survey_min_tiling'][global_idx, b] = np.min(s_tiling_all[mask])
            #     calculated_features['night_min_tiling'][global_idx, b] = np.min(n_tiling_all[mask])

            # --- VECTORIZED MIN TILING  --- #
            s_tiling_all = v_survey_counts / v_max_v
            n_tiling_all = v_night_counts / v_max_v

            # Init with sentinel -0.1, but use high value for intermediate min check
            s_mins = np.full(n_bins, 2.0, dtype=np.float32)
            n_mins = np.full(n_bins, 2.0, dtype=np.float32)
            
            np.minimum.at(s_mins, v_bins, s_tiling_all)
            np.minimum.at(n_mins, v_bins, n_tiling_all)
            
            # Reset bins with no fields back to -0.1
            s_mins[~active_bins] = -0.1
            n_mins[~active_bins] = -0.1
            
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins

            if obs_fid != -1:
                idx = fid2idx[obs_fid]
                if idx != -1: # Make sure fid is a valid field (for case of sparse field ids)
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1

            global_idx += 1
            
    return calculated_features


# def old_calculate_historical_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits):
    
#     # Assume global_index tracks the position in the entire dataset (pt_df)
#     global_idx = 0

#     survey_unique_field_ids = np.array(list(field2maxvisits))
#     survey_field_radecs = np.array([field2radec[fid] for fid in field2maxvisits.keys()])
#     survey_field_maxvisits = np.array([field2maxvisits[fid] for fid in survey_unique_field_ids]) 
    
#     for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
        
#         # Get field ids and radecs for all fields visited tonight
#         night_unique_field_ids, night_field_maxvisits = np.unique(group['field_id'][group['object'] != 'zenith'].to_numpy(dtype=np.int32), return_counts=True)
#         night_fieldradecs = np.array([field2radec[fid] for fid in night_unique_field_ids])

#         # Track night field visit counts
#         night_field_visit_counter = defaultdict(int)

#         # Get survey-wide historical field visit counts up to this night
#         survey_field_visit_counter = night2visithistory[night]
        
#         # For each timestep tonight, update each bin's features array_dims=(timestamp, nbins)
#         for fid, timestamp in zip(group['field_id'].values.astype(np.int32), group['timestamp'].values):
#             if fid != -1:
#                 night_field_visit_counter[fid] += 1

#             # ------------ FEATURES ASSUMING INFO FROM TONIGHT ONLY --------------- #
#             # Get bin membership of (tonight's) fields at this time
#             _az, _el = ephemerides.equatorial_to_topographic(ra=night_fieldradecs[:, 0], dec=night_fieldradecs[:, 1], time=timestamp)
#             field_bins = hpGrid.ang2idx(lon=_az, lat=_el)
#             above_horizon_fields_mask = field_bins != None
#             field_bins = field_bins[above_horizon_fields_mask]
#             bins_with_fields = np.unique(field_bins)

#             # For each bin that currently has fields, get historical features
#             for bid in bins_with_fields:
#                 bin_mask = field_bins == bid
#                 fids_in_bin = night_unique_field_ids[above_horizon_fields_mask][bin_mask]
#                 fid_max_counts = night_field_maxvisits[above_horizon_fields_mask][bin_mask]
#                 f_visit_counts = np.array(([night_field_visit_counter[fid] for fid in fids_in_bin]))
#                 night_num_fields_in_bin = len(fids_in_bin) # For normalizing
                
#                 # Get min tiling fraction
#                 night_min_tiling_fractions = f_visit_counts / fid_max_counts
#                 night_min_tiling_frac = night_min_tiling_fractions.min()
#                 calculated_features['night_min_tiling'][global_idx, bid] = night_min_tiling_frac # max tiling is 12

#                 # Get number of unvisited fields
#                 night_n_unvisited = np.sum(f_visit_counts == 0)
#                 calculated_features['night_num_unvisited_fields'][global_idx, bid] = night_n_unvisited / night_num_fields_in_bin

#                 # Get number of incomplete fields
#                 n_incomplete = np.sum(fid_max_counts != f_visit_counts)
#                 calculated_features['night_num_incomplete_fields'][global_idx, bid] = n_incomplete / night_num_fields_in_bin
#             # ------------------------------------------------------------ #

#             # ------------ FEATURES ASSUMING INFO FROM ENTIRE SURVEY --------------- #
#             # Now do get survey-wide features
#             # Get bin membership of (all) fields at this time
#             _az, _el = ephemerides.equatorial_to_topographic(ra=survey_field_radecs[:, 0], dec=survey_field_radecs[:, 1], time=timestamp)
#             survey_field_bins = hpGrid.ang2idx(lon=_az, lat=_el)
#             survey_above_horizon_mask = survey_field_bins != None
#             survey_field_bins = survey_field_bins[survey_above_horizon_mask]
#             survey_bins_with_fields = np.unique(survey_field_bins)

#             # For each bin with at least one field, update feature arrays
#             for bid in survey_bins_with_fields:
#                 survey_bin_mask = survey_field_bins == bid
#                 survey_fids_in_bin = survey_unique_field_ids[survey_above_horizon_mask][survey_bin_mask]
#                 survey_fid_max_counts = survey_field_maxvisits[survey_above_horizon_mask][survey_bin_mask]
#                 survey_f_visit_counts = np.array(([survey_field_visit_counter[fid] for fid in survey_fids_in_bin]))
#                 survey_num_fields_in_bin = len(survey_fids_in_bin)

#                 # Get min tiling fraction
#                 survey_min_tiling_fractions = survey_f_visit_counts / survey_fid_max_counts
#                 survey_min_tiling_frac = survey_min_tiling_fractions.min()
#                 calculated_features['survey_min_tiling'][global_idx, bid] = survey_min_tiling_frac

#                 # Get number of unvisited fields
#                 survey_n_unvisited = np.sum(survey_f_visit_counts == 0) / survey_num_fields_in_bin
#                 calculated_features['survey_num_unvisited_fields'][global_idx, bid] = survey_n_unvisited

#                 # Get number of incomplete fields
#                 survey_n_incomplete = np.sum(survey_fid_max_counts != survey_f_visit_counts) / survey_num_fields_in_bin
#                 calculated_features['survey_num_incomplete_fields'][global_idx, bid] = survey_n_incomplete

#                 if survey_min_tiling_frac > 1. or survey_min_tiling_frac < -.1:
#                     print(f'SURVEY MIN TILING IS {survey_min_tiling_frac}')
#                 if survey_n_unvisited > 1. or survey_n_unvisited < -.1:
#                     print(f'SURVEY N UNVISITED IS {survey_n_unvisited}')
#                 if survey_n_incomplete > 1. or survey_n_incomplete < -.1:
#                     print(f'SURVEY N INCOMPLETE IS {survey_n_incomplete}')
            
#             # ------------------------------------------------------------ #

#             # Next timestamp
#             global_idx += 1

#     return calculated_features

# def old_calculate_night_history_bin_features(pt_df, hpGrid, n_bins, do_night_num_visits, do_night_num_unvisited_fields, do_night_num_incomplete_fields, do_min_tiling):
#     calculated_features = {}
#     index = -1
#     for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
#         if not hpGrid.is_azel:
#             unique_field_ids, unique_field_counts = np.unique(
#                 group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32), 
#                 return_counts=True
#             )
#             # Dictionary carring max number of times a field can be visited
#             field2nvisits = {int(fid): int(c) for fid, c in zip(unique_field_ids, unique_field_counts)}
#             total_num_observations = len(group)
            
#             # Build bin2fields map
#             bins_arr = group['bin'].to_numpy(dtype=np.int32)
#             fields_arr = group['field_id'].to_numpy(dtype=np.int32)
#             bin2fields_in_bin = {}
#             for b, f in zip(bins_arr, fields_arr):
#                 if f != -1:
#                     bin2fields_in_bin.setdefault(int(b), set()).add(int(f))
            
#             field_visit_counter = defaultdict(int)
#             num_visits_tracking = np.zeros(n_bins)
#             num_unvisited_fields_tracking = np.zeros(n_bins)
#             num_incomplete_fields_tracking = np.zeros(n_bins)
#             min_tiling_tracking = np.zeros(n_bins)
            
#             for i_row in range(len(group)):
#                 index += 1
#                 field_id = int(fields_arr[i_row])
#                 bin_num = int(bins_arr[i_row])
                
#                 if field_id == -1:
#                     # Initialize once per night
#                     for bid, flist in bin2fields_in_bin.items():
#                         num_unvisited_fields_tracking[bid] = len(flist)
#                         num_incomplete_fields_tracking[bid] = len(flist)
#                 else:
#                     field_visit_counter[field_id] += 1
                    
#                     # Update bin visits tracker
#                     num_visits_tracking[bin_num] += 1
                    
#                     # Update minimum tiling number of this bin
#                     min = 0
#                     for fid in bin2fields_in_bin[bin_num]:
#                         if field_visit_counter[fid] < min:
#                             min = field_visit_counter[fid].copy()
#                     min_tiling_tracking[bin_num] = min
                    
#                     if field_visit_counter[field_id] == 1:
#                         num_unvisited_fields_tracking[bin_num] -= 1
                    
#                     if field_visit_counter[field_id] == field2nvisits[field_id]:
#                         num_incomplete_fields_tracking[bin_num] -= 1
                
#                 if do_night_num_visits:
#                     calculated_features['night_num_visits'][index] = num_visits_tracking / total_num_observations
#                 if do_night_num_unvisited_fields:
#                     calculated_features['night_num_unvisited_fields'][index] = num_unvisited_fields_tracking / total_num_observations
#                 if do_night_num_incomplete_fields:
#                     calculated_features['night_num_incomplete_fields'][index] = num_incomplete_fields_tracking / total_num_observations
#                 if do_min_tiling:
#                     calculated_features['night_min_tiling'][index] = min_tiling_tracking  / 12 # max tiling amongst all exposures with teff > .3 is 12
#         else: # if azel
#             raise NotImplementedError("Night history features not yet implemented for az/el grid")