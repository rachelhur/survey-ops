import numpy as np
from survey_ops.utils import ephemerides, units
import logging
import pandas as pd
import torch

logger = logging.getLogger(__name__)

def get_fields_in_bin(bin_num, is_azel, timestamp, field2nvisits, field_ids, field_radecs, hpGrid, visited, bin2fields_in_bin=None):
    if is_azel:
        mask_completed_fields = np.array([visited.count(fid) < field2nvisits[fid] for fid in field_ids], dtype=bool)
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=field_radecs[:, 0], dec=field_radecs[:, 1], time=timestamp)
        mask_fields_below_horizon = fields_el > 0
        field_bins = hpGrid.ang2idx(lon=fields_az, lat=fields_el)
        # valid fields are fields in bin and fields which have not been completed
        sel_valid_fields = (field_bins == bin_num) & mask_fields_below_horizon & mask_completed_fields
        fields_in_bin = field_ids[sel_valid_fields]
    else:
        bin_num = str(bin_num)
        fields_in_bin = bin2fields_in_bin.get(bin_num)
        sel_valid_fields = np.array([visited.count(fid) < field2nvisits[fid] for fid in fields_in_bin], dtype=bool)
        fields_in_bin = np.array(fields_in_bin)[sel_valid_fields]
    return fields_in_bin

def remove_specific_objects(objects_to_remove, df):
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
    
    for night, group in df.groupby('night'):
        bin_history = set()
        for i_row, row in group.iterrows():
            if row['object'] == 'zenith':
                bins_visited.append(0)
            else:
                bin_history.add(row['bin'])
                bins_visited.append(len(bin_history))
    
    df['bins_visited_in_night'] = bins_visited
    return df

def do_noncyclic_normalizations(state, state_feature_names,
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