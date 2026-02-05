import numpy as np
from survey_ops.utils import ephemerides
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