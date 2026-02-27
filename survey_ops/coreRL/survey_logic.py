import numpy as np
from survey_ops.utils import ephemerides, units
import logging
import pandas as pd
import torch

logger = logging.getLogger(__name__)

def get_fields_in_bin(bin_num, is_azel, timestamp, field2nvisits, field_ids, field_radecs, hpGrid, visited, bin2fields_in_bin=None):
    if is_azel:
        mask_completed_fields = np.array([visited[fid] < field2nvisits[fid] for fid in field_ids], dtype=bool)
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=field_radecs[:, 0], dec=field_radecs[:, 1], time=timestamp)
        mask_fields_below_horizon = fields_el > 0
        field_bins = hpGrid.ang2idx(lon=fields_az, lat=fields_el)
        # valid fields are fields in bin and fields which have not been completed
        sel_valid_fields = (field_bins == bin_num) & mask_fields_below_horizon & mask_completed_fields
        fields_in_bin = field_ids[sel_valid_fields]
    else:
        bin_num = str(bin_num)
        fields_in_bin = bin2fields_in_bin.get(bin_num)
        sel_valid_fields = np.array([visited[fid] < field2nvisits[fid] for fid in fields_in_bin], dtype=bool)
        fields_in_bin = np.array(fields_in_bin)[sel_valid_fields]
    return fields_in_bin

