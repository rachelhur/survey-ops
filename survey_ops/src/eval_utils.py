import numpy as np
from survey_ops.utils import ephemerides

def get_fields_in_azel_bin(bin_num, timestamp, field2nvisits, field_ids, field_radecs, hpGrid, visited, bin2fields_in_bin=None):
    mask_completed_fields = np.array([visited.count(fid) < field2nvisits[fid] for fid in field_ids])
    fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=field_radecs[:, 0], dec=field_radecs[:, 1], time=timestamp)
    field_bins = hpGrid.ang2idx(lon=fields_az, lat=fields_el)

    # valid fields are fields in bin and fields which have not been completed
    mask_below_horizon = field_bins != None
    mask_invalid_fields = (field_bins == bin_num) & mask_completed_fields & mask_below_horizon
    fields_in_bin = field_ids[mask_invalid_fields]
    # return field_bins, incomplete_fields_mask
    return fields_in_bin

def get_fields_in_radec_bin(bin_num, bin2fields_in_bin, timestamp=None, field2nvisits=None, field_ids=None, field_radecs=None, hpGrid=None, visited=None):
    return bin2fields_in_bin.get(bin_num)