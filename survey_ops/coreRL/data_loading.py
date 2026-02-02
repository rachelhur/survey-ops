import fitsio
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def load_raw_data_to_dataframe(fits_path, json_path):
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
    return df