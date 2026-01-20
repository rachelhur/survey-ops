
import os
import pickle
import sys
import logging

import numpy as np
import torch
import pandas as pd
import fitsio

from survey_ops.src.offline_dataset import OfflineDECamDataset
from survey_ops.src.algorithms import DDQN, BehaviorCloning
import torch.nn as nn

def setup_algorithm(save_dir=None, algorithm_name=None, obs_dim=None, num_actions=None, loss_fxn=None, hidden_dim=None, lr=None, lr_scheduler=None, device=None, lr_scheduler_kwargs=None, gamma=None, tau=None):
    model_hyperparams = {
        'obs_dim': obs_dim,
        'num_actions': num_actions, 
        'hidden_dim': hidden_dim,
        'lr': lr,
        'lr_scheduler': lr_scheduler,
        'lr_scheduler_kwargs': lr_scheduler_kwargs,
    }

    if algorithm_name == 'ddqn' or algorithm_name == 'dqn':
        assert gamma is not None, "Gamma (discount factor) must be specified for DDQN."
        assert tau is not None, "Tau (target network update rate) must be specified for DDQN."
        # assert loss_fxn in ['mse', 'huber'], "DDQN only supports mse or huber loss functions."

        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn
        elif loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        elif loss_fxn == 'huber':
            loss_fxn = nn.HuberLoss()
        else:
            raise NotImplementedError

        model_hyperparams .update( {
            'gamma': gamma,
            'tau': tau,
            'use_double': algorithm_name == 'ddqn',
            'loss_fxn': loss_fxn
        } )

        algorithm = DDQN(
            device=device,
            **model_hyperparams
        )

    elif algorithm_name == 'behavior_cloning':
        # assert loss_fxn in ['cross_entropy', 'mse'], "Behavior Cloning only supports cross_entropy or mse loss functions."
        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn
        elif loss_fxn == 'cross_entropy':
            loss_fxn = nn.CrossEntropyLoss(reduction='mean')
        elif loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        else:
            print(loss_fxn)
            raise NotImplementedError

        model_hyperparams.update({
        'loss_fxn': loss_fxn
        })
        algorithm = BehaviorCloning(
            device=device,
            **model_hyperparams
        )
    else:
        raise NotImplementedError

    if save_dir is not None:
        model_hyperparams['algorithm_name'] = algorithm_name
        with open(save_dir + 'model_hyperparams.pkl', 'wb') as f:
            pickle.dump(model_hyperparams, f)
    return algorithm


def setup_logger(save_dir, logging_filename):
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(save_dir + logging_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add to handlers
    # console_format = logging.Formatter('%(levelname)s - %(message)s')
    format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(format)
    file_handler.setFormatter(format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"   
    )
    return device

def load_raw_data_to_dataframe(fits_path, json_path):
    try:
        # --- Load json df ---- #
        df = pd.read_json(json_path)
        print('Loaded data from json')
    except:
        # --- Load fits ---- #
        print(json_path, 'DNE. Loading and processing data from fits.')
        d = fitsio.read(fits_path)
        sel = (d['propid'] == '2012B-0001') & (d['exptime'] > 40) & (d['exptime'] < 100) & (~np.isnan(d['teff']))
        selected_d = d[sel]
        column_names = selected_d.dtype.names
        df = pd.DataFrame(selected_d, columns=column_names)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

def save_field_and_bin_schedules(eval_metrics, pd_group, outdir, date_str):
    # Save timestamps, field_ids, and bin numbers
    _timestamps = eval_metrics['ep-0']['timestamp'] \
                if len(eval_metrics['ep-0']['timestamp']) > len(pd_group['timestamp']) \
                else pd_group['timestamp'] 
    eval_field_schedule = {
        'time': _timestamps,
        'field_id': eval_metrics['ep-0']['field_id']
    }
    
    expert_field_schedule = {
        'time': _timestamps,
        'field_id': pd_group['field_id'].values
    }
    
    bin_schedule = {
        'time': _timestamps,
        'policy_bin_id': eval_metrics['ep-0']['bin'].astype(np.int32),
        'bin_id': pd_group['bin'].values
    }
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for data, filename in zip(
        [expert_field_schedule, eval_field_schedule, bin_schedule],
        ['expert_field_schedule.csv', 'new_field_schedule.csv', 'bin_schedule.csv']
        ):
        series_data = {key: pd.Series(value) for key, value in data.items()}
        _df = pd.DataFrame(series_data)
        if 'bin' in filename:
            _df['policy_bin_id'] = _df['policy_bin_id'].fillna(0).astype('Int64')
            _df['bin_id'] = _df['bin_id'].fillna(0).astype('Int64')
        output_filepath = outdir + f'_{date_str}' + filename
        with open(output_filepath, 'w') as f:
            _df.to_csv(f, index=False)