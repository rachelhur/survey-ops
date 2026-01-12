
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
        'loss_fxn': loss_fxn
    }

    if algorithm_name == 'ddqn' or algorithm_name == 'dqn':
        assert gamma is not None, "Gamma (discount factor) must be specified for DDQN."
        assert tau is not None, "Tau (target network update rate) must be specified for DDQN."
        assert loss_fxn in ['mse', 'huber'], "DDQN only supports mse or huber loss functions."

        if loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        elif loss_fxn == 'huber':
            loss_fxn = nn.HuberLoss()

        model_hyperparams .update( {
            'gamma': gamma,
            'tau': tau,
            'use_double': algorithm_name == 'ddqn',
        } )

        algorithm = DDQN(
            device=device,
            **model_hyperparams
        )

    elif algorithm_name == 'behavior_cloning':
        assert loss_fxn in ['cross_entropy', 'mse'], "Behavior Cloning only supports cross_entropy or mse loss functions."
        if loss_fxn == 'cross_entropy':
            loss_fxn = nn.CrossEntropyLoss(reduction='mean')
        elif loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')

        algorithm = BehaviorCloning(
            device=device,
            **model_hyperparams
        )

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

def get_offline_dataset(df, binning_method, nside, bin_space, specific_years, specific_months, specific_days, no_bin_features, no_cyclical_norm, no_max_norm, no_inverse_airmass, include_default_features=True):
    dataset =  OfflineDECamDataset(
        df, 
        binning_method=binning_method,
        nside=nside,
        bin_space=bin_space,
        specific_years=specific_years,
        specific_months=specific_months,
        specific_days=specific_days,
        include_default_features=include_default_features,
        include_bin_features=not no_bin_features,
        do_cyclical_norm=not no_cyclical_norm,
        do_max_norm=not no_max_norm,
        do_inverse_airmass=not no_inverse_airmass
    )
    return dataset