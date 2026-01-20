import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import gymnasium as gym

import json
import fitsio
import pandas as pd
import time
import logging

from survey_ops.plotting import *
from survey_ops.utils import pytorch_utils
from survey_ops.src.agents import Agent
from survey_ops.src.algorithms import DDQN, BehaviorCloning
from survey_ops.utils.pytorch_utils import seed_everything
from survey_ops.utils.script_utils import setup_algorithm, setup_logger, get_device, load_raw_data_to_dataframe
from survey_ops.src.environments import OfflineEnv
from survey_ops.src.offline_dataset import OfflineDECamDataset

import argparse

def save_field_and_bin_schedules(eval_metrics, pd_group, save_dir, night_idx, make_gif=True, nside=None):
    # Save timestamps, field_ids, and bin numbers
    _timestamps = eval_metrics['ep-0']['timestamp'][f'night-{night_idx}'] \
                if len(eval_metrics['ep-0']['timestamp'][f'night-{night_idx}']) > len(pd_group['timestamp']) \
                else pd_group['timestamp'] 
    eval_field_schedule = {
        'time': _timestamps,
        'field_id': eval_metrics['ep-0']['field_id'][f'night-{night_idx}']
    }
    
    expert_field_schedule = {
        'time': _timestamps,
        'field_id': pd_group['field_id'].values
    }
    
    bin_schedule = {
        'time': _timestamps,
        'policy_bin_id': eval_metrics['ep-0']['bin'][f'night-{night_idx}'].astype(np.int32),
        'bin_id': pd_group['bin'].values
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data, filename in zip(
        [expert_field_schedule, eval_field_schedule, bin_schedule],
        ['expert_field_schedule.csv', 'new_field_schedule.csv', 'bin_schedule.csv']
        ):
        series_data = {key: pd.Series(value) for key, value in data.items()}
        _df = pd.DataFrame(series_data)
        if 'bin' in filename:
            _df['policy_bin_id'] = _df['policy_bin_id'].fillna(0).astype('Int64')
            _df['bin_id'] = _df['bin_id'].fillna(0).astype('Int64')
        output_filepath = save_dir + filename
        with open(output_filepath, 'w') as f:
            _df.to_csv(f, index=False)
        
        if make_gif:
            if 'bin' in filename:
                create_gif(
                    bin_schedule_filepath=output_filepath, 
                    id2pos_filepath=f'../data/nside{nside}_bin2radec.json',
                    outdir=save_dir,
                    nside=nside
                    )
            elif 'expert_field' in filename:
                continue
                raise NotImplementedError
            elif 'new_field' in filename:
                continue
                raise NotImplementedError

def create_gif(bin_schedule_filepath, id2pos_filepath, outdir, nside=None, plot_bins=True):
    schedule = pd.read_csv(bin_schedule_filepath)
    with open(id2pos_filepath) as f:
        id2pos = json.load(f)
    print(outdir)
    if plot_bins:
        plot_bins_movie(
            outfile=outdir + 'bin_schedule.gif',
            nside=nside,
            times=schedule["time"].values,
            idxs=schedule["bin_id"].values,
            alternate_idxs=schedule["policy_bin_id"].values,
            sky_bin_mapping=id2pos,
        )
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')

    # Test data selection
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')

    parser.add_argument('--trained_model_dir', type=str, default='../experiment_results/test_experiment/', help='Directory of the trained model to evaluate')
    parser.add_argument('--evaluation_name', type=str, default='evaluation_1', help='Name for this evaluation run')
    parser.add_argument('--specific_years', type=int, nargs='*', default=None, help='Specific years to include in the test dataset')
    parser.add_argument('--specific_months', type=int, nargs='*', default=None, help='Specific months to include in the test dataset')
    parser.add_argument('--specific_days', type=int, nargs='*', default=None, help='Specific days to include in the test dataset')

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')

    # Parse args
    args = parser.parse_args()
    args_dict = vars(args)

    # Set up output directories
    results_outdir = args.trained_model_dir + args.evaluation_name + '/'
    if not os.path.exists(results_outdir):
        os.makedirs(results_outdir)

    # Set up logging
    logger = setup_logger(save_dir=results_outdir, logging_filename='eval.log')
    logger.info("Saving results in " + results_outdir)

    # Print args
    logger.debug("Experiment parameters:")
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")

    # Seed everything
    pytorch_utils.seed_everything(args.seed)
    # torch.set_default_dtype(torch.float32)

    device = get_device()
    logger.info("Loading raw data...")
    raw_data_df = load_raw_data_to_dataframe(args.fits_path, args.json_path)
    logger.info("Processing raw data into OfflineDataset()...")
    
    with open(args.trained_model_dir + 'offline_dataset_config.pkl', 'rb') as f:
        OFFLINE_DATASET_CONFIG = pickle.load(f)
    nside = OFFLINE_DATASET_CONFIG['nside']


    logger.info("Loading test dataset with same config as training dataset...")
    test_dataset = OfflineDECamDataset(raw_data_df, specific_years=args.specific_years, specific_months=args.specific_months, specific_days=args.specific_days, **OFFLINE_DATASET_CONFIG) 
                                    #    args.binning_method, args.nside, args.bin_space, args.test_specific_years, args.test_specific_months, args.test_specific_days, \
                                        # args.no_bin_features, args.no_cyclical_norm, args.no_max_norm, args.no_inverse_airmass)
    

    with open(args.trained_model_dir + 'model_hyperparams.pkl', 'rb') as f:
        model_hyperparams = pickle.load(f)

    logger.info("Setting up agent...")
    algorithm = setup_algorithm(**model_hyperparams, device=device)
    agent = Agent(
        algorithm=algorithm,
        train_outdir=args.trained_model_dir,
    )
    agent.load(args.trained_model_dir + 'best_weights.pt')

    # Initialize environment
    logger.info("Setting up environment...")
    env_name = 'OfflineDECamEnv-v0'
    gym.register(
    id=f"gymnasium_env/{env_name}",
    entry_point=OfflineEnv,
    )

    env = gym.make(id=f"gymnasium_env/{env_name}", test_dataset=test_dataset, max_nights=None)

    logger.info("Starting evaluation...")
    agent.evaluate(env=env, num_episodes=args.num_episodes, field_choice_method='random', eval_outdir=results_outdir)
    logger.info("Evaluation complete.")

    with open(results_outdir + 'eval_metrics.pkl', 'rb') as handle:
        eval_metrics = pickle.load(handle)

    logger.info("Generating evaluation plots...")

    ep_num = 0
    # Plot field_id and bin vs step for first episode
    for night_idx, (night_name, night_group) in enumerate(test_dataset._df.groupby('night')):
        # Get date in string form for plots
        date = night_name.date()
        date_str = f"{date.year}-{date.month}-{date.day}"
        subdir_path = results_outdir + date_str + '/'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        # Plot bins vs step
        fig_b, axs_b = plt.subplots(2, figsize=(10,7), sharex=True)
        axs_b[0].plot(eval_metrics[f'ep-{ep_num}']['bin'][f'night-{night_idx}'], marker='o', label='pred', alpha=.5)
        axs_b[0].plot(night_group['bin'].values.astype(int), marker='o', label='true', alpha=.5)
        axs_b[0].legend()
        axs_b[0].set_ylabel('bin')
        axs_b[1].plot(eval_metrics[f'ep-{ep_num}']['bin'][f'night-{night_idx}'][:len(night_group['bin'].values.astype(int))] \
                    - night_group['bin'].values.astype(int)[:len(eval_metrics[f'ep-{ep_num}']['bin'][f'night-{night_idx}'])])
        axs_b[1].set_ylabel('bin' + '\n (residuals)')
        fig_b.suptitle(date_str)
        fig_b.savefig(subdir_path + f'bin_vs_step.png')
        plt.close()
        
        # Plot fields vs step
        fig_f, axs_f = plt.subplots(2, figsize=(10,7), sharex=True)
        axs_f[0].plot(eval_metrics[f'ep-{ep_num}']['field_id'][f'night-{night_idx}'], marker='o', label='pred', alpha=.5)
        axs_f[0].plot(night_group['field_id'].values.astype(int), marker='o', label='true', alpha=.5)
        axs_f[0].legend()
        axs_f[0].set_ylabel('field_id')
        axs_f[1].plot(eval_metrics[f'ep-{ep_num}']['field_id'][f'night-{night_idx}'][:len(night_group['field_id'].values.astype(int))] \
                    - night_group['field_id'].values.astype(int)[:len(eval_metrics[f'ep-{ep_num}']['field_id'][f'night-{night_idx}'])])
        axs_f[1].set_ylabel('field_id' + '\n (residuals)')
            
        fig_f.suptitle(date_str)
        fig_f.savefig(subdir_path  + f'field_id_vs_step.png')
        plt.close()

        # Plot state features vs timestamp for first episode
        fig, axs = plt.subplots(len(test_dataset.state_feature_names), figsize=(10, len(test_dataset.state_feature_names)*5))
        for i, feature_row in enumerate(eval_metrics['ep-0']['observations'][f'night-{night_idx}'].T):
            feat_name = env.unwrapped.test_dataset.state_feature_names[i]
            eval_timestamps = eval_metrics['ep-0']['timestamp'][f'night-{night_idx}']
            eval_data = feature_row.copy()
            if feat_name == 'airmass':
                eval_data = 1 / feature_row
            elif 'dec' in feat_name or 'el' in feat_name:
                eval_data = feature_row * (np.pi/2)
            else:
                eval_data = feature_row
            axs[i].plot(eval_timestamps, eval_data, label='policy roll out')
            axs[i].plot(night_group['timestamp'].values, night_group[feat_name].values, label='original schedule')
            axs[i].set_title(feat_name)
            axs[i].legend()
        fig.savefig(subdir_path + f'state_features_vs_timestep.png')
        plt.close()

        # Plot static bin and field radec scatter plots
        eval_bin_radecs = np.array([env.unwrapped.test_dataset.bin2coord[bin_num] for bin_num in eval_metrics['ep-0']['bin'][f'night-{night_idx}'].astype(int) if bin_num != -1])
        orig_bin_radecs = np.array([env.unwrapped.test_dataset.bin2coord[bin_num] for bin_num in night_group['bin'].values if bin_num != -1])
        
        eval_field_radecs = np.array([env.unwrapped.test_dataset.field2radec[field_id] for field_id in eval_metrics['ep-0']['field_id'][f'night-{night_idx}'].astype(int) if field_id != -1])
        orig_field_radecs = np.array([env.unwrapped.test_dataset.field2radec[field_id] for field_id in night_group['field_id'].values.astype(int) if field_id != -1])
        if len(orig_field_radecs) != 1:
            
            # Plot bins
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_bin_radecs[:, 0], orig_bin_radecs[:, 1], label='orig schedule', cmap='Reds', c=np.arange(len(orig_bin_radecs)))
            axs[1].scatter(eval_bin_radecs[:, 0], eval_bin_radecs[:, 1], label='policy roll out', cmap='Blues', c=np.arange(len(eval_bin_radecs)))
            for ax in axs:
                ax.set_xlabel('x (ra or az)')
                ax.legend()
            axs[0].set_ylabel('y (dec or el)')
            fig.suptitle(f'Bins {night_name}')
            fig.savefig(subdir_path + f'bins_ra_vs_dec.png')
            plt.close()
            
            # Plot fields
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_field_radecs[:, 0], orig_field_radecs[:, 1], label='orig schedule', cmap='Reds', c=np.arange(len(orig_field_radecs)), s=10)
            axs[1].scatter(eval_field_radecs[:, 0], eval_field_radecs[:, 1], label='policy roll out', cmap='Blues', c=np.arange(len(eval_field_radecs)), s=10)
            for ax in axs:
                ax.set_xlabel('ra')
                ax.legend()
            axs[0].set_ylabel('dec')
            fig.suptitle(f'Fields {night_name}')
            fig.savefig(subdir_path + f'fields_ra_vs_dec.png')
            plt.close()

        logger.info(f'Creating schedule gif for {night_idx}th night')
        save_field_and_bin_schedules(eval_metrics=eval_metrics, pd_group=night_group, save_dir=subdir_path, night_idx=night_idx, nside=nside, make_gif=True)

if __name__ == "__main__":
    main()