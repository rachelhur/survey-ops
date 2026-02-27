import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import gymnasium as gym

import json
import pandas as pd
import logging

from survey_ops.plotting import plot_schedule_from_file
from survey_ops.coreRL.agents import Agent
from survey_ops.utils.sys_utils import seed_everything
from survey_ops.algorithms import setup_algorithm
from survey_ops.utils.sys_utils import setup_logger, get_device
from survey_ops.coreRL.data_processing import load_raw_data_to_dataframe, get_nautical_twilight
from survey_ops.coreRL.environments import OfflineDECamTestingEnv
from survey_ops.coreRL.offline_dataset import OfflineDELVEDataset
from survey_ops.utils.config import save_config, load_global_config, dict_to_nested
import logging
logger = logging.getLogger(__name__)

import argparse


from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

def save_schedule(eval_metrics, pd_group, save_dir, night_idx, make_gifs=True, nside=None, is_azel=False, whole=False, bin2pos_filepath=None, field2radec_filepath=None):
    # Save timestamps, field_ids, and bin numbers
    bin_space = 'azel' if is_azel else 'radec'
    assert os.path.exists(save_dir)

    eval_timestamps = eval_metrics['ep-0']['timestamp'][f'night-{night_idx}']
    expert_timestamps = pd_group['timestamp'].values
    
    _timestamps = eval_timestamps if len(eval_timestamps) > len(expert_timestamps) else expert_timestamps
    
    schedule_full = {
        'agent_timestamp': eval_timestamps,
        'agent_field_id': eval_metrics['ep-0']['field_id'][f'night-{night_idx}'],
        'agent_bin_id': eval_metrics['ep-0']['bin'][f'night-{night_idx}'],
        'expert_timestamp': expert_timestamps,
        'expert_field_id': pd_group['field_id'].values,
        'expert_bin_id': pd_group['bin'].values,
        'timestamp': _timestamps
    }

    df = pd.DataFrame(data={k: pd.Series(v) for k, v in schedule_full.items()}).fillna(0).astype(int)

    output_filepath = save_dir + 'schedule.csv'
    df.to_csv(output_filepath, index=False)

    # schedule = pd.read_csv(output_filepath)
    logger.info("Creating fieldbin movies")
    # Create binfield movies
    plot_schedule_from_file(
        outfile=save_dir + 'agent_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=field2radec_filepath,
        bins_file=bin2pos_filepath,
        whole=False,
        compare=False,
        expert=False,
        is_azel=bin_space=='azel',
        mollweide=False,
    )
    plot_schedule_from_file(
        outfile=save_dir + 'expert_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=field2radec_filepath,
        bins_file=bin2pos_filepath,
        whole=False,
        compare=False,
        expert=True,
        is_azel=bin_space=='azel',
        mollweide=False,
    )

    if make_gifs:
        # Create fields movies
        logger.info("Creating field movies")
        if not is_azel:
            plot_schedule_from_file(
                outfile=save_dir + 'expert_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=False,
                compare=False,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=False,
            )
            plot_schedule_from_file(
                outfile=save_dir + 'agent_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=False,
                compare=False,
                expert=False,
                is_azel=bin_space=='azel',
                mollweide=False,
            )

        plot_schedule_from_file(
            outfile=save_dir + 'agent_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=bin_space=='azel',
            mollweide=False,
        ) 

        # Create bin movies   
        logger.info("Creating bin movies")
        plot_schedule_from_file(
            outfile=save_dir + 'bin_comparison_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=True,
            expert=True,
            is_azel=bin_space=='azel',
            mollweide=False,
        )
        plot_schedule_from_file(
            outfile=save_dir + 'expert_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=False,
            expert=True,
            is_azel=bin_space=='azel',
            mollweide=False,
        )

        if bin_space == 'radec':
            # Mollefield
            logger.info("Creating static plots")
            plot_schedule_from_file(
                outfile=save_dir + 'mollweide.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=True,
            )  
            plot_schedule_from_file(
                outfile=save_dir + 'ortho.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=False,
            )  

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    # Test data selection
    # parser.add_argument('--SISPI_file', type=str, help='Path to a SISPI-like json file with a list of fields.')
    parser.add_argument('-t', '--trained_model_dir', type=str, default='../experiment_results/test_experiment/', help='Directory of the trained model to evaluate')
    parser.add_argument('-n', '--evaluation_name', type=str, default='evaluation_1', help='Name for this evaluation run')
    parser.add_argument('-y', '--specific_years', type=int, nargs='*', default=None, help='Specific years to include in the test dataset')
    parser.add_argument('-m','--specific_months', type=int, nargs='*', default=None, help='Specific months to include in the test dataset')
    parser.add_argument('-d', '--specific_days', type=int, nargs='*', default=None, help='Specific days to include in the test dataset')
    parser.add_argument('--specific_filters', type=str, nargs='*', default=None, help='Specific days to include in the test dataset')
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')
    parser.add_argument('--make_gifs', action='store_true', help="Whether to create the set of gifs. Currently can only choose to make all or none.")

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')

    # Parse args
    args = parser.parse_args()
    args_dict = vars(args)

    # Get configs
    global_cfg = load_global_config(PROJECT_ROOT / 'configs' / 'global_config.json')
    config_path = Path(args.trained_model_dir) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    results_outdir = cfg['metadata']['outdir'] + '/' + args.evaluation_name + '/'
    if not os.path.exists(results_outdir):
        os.makedirs(results_outdir)

    # Set up logging
    logger = setup_logger(save_dir=Path(results_outdir).resolve(), logging_filename='eval.log', logging_level=args.logging_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)

    logger.info("Saving results in " + results_outdir)

    # Print args
    logger.warning("Experiment parameters:")
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")

    # Seed everything
    seed_everything(args.seed)

    device = get_device()
    logger.info("Loading raw data...")
    df = load_raw_data_to_dataframe(Path(global_cfg['paths']['FITS_DIR']) / Path(global_cfg['files']['DECFITS']))
    
    nside = cfg['data']['nside']

    logger.info("Loading test dataset with same config as training dataset...")
    test_dataset = OfflineDELVEDataset(
        df=df,
        cfg=cfg,
        gcfg=global_cfg,
        specific_years=args.specific_years,
        specific_months=args.specific_months,
        specific_days=args.specific_days,
        specific_filters=args.specific_filters
        ) 
        
    # Plot State x action space via cornerplot
    corner_plot = sns.pairplot(test_dataset._df,
             vars=test_dataset.global_feature_names + ['bin'],
             kind='hist',
             corner=True
            )
    corner_plot.figure.savefig(results_outdir + 'state_times_action_space_corner_plot.png')

    logger.info("Setting up agent...")
    algorithm = setup_algorithm(algorithm_name=cfg['model']['algorithm'], 
                                num_actions=cfg['data']['num_actions'],
                                n_global_features = test_dataset.states.shape[-1],
                                n_bin_features=0 if test_dataset.bin_states is None else test_dataset.bin_states.shape[-1],
                                grid_network=cfg['model']['grid_network'],
                                loss_fxn=cfg['model']['loss_function'],
                                hidden_dim=cfg['train']['hidden_dim'], lr=cfg['train']['lr'], lr_scheduler=cfg['train']['lr_scheduler'], 
                                device=device, lr_scheduler_kwargs=cfg['train']['lr_scheduler_kwargs'], lr_scheduler_epoch_start=cfg['train']['lr_scheduler_epoch_start'], 
                                lr_scheduler_num_epochs=cfg['train']['lr_scheduler_num_epochs'],
                                gamma=cfg['model']['gamma'], 
                                tau=cfg['model']['tau'],
                                activation=cfg['model']['activation']
                                )
    
    agent = Agent(
        algorithm=algorithm,
        train_outdir=args.trained_model_dir,
    )
    agent.load(args.trained_model_dir + 'best_weights.pt')

    # Initialize environment
    logger.info("Setting up environment...")
    env_name = 'OfflineDECamTestingEnv-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=OfflineDECamTestingEnv,
    )

    # Creat env
    global_pd_nightgroup = test_dataset._df.groupby('night')
    if len(cfg['data']['additional_bin_features']) > 0:
        bin_pd_nightgroup = test_dataset._bin_df.groupby('night')
    else:
        bin_pd_nightgroup = None
    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, gcfg=global_cfg, max_nights=None, global_pd_nightgroup=global_pd_nightgroup, bin_pd_nightgroup=bin_pd_nightgroup)
    
    # Plot predicted action for each state
    with torch.no_grad():
        q_vals = agent.algorithm.policy_net(test_dataset.states.to(device), test_dataset.bin_states.to(device) if test_dataset.bin_states is not None else None)
        eval_actions = torch.argmax(q_vals, dim=1).to('cpu').detach().numpy()
    
    # Sequence of actions from target (original schedule) and policy
    target_sequence = test_dataset.actions.detach().numpy()
    eval_sequence = eval_actions
    time_idx = np.where(np.array(test_dataset.state_feature_names) == 'time_fraction_since_start')[0]
    first_night_indices = np.where(test_dataset.states[:, time_idx] == 0)[0]

    fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
    
    axs[0].plot(target_sequence, marker='*', alpha=.3, label='true')
    axs[0].plot(eval_sequence, marker='o', alpha=.3, label='pred')
    axs[0].legend()
    axs[0].set_ylabel('bin number')
    axs[0].vlines(first_night_indices, ymin=0, ymax=len(test_dataset.hpGrid.lon), color='black', linestyle='--')
    axs[1].plot(eval_sequence - target_sequence, marker='o', alpha=.5)
    axs[1].set_ylabel('Eval sequence - target sequence \n[bin number]')
    axs[1].set_xlabel('observation index')
    fig.savefig(results_outdir + 'eval_and_target_bin_sequences.png')

    # Roll out policy
    logger.info("Starting evaluation...")
    agent.evaluate(env=env, cfg=cfg, num_episodes=args.num_episodes, field_choice_method='random', eval_outdir=results_outdir)
    logger.info("Evaluation complete.")

    with open(results_outdir + 'eval_metrics.pkl', 'rb') as handle:
        eval_metrics = pickle.load(handle)

    logger.info("Generating evaluation plots...")

    bin2pos_filepath = global_cfg['paths']['LOOKUP_DIR'] + f"nside{nside}_bin2{cfg['data']['bin_space']}.json"
    field2radec_filepath = global_cfg['paths']['LOOKUP_DIR'] + global_cfg['files']['FIELD2RADEC']
    with open(field2radec_filepath, 'r') as f:
        FIELD2RADEC = json.load(f)

    ep_num = 0
    
    # Plot field_id and bin vs step for first episode
    for night_idx, (night_name, night_group) in enumerate(test_dataset._df.groupby('night')):
        # Get date in string form for plots
        date = night_name.date()
        date_str = f"{date.year}-{date.month}-{date.day}"
        logger.info(f'Drawing plots for night {date_str}')
        subdir_path = results_outdir + date_str + '/'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        # Mask zenith observations in plotting
        eval_zenith_mask = eval_metrics['ep-0']['field_id'][f'night-{night_idx}'] != -1
        data_zenith_mask = night_group['field_id'] != -1
        
        eval_timestamps = eval_metrics['ep-0']['timestamp'][f'night-{night_idx}']
        sunset = get_nautical_twilight(night_group['timestamp'].values[0], event_type='set')
        eval_timestamps = (eval_timestamps - sunset) / 3600
        data_timestamps = (night_group['timestamp'].values - sunset) / 3600 
    
        # Plot bins vs timestamp        
        fig_b, axb = plt.subplots()
        axb.plot(eval_timestamps[eval_zenith_mask],
                      eval_metrics[f'ep-{ep_num}']['bin'][f'night-{night_idx}'][eval_zenith_mask],
                      marker='o', label='pred', alpha=.5)
        axb.plot(data_timestamps[data_zenith_mask],
                      night_group['bin'].values.astype(int)[data_zenith_mask],
                      marker='o', label='true', alpha=.5)
        axb.legend()
        axb.set_xlabel('Hours since sunset \n (-10 deg)')
        axb.set_ylabel('bin')
        fig_b.suptitle(date_str)
        fig_b.tight_layout()
        fig_b.savefig(subdir_path + f'bin_vs_step.png')
        plt.close()

        # Plot state features vs timestamp for first episode
        fig, axs = plt.subplots(len(test_dataset.global_feature_names), figsize=(10, len(test_dataset.global_feature_names)*5))
        for i, feature_row in enumerate(eval_metrics['ep-0']['glob_observations'][f'night-{night_idx}'].T[:len(test_dataset.global_feature_names)]):
            feat_name = test_dataset.global_feature_names[i]
            eval_data = feature_row.copy()
            if feat_name == 'airmass':
                eval_data = 1 / feature_row
            elif 'dec' in feat_name or 'el' in feat_name:
                eval_data = feature_row * (np.pi/2)
            else:
                eval_data = feature_row

            axs[i].plot(eval_timestamps[eval_zenith_mask], eval_data[eval_zenith_mask], label='policy roll out', marker='o')
            axs[i].plot(data_timestamps[data_zenith_mask], night_group[feat_name].values[data_zenith_mask], label='original schedule', marker='o')
            axs[i].set_title(feat_name)
            axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
            axs[i].legend()
        fig.tight_layout()
        fig.savefig(subdir_path + f'state_features_vs_timestamp.png')
        plt.close()

        # Plot static bin and field radec scatter plots
        bin2coord = {int(i): (lon, lat) for i, (lon, lat) in enumerate(zip(test_dataset.hpGrid.lon, test_dataset.hpGrid.lat))}

        eval_bin_radecs = np.array([bin2coord[bin_num] for bin_num in eval_metrics['ep-0']['bin'][f'night-{night_idx}'].astype(int) if bin_num != -1])
        orig_bin_radecs = np.array([bin2coord[bin_num] for bin_num in night_group['bin'].values if bin_num != -1])
        
        eval_field_radecs = np.array([FIELD2RADEC[str(field_id)] for field_id in eval_metrics['ep-0']['field_id'][f'night-{night_idx}'].astype(int) if field_id != -1])
        orig_field_radecs = np.array([FIELD2RADEC[str(field_id)] for field_id in night_group['field_id'].values.astype(int) if field_id != -1])
        
        if len(orig_field_radecs) != 1:
            # Plot bins
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_bin_radecs[:, 0], orig_bin_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_bin_radecs)))
            axs[1].scatter(eval_bin_radecs[:, 0], eval_bin_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(eval_bin_radecs)))
            for ax in axs:
                ax.set_xlabel('x (ra or az)')
                ax.legend()
            axs[0].set_ylabel('y (dec or el)')
            fig.suptitle(f'Bins {night_name}')
            fig.savefig(subdir_path + f'bins_ra_vs_dec.png')
            plt.close()
            
            # Plot fields
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_field_radecs[:, 0], orig_field_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_field_radecs)), s=10)
            axs[1].scatter(eval_field_radecs[:, 0], eval_field_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(eval_field_radecs)), s=10)
            for ax in axs:
                ax.set_xlabel('ra')
                ax.legend() 
            axs[0].set_ylabel('dec')
            fig.suptitle(f'Fields {night_name}')
            fig.savefig(subdir_path + f'fields_ra_vs_dec.png')
            plt.close()

        logger.info(f'Creating schedule gif for {night_idx}th night')
        save_schedule(eval_metrics=eval_metrics, pd_group=night_group, save_dir=subdir_path, night_idx=night_idx, nside=nside, make_gifs=args.make_gifs, 
                      is_azel=test_dataset.hpGrid.is_azel, bin2pos_filepath=bin2pos_filepath, field2radec_filepath=field2radec_filepath)
        
if __name__ == "__main__":
    main()