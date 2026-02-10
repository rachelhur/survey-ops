import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import time
import pickle

from survey_ops.coreRL.agents import Agent
from survey_ops.algorithms.factory import setup_algorithm
from survey_ops.utils import geometry
from survey_ops.utils import units
from survey_ops.utils.sys_utils import setup_logger, get_device, seed_everything
from survey_ops.coreRL.data_processing import load_raw_data_to_dataframe 
from survey_ops.coreRL.offline_dataset import OfflineDECamDataset
from survey_ops.utils.config import save_config, load_global_config, dict_to_nested

import argparse
import logging
import json

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] 

"""
Features implemented so far:
POINTING FEATURES:
    "ra",
    "dec",
    "az",
    "el",
    "airmass",
    "ha",
    "sun_ra",
    "sun_dec",
    "sun_az",
    "sun_el",
    "moon_ra",
    "moon_dec",
    "moon_az",
    "moon_el",
    "time_fraction_since_start"
    "bins_visited_in_night": Number of unique bins visited so far in the night


BIN FEATURES:
    "ha",
    "airmass",
    "moon_distance",
    "night_num_visits",
    "night_num_unvisited_fields",
    "night_num_incomplete_fields"
"""

def plot_metrics(results_outdir, dataset):
    with open(results_outdir / 'train_metrics.pkl', 'rb') as f:
        train_metrics = pickle.load(f)
    with open(results_outdir / 'val_metrics.pkl', 'rb') as f:
        val_metrics = pickle.load(f)
    with open(results_outdir / 'val_train_metrics.pkl', 'rb') as f:
        val_train_metrics = pickle.load(f)
    # val_steps = np.linspace(0, len(train_metrics['train_loss']), len(val_metrics['accuracy']))

    # Plot train and val loss
    fig, axs = plt.subplots(4, sharex=True, figsize=(5, 12))

    axs[0].plot(train_metrics['epoch'], train_metrics['train_loss'], label='train loss', color='grey', alpha=.5, linestyle='dotted')
    axs[0].plot(val_metrics['epoch'], val_metrics['val_loss'], label='val loss')
    axs[0].hlines(y=0, xmin=0, xmax=np.max(val_metrics['epoch']), color='red', linestyle='dashed')
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[0].legend()

    axs[1].plot(val_train_metrics['epoch'], val_train_metrics['accuracy'], label='train accuracy', color='grey', alpha=.5, linestyle='dotted')
    axs[1].plot(val_metrics['epoch'], val_metrics['accuracy'], label='val accuracy')
    axs[1].hlines(y=1, xmin=0, xmax=np.max(train_metrics['epoch']), color='red', linestyle='dotted')
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].legend()

    axs[2].scatter(train_metrics['epoch'], train_metrics['lr'], marker='o', s=10)
    axs[2].set_ylabel('LR', fontsize=14)

    i = 0
    for key in val_metrics.keys():
        if key != 'accuracy' and key != 'epoch' and 'loss' not in key:
            axs[3].plot(val_metrics['epoch'], val_metrics[key], label='val ' + key, color=f"C{i}")
            axs[3].plot(val_metrics['epoch'], val_train_metrics[key], color=f"C{i}", linestyle='dotted', alpha=.5)
            i += 1
    axs[3].hlines(0, xmin=0, xmax=np.max(val_metrics['epoch']), linestyle='--', color='red')
    axs[3].legend()
    axs[3].set_xlabel('Epoch', fontsize=14)

    fig.tight_layout()
    fig.savefig(results_outdir / 'figures' / 'loss_and_metrics_history.png')

    if 'ang_sep' in val_metrics:
        lonlat = np.array((dataset.hpGrid.lon, dataset.hpGrid.lat))
        pos1 = lonlat[:, :-1]
        pos2 = lonlat[:, 1:]
        ang_seps = geometry.angular_separation(pos1=pos1, pos2=pos2)
        average_bin_sep = np.mean(ang_seps)
        fig, ax = plt.subplots()
        ax.plot(val_train_metrics['epoch'], val_train_metrics['ang_sep'], label='train', color='grey', alpha=.5, linestyle='dotted')
        ax.plot(val_metrics['epoch'], val_metrics['ang_sep'], label='val')
        ax.set_ylabel('Angular separation (rad)', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.hlines(y=average_bin_sep, xmin=0, xmax=np.max(val_train_metrics['epoch']), label='average bin center separation', color='black', linestyle='dotted')
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(results_outdir / 'figures' / 'angular_separation_history.png')

    if 'unique_bins' in val_metrics:
        # Count bins with < 10 examples
        bin_ids, _ = np.unique(dataset.actions.detach().numpy(), return_counts=True)
        total_bin_diversity = len(bin_ids)/dataset.num_actions
        fig, ax = plt.subplots()
        ax.plot(val_train_metrics['epoch'], val_train_metrics['unique_bins'], label='train', color='grey', alpha=.5, linestyle='dotted')
        ax.plot(val_metrics['epoch'], val_metrics['unique_bins'], label='val')
        ax.set_ylabel('Unique bins \n (normalized by total number of bins)', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.hlines(y=total_bin_diversity, xmin=0, xmax=np.max(val_train_metrics['epoch']), label='dataset-wide unique bin visit', color='black', linestyle='dotted')
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(results_outdir / 'figures' / 'unique_bins_history.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    
    # Data input and output file and dir setups
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')
    parser.add_argument('--metadata.parent_results_dir', type=str, default='experiment_results', help='Name (not path) of results directory')
    parser.add_argument('--metadata.exp_name', type=str, default='test_experiment', help='Name of the experiment -- used to create the subdir in parents_results_dir')
    parser.add_argument('--metadata.seed', type=int, default=10, help='Random seed for reproducibility')
    
    # Algorithm setup
    parser.add_argument('--model.algorithm', type=str, default='ddqn', help='Algorithm to use for training (DDQN or BC)')
    parser.add_argument('--model.loss_function', type=str, default='cross_entropy', help='Loss function. Options: mse, cross_entropy, huber, mse')
    parser.add_argument('--model.tau', type=float, default=0.005, help='Target network update rate for DDQN')
    parser.add_argument('--model.gamma', type=float, default=0.99, help='Discount factor for DDQN')
    parser.add_argument('--model.activation', type=str, default='relu', help='The activation function to use in the neural network. Options: relu, mish, swish ')

    # Data selection and setup
    parser.add_argument('--data.bin_method', type=str, default='healpix', help='Binning method to use (healpix or uniform)')
    parser.add_argument('--data.nside', type=int, default=16, help='Healpix nside parameter (only used if binning_method is healpix)')
    parser.add_argument('--data.num_bins_1d', type=int, default=16, help='Number of bins in 1dim (only used if binning_method is uniform)')
    parser.add_argument('--data.bin_space', type=str, default='radec', help='Binning space to use (azel or radec)')
    parser.add_argument('--data.specific_years', type=int, nargs='*', default=None, help='Specific years to include in the dataset')
    parser.add_argument('--data.specific_months', type=int, nargs='*', default=None, help='Specific months to include in the dataset')
    parser.add_argument('--data.specific_days', type=int, nargs='*', default=None, help='Specific days to include in the dataset')
    parser.add_argument('--data.specific_filters', type=str, nargs='*', default=None, help='Specific filters to include in the dataset')
    # parser.add_argument('--include_default_features', action='store_true', help='Whether to include default features in the dataset')
    parser.add_argument('--data.include_bin_features', action='store_true', help='Whether to include bin features in the dataset')
    parser.add_argument('--data.do_cyclical_norm', action='store_true', help='Whether to apply cyclical normalization to the features')
    parser.add_argument('--data.do_max_norm', action='store_true', help='Whether to apply max normalization to the features')
    parser.add_argument('--data.do_inverse_norm', action='store_true', help='Whether to include inverse normalizations to features')
    parser.add_argument('--data.remove_large_time_diffs', action='store_true', help='New method of calculating transitions which removes any transitions with time difference greater than 10 min')
    parser.add_argument('--data.additional_bin_features', type=str, nargs='*', default=[], help='Other bin feautures to include')
    parser.add_argument('--data.additional_pointing_features', type=str, nargs='*', default=[], help='Other pointing feautures to include')

    # Training hyperparameters
    parser.add_argument('--train.max_epochs', type=float, default=10, help='Maximum number of passes through train dataset')
    parser.add_argument('--train.batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--train.num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--train.use_train_as_val', action='store_true', help='Instead of using validation samples during training, use the training samples')
    parser.add_argument('--train.lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train.lr_scheduler', type=str, default=None, help='cosine_annealing or None')
    parser.add_argument('--train.lr_scheduler_num_epochs', type=int, default=0, help='Number of epochs to reach min lr (must be less than num_epochs)')
    parser.add_argument('--train.lr_scheduler_epoch_start', type=int, default=100, help='Epoch at which to start lr scheduler')
    parser.add_argument('--train.eta_min', type=float, default=1e-5, help='Minimum learning rate for cosine annealing scheduler')
    parser.add_argument('--train.hidden_dim', type=int, default=1024, help='Hidden dimension size for the model')
    parser.add_argument('--train.patience', type=int, default=0, help='Early stopping patience (in epochs). If 0, patience will not be used.')
    
    args = parser.parse_args()

    # If a config file is passed, overwrite the argparse defaults
    if args.cfg is not None:
        assert Path(args.cfg).exists(), f"Config file at {args.cfg} does not exist."
            
        with open(args.cfg, 'r') as f:
            print(args.cfg)
            file_conf = json.load(f)
            for section, values in file_conf.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        setattr(args, f"{section}.{k}", v)
    return args

def main():
    args = get_args()
    cfg = dict_to_nested(vars(args))
    global_cfg = load_global_config(PROJECT_ROOT / 'configs' / 'global_config.json')

    results_outdir = PROJECT_ROOT / Path(cfg['metadata']['parent_results_dir']) / cfg['metadata']['exp_name']
    fig_outdir = results_outdir / 'figures'
    if not os.path.exists(results_outdir):
        os.makedirs(results_outdir)
    if not os.path.exists(fig_outdir):
        os.makedirs(fig_outdir)

    # Set up logging
    logger = setup_logger(save_dir=results_outdir, logging_filename='training.log')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)
    logger.info(f'Setting outdir {results_outdir}')

    # Get training configs used more than once
    batch_size = cfg['train']['batch_size'] #cfg.get('experiment.training.batch_size')
    max_epochs = cfg['train']['max_epochs'] #cfg.get('experiment.training.max_epochs')
    lr_scheduler = cfg['train']['lr_scheduler'] #cfg.get('experiment.training.lr_scheduler')
    lr_scheduler_epoch_start = cfg['train']['lr_scheduler_epoch_start'] #cfg.get('experiment.training.lr_scheduler_epoch_start')
    lr_scheduler_num_epochs = cfg['train']['lr_scheduler_num_epochs'] #cfg.get('experiment.training.lr_scheduler_num_epochs')
    bin_space = cfg['data']['bin_space'] #cfg.get('experiment.data.bin_space')
    if bin_space == 'azel':
        cfg['data']['additional_bin_features'] += ['ra', 'dec']
    else:
        cfg['data']['additional_bin_features'] += ['az', 'el']
    cfg['data']['additional_bin_features'] = list(set(cfg['data']['additional_bin_features']))
    
    # assert errors dne before running rest of code
    if lr_scheduler is not None:
        assert max_epochs - lr_scheduler_epoch_start - lr_scheduler_num_epochs >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_num_epochs"

    logger.info("Saving results in " + str(results_outdir))

    # Seed everything
    seed_everything(cfg['metadata']['seed'])

    device = get_device()

    logger.info("Loading raw data...")

    df = load_raw_data_to_dataframe(Path(global_cfg['paths']['FITS_DIR']) / Path(global_cfg['files']['DECFITS']), 
                                    Path(global_cfg['paths']['FITS_DIR']) / Path(global_cfg['files']['DECJSON']))

    logger.info("Processing raw data into OfflineDataset()...")
    # Need to include paths.lookup_dir in cfg before sending to offline dataset -- brittle
    train_dataset = OfflineDECamDataset(
        df=df,
        cfg=cfg,
        glob_cfg=global_cfg
        )
    logger.info("Finished constructing train_dataset")

    # Plot bin membership for fields in ra vs dec
    colors = [f'C{i}' for i in range(7)]
    for i, (bin_id, g) in enumerate(train_dataset._df.groupby('bin')):
        plt.scatter(g.ra, g.dec, label=bin_id, color=colors[i%len(colors)], s=1)
    plt.title("Fields in train data, colored by bin membership")
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(fig_outdir / 'train_data_fields_dec_vs_ra.png')

    logger.info("Plotting S x A (state x action) space cornerplot (this will take some time...)")
    # Plot State x action space via cornerplot
    corner_plot = sns.pairplot(train_dataset._df,
             vars=train_dataset.pointing_feature_names + ['bin'],
             kind='hist',
             corner=True
            )
    corner_plot.figure.savefig(fig_outdir / 'state_times_action_space_corner_plot.png')
    logger.info("Corner plot saved")

    fig, axs = plt.subplots(len(train_dataset.pointing_feature_names), figsize=(4, len(train_dataset.pointing_feature_names)*3))
    next_pointing_states = train_dataset.next_states.T
    for i, feat_name in enumerate(train_dataset.pointing_feature_names):
        axs[i].hist(next_pointing_states[i])
        axs[i].set_title(f"Train distribution ({feat_name})")
    fig.tight_layout()
    fig.savefig(fig_outdir / 'train_data_pointing_feature_distributions.png')
        
    if cfg['train']['use_train_as_val']:
        trainloader = train_dataset.get_dataloader(batch_size, num_workers=cfg['train']['num_workers'], pin_memory=True if device.type == 'cuda' else False, random_seed=cfg.get('experiment.metadata.seed'), return_train_and_val=False)
        valloader = trainloader
    else:
        trainloader, valloader = train_dataset.get_dataloader(batch_size, num_workers=cfg['train']['num_workers'], pin_memory=True if device.type == 'cuda' else False, random_seed=cfg['metadata']['seed'], return_train_and_val=True)

    # Initialize algorithm and agent
    logger.info("Initializing agent...")

    steps_per_epoch = np.max([int(len(trainloader.dataset) // batch_size), 1])
    num_lr_scheduler_steps = np.max([1, int(lr_scheduler_num_epochs * steps_per_epoch)])
    lr_scheduler_kwargs = {'T_max': num_lr_scheduler_steps, 'eta_min': cfg['train']['eta_min']} if lr_scheduler == 'cosine_annealing' else {}

    algorithm = setup_algorithm(algorithm_name=cfg['model']['algorithm'], 
                                obs_dim=train_dataset.obs_dim, num_actions=train_dataset.num_actions, loss_fxn=cfg['model']['loss_function'],
                                hidden_dim=cfg['train']['hidden_dim'], lr=cfg['train']['lr'], lr_scheduler=lr_scheduler, 
                                device=device, lr_scheduler_kwargs=lr_scheduler_kwargs, lr_scheduler_epoch_start=lr_scheduler_epoch_start, 
                                lr_scheduler_num_epochs=lr_scheduler_num_epochs, gamma=cfg['model']['gamma'], 
                                tau=cfg['model']['tau'], activation=cfg['model']['activation'])

    agent = Agent(
        algorithm=algorithm,
        train_outdir=str(results_outdir) + '/',
    )

    # Save (or update) config file after updating
    cfg['data']['obs_dim'] = train_dataset.obs_dim
    cfg['data']['num_actions'] = train_dataset.num_actions
    cfg['metadata']['outdir'] = str(PROJECT_ROOT / cfg['metadata']['parent_results_dir'] / cfg['metadata']['exp_name'])
    cfg['train']['lr_scheduler_kwargs'] = {key: int(val) for key, val in lr_scheduler_kwargs.items()}
    save_config(config_dict=cfg, outdir=results_outdir)
    logger.info("Starting training...")

    # Train agent
    start_time = time.time()
    agent.fit(
        num_epochs=max_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=batch_size,
        patience=cfg['train']['patience'],
        hpGrid=train_dataset.hpGrid
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")
    logger.info("Plotting training loss curve...")
    plot_metrics(results_outdir, dataset=train_dataset)

    # Plot predicted action for each state in train dataset
    dataset = trainloader.dataset.dataset
    val_states, val_actions, _, _, _, _ = dataset[valloader.dataset.indices]
    train_states, train_actions, _, _, _, _ = dataset[trainloader.dataset.indices]
    for prefix, (states, actions) in zip(['val_', 'train_'], [ (val_states, val_actions), (train_states, train_actions) ]):
        with torch.no_grad():
            q_vals = agent.algorithm.policy_net(states.to(device))
            eval_actions = torch.argmax(q_vals, dim=1).to('cpu').detach().numpy()

        # Sequence of actions from target (original schedule) and policy
        target_sequence = actions.detach().numpy()
        eval_sequence = eval_actions
        first_night_indices = np.where(states[:, -1] == 0)

        fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
        
        axs[0].plot(target_sequence, marker='*', alpha=.3, label='true')
        axs[0].plot(eval_sequence, marker='o', alpha=.3, label='pred')
        axs[0].legend()
        axs[0].set_ylabel('bin number')
        axs[0].vlines(first_night_indices, ymin=0, ymax=len(dataset.hpGrid.lon), color='black', linestyle='--')
        axs[1].plot(eval_sequence - target_sequence, marker='o', alpha=.5)
        axs[1].set_ylabel('Eval sequence - target sequence \n[bin number]')
        axs[1].set_xlabel('observation index')
        fig.savefig(fig_outdir / (prefix + 'eval_and_target_bin_sequences.png'))

if __name__ == "__main__":
    main()