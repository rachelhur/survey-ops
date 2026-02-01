import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

import time
import pickle

from survey_ops.utils import pytorch_utils
from survey_ops.src.agents import Agent
from survey_ops.src.algorithms import DDQN, BehaviorCloning
from survey_ops.utils.script_utils import setup_logger, get_device, load_raw_data_to_dataframe, setup_algorithm
from survey_ops.src.offline_dataset import OfflineDECamDataset

import argparse
import logging


def plot_metrics(results_outdir):
    with open(results_outdir + 'train_metrics.pkl', 'rb') as f:
        train_metrics = pickle.load(f)
    with open(results_outdir + 'val_metrics.pkl', 'rb') as f:
        val_metrics = pickle.load(f)
    with open(results_outdir + 'val_train_metrics.pkl', 'rb') as f:
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
    axs[2].set_xlabel('Epoch', fontsize=14)

    i = 0
    for key in val_metrics.keys():
        if key != 'accuracy' and key != 'epoch' and 'loss' not in key:
            axs[3].plot(val_metrics['epoch'], val_metrics[key], label='val ' + key, color=f"C{i}")
            axs[3].plot(val_metrics['epoch'], val_train_metrics[key], color=f"C{i}", linestyle='dotted', alpha=.5)
            i += 1
    axs[3].hlines(0, xmin=0, xmax=np.max(val_metrics['epoch']), linestyle='--', color='red')
    axs[3].legend()

    fig.tight_layout()
    fig.savefig(results_outdir + 'figures/' + 'loss_and_metrics_history.png')

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    
    # Data input and output file and dir setups
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')
    parser.add_argument('--parent_results_dir', type=str, default='../experiment_results/', help='Path to save trained model')
    parser.add_argument('--exp_name', type=str, default='test_experiment', help='Name of the experiment -- used to create the output directory')
    
    # Data selection and setup
    parser.add_argument('--binning_method', type=str, default='healpix', help='Binning method to use (healpix or grid)')
    parser.add_argument('--nside', type=int, default=16, help='Healpix nside parameter (only used if binning_method is healpix)')
    parser.add_argument('--bin_space', type=str, default='radec', help='Binning space to use (azel or radec)')
    parser.add_argument('--specific_years', type=int, nargs='*', default=None, help='Specific years to include in the dataset')
    parser.add_argument('--specific_months', type=int, nargs='*', default=None, help='Specific months to include in the dataset')
    parser.add_argument('--specific_days', type=int, nargs='*', default=None, help='Specific days to include in the dataset')
    # parser.add_argument('--include_default_features', action='store_true', help='Whether to include default features in the dataset')
    parser.add_argument('--include_bin_features', action='store_true', help='Whether to include bin features in the dataset')
    # parser.add_argument('--do_z_score_norm', action='store_true', help='Whether to apply z-score normalization to the features')
    parser.add_argument('--do_cyclical_norm', action='store_true', help='Whether to apply cyclical normalization to the features')
    parser.add_argument('--do_max_norm', action='store_true', help='Whether to apply max normalization to the features')
    parser.add_argument('--do_inverse_airmass', action='store_true', help='Whether to include inverse airmass as a feature')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=float, default=10, help='Number of passes through train dataset')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use_train_as_val', action='store_true', help='Instead of using validation samples during training, use the training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None, help='cosine_annealing or None')
    parser.add_argument('--lr_scheduler_num_epochs', type=int, default=0, help='Number of epochs to reach min lr (must be less than num_epochs)')
    parser.add_argument('--lr_scheduler_epoch_start', type=int, default=100, help='Epoch at which to start lr scheduler')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for cosine annealing scheduler')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension size for the model')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (in epochs). If 0, patience will not be used.')
    
    # Algorithm setup
    parser.add_argument('--algorithm_name', type=str, default='ddqn', help='Algorithm to use for training (ddqn or behavior_cloning)')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', help='Loss function. Options: mse, cross_entropy, huber, mse')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate for DDQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for DDQN')
    parser.add_argument('--activation', type=str, default='relu', help='The activation function to use in the neural network. Options: relu, mish, swish ')

    
    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # assert errors dne before running rest of code
    if args.lr_scheduler is not None:
        assert args.num_epochs - args.lr_scheduler_epoch_start - args.lr_scheduler_num_epochs >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_num_epochs"

    # Set up results directory to save outputs
    results_outdir = args.parent_results_dir + args.exp_name + '/'
    fig_outdir = results_outdir + 'figures/'
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


    logger.info("Saving results in " + results_outdir)

    # Print args
    logger.info("Experiment parameters:")
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")

    # Seed everything
    pytorch_utils.seed_everything(args.seed)
    # torch.set_default_dtype(torch.float32)

    device = get_device()

    logger.info("Loading raw data...")
    raw_data_df = load_raw_data_to_dataframe(args.fits_path, args.json_path)

    # Save these args for test data arguments in eval.py
    OFFLINE_DATASET_CONFIG = {
        'binning_method': args.binning_method,
        'nside': args.nside,
        'bin_space': args.bin_space,
        'include_default_features': True,
        'include_bin_features': args.include_bin_features,
        'do_cyclical_norm': args.do_cyclical_norm,
        'do_max_norm': args.do_max_norm,
        'do_inverse_airmass': args.do_inverse_airmass,
        'calculate_action_mask': 'dqn' in args.algorithm_name
    }
    
    logger.debug(f'Offline dataset config: {OFFLINE_DATASET_CONFIG}')

    logger.info("Processing raw data into OfflineDataset()...")
    train_dataset = OfflineDECamDataset(
        df=raw_data_df,
        specific_years=args.specific_years, 
        specific_months=args.specific_months, 
        specific_days=args.specific_days,
        **OFFLINE_DATASET_CONFIG
        )
    logger.info("Finished constructing train_dataset")

    assert args.batch_size 
    
    # Plot bin membership for fields in ra vs dec
    colors = [f'C{i}' for i in range(7)]
    for i, (bin_id, g) in enumerate(train_dataset._df.groupby('bin')):
        plt.scatter(g.ra, g.dec, label=bin_id, color=colors[i%len(colors)], s=1)
    plt.title("Fields in train data, colored by bin membership")
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(fig_outdir + 'train_data_fields_dec_vs_ra.png')

    logger.info("Plotting S x A (state x action) space cornerplot (this will take some time...)")
    # Plot State x action space via cornerplot
    corner_plot = sns.pairplot(train_dataset._df,
             vars=train_dataset.pointing_feature_names + ['bin'],
             kind='hist',
             corner=True
            )
    corner_plot.figure.savefig(fig_outdir + 'state_times_action_space_corner_plot.png')
    logger.info("Corner plot saved")

    fig, axs = plt.subplots(len(train_dataset.pointing_feature_names), figsize=(4, len(train_dataset.pointing_feature_names)*3))
    next_pointing_states = train_dataset.next_states.T
    for i, feat_name in enumerate(train_dataset.pointing_feature_names):
        axs[i].hist(next_pointing_states[i])
        axs[i].set_title(f"Train distribution ({feat_name})")
    fig.tight_layout()
    fig.savefig(fig_outdir + 'train_data_pointing_feature_distributions.png')
        
    with open(results_outdir + 'offline_dataset_config.pkl', 'wb') as f:
        pickle.dump(OFFLINE_DATASET_CONFIG, f)
    
    if args.use_train_as_val:
        trainloader = train_dataset.get_dataloader(args.batch_size, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False, random_seed=args.seed, return_train_and_val=False)
        valloader = trainloader
    else:
        trainloader, valloader = train_dataset.get_dataloader(args.batch_size, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False, random_seed=args.seed, return_train_and_val=True)

    # valloader = train_dataset.get_dataloader(args.batch_size, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False, random_seed=np.random.randint(low=0, high=10000))

    # Initialize algorithm and agent
    logger.info("Initializing agent...")

    steps_per_epoch = np.max([int(len(trainloader.dataset) // args.batch_size), 1])
    logger.debug(f'{len(trainloader.dataset)} // {args.batch_size}')
    num_lr_scheduler_steps = np.max([1, int(args.lr_scheduler_num_epochs * steps_per_epoch)])
    logger.debug(f'lr_scheduler_num_epochs {args.lr_scheduler_num_epochs} * {steps_per_epoch}')
    lr_scheduler_kwargs = {'T_max': num_lr_scheduler_steps, 'eta_min': args.eta_min} if args.lr_scheduler == 'cosine_annealing' else {}
    logger.debug(f'lr_scheduler_kwargs {lr_scheduler_kwargs}')

    algorithm = setup_algorithm(save_dir=results_outdir, algorithm_name=args.algorithm_name, obs_dim=train_dataset.obs_dim, num_actions=train_dataset.num_actions, \
                                loss_fxn=args.loss_function, hidden_dim=args.hidden_dim, lr=args.lr, lr_scheduler=args.lr_scheduler, device=device, \
                                lr_scheduler_kwargs=lr_scheduler_kwargs, lr_scheduler_epoch_start=args.lr_scheduler_epoch_start, \
                                lr_scheduler_num_epochs=args.lr_scheduler_num_epochs, gamma=args.gamma, tau=args.tau, activation=args.activation)

    agent = Agent(
        algorithm=algorithm,
        train_outdir=results_outdir,
    )
    logger.info("Starting training...")

    # Train agent
    start_time = time.time()
    agent.fit(
        num_epochs=args.num_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")
    logger.info("Plotting training loss curve...")
    plot_metrics(results_outdir)

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
        fig.savefig(fig_outdir + prefix + 'eval_and_target_bin_sequences.png')

if __name__ == "__main__":
    main()