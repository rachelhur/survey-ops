import os
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

import time
import pickle

from survey_ops.utils import pytorch_utils
from survey_ops.src.agents import Agent
from survey_ops.src.algorithms import DDQN, BehaviorCloning
from survey_ops.utils.script_utils import setup_logger, get_device, load_raw_data_to_dataframe, setup_algorithm
from survey_ops.src.offline_dataset import OfflineDECamDataset
import argparse


def plot_loss_curve(results_outdir):
    with open(results_outdir + 'train_metrics.pkl', 'rb') as f:
        train_metrics = pickle.load(f)
    with open(results_outdir + 'val_metrics.pkl', 'rb') as f:
        val_metrics = pickle.load(f)
    fig, axs = plt.subplots(2, sharex=True, figsize=(5, 5))
    axs[0].plot(train_metrics['train_loss'])
    axs[0].hlines(y=0, xmin=0, xmax=len(train_metrics['train_loss']), color='red', linestyle='--')
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[1].plot(np.linspace(0, len(train_metrics['train_loss']), len(val_metrics['val_mean_accuracy'])), val_metrics['val_mean_accuracy'])
    axs[1].hlines(y=1, xmin=0, xmax=len(train_metrics['train_loss']), color='red', linestyle='--')
    axs[1].set_xlabel('Train step', fontsize=14)
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].set_xlabel('Train step', fontsize=14)
    
    fig.tight_layout()
    fig.savefig(results_outdir + 'figures/' + 'train_history.png')
    plt.show()

def main():

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--num_epochs', type=float, default=10, help='Number of iterations through dataset during training')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None, help='cosine_annealing or None')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension size for the model')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for cosine annealing scheduler')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (in epochs)')
    
    # Algorithm setup
    parser.add_argument('--algorithm_name', type=str, default='ddqn', help='Algorithm to use for training (ddqn or behavior_cloning)')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', help='Loss function. Options: mse, cross_entropy, huber, mse')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate for DDQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for DDQN')

    
    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Set up results directory to save outputs
    results_outdir = args.parent_results_dir + args.exp_name + '/'
    fig_outdir = results_outdir + 'figures/'
    if not os.path.exists(results_outdir):
        os.makedirs(results_outdir)
    if not os.path.exists(fig_outdir):
        os.makedirs(fig_outdir)

    # Set up logging
    logger = setup_logger(save_dir=results_outdir, logging_filename='training.log')
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
    
    colors = [f'C{i}' for i in range(7)]
    for i, (bin_id, g) in enumerate(train_dataset._df.groupby('bin')):
        plt.scatter(g.ra, g.dec, label=bin_id, color=colors[i%len(colors)], s=1)
    plt.savefig(fig_outdir + 'train_data_dec_vs_ra.png')


    with open(results_outdir + 'offline_dataset_config.pkl', 'wb') as f:
        pickle.dump(OFFLINE_DATASET_CONFIG, f)
    
    trainloader = train_dataset.get_dataloader(args.batch_size, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    # Initialize algorithm and agent
    logger.info("Initializing agent...")
    batches_per_dataset = len(train_dataset) // args.batch_size
    lr_scheduler_kwargs = {'T_max': batches_per_dataset, 'eta_min': args.eta_min} if args.lr_scheduler == 'cosine_annealing' else {}

    algorithm = setup_algorithm(save_dir=results_outdir, algorithm_name=args.algorithm_name, obs_dim=train_dataset.obs_dim, num_actions=train_dataset.num_actions, \
                                loss_fxn=args.loss_function, hidden_dim=args.hidden_dim, lr=args.lr, lr_scheduler=args.lr_scheduler, device=device, \
                                    lr_scheduler_kwargs=lr_scheduler_kwargs, gamma=args.gamma, tau=args.tau)
    
    agent = Agent(
        algorithm=algorithm,
        train_outdir=results_outdir,
    )
    logger.info("Starting training...")

    # Train agent
    start_time = time.time()
    agent.fit(
        num_epochs=args.num_epochs,
        dataloader=trainloader,
        batch_size=args.batch_size,
        eval_freq=100,
        patience=args.patience
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")
    logger.info("Plotting training loss curve...")
    plot_loss_curve(results_outdir)


if __name__ == "__main__":
    main()