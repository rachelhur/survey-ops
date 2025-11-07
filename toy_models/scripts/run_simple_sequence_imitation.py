import numpy as np
import matplotlib.pyplot as plt
import random

import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C, DDPG, SAC, TD3, HER
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import optuna

import os
import time
import pickle
import sys
sys.path.append('../src/')
from environments import SimpleTelEnv
from utils import save_results, load_results

import sys
import argparse

def cli_args():
        # Make parser object
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("-Nf", "--Nfields", type=int)
    parser.add_argument("-vmax", "--max-visits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("-v", "--verbosity", type=bool)
    parser.add_argument("--timesteps", type=int, default=1e4)
                   
    return(parser.parse_args())

# python run_simple_sequence_imitation.py -Nf 100 -vmax 3 --seed 20 -v True --timesteps 1e5

if __name__ == '__main__':
    args = cli_args()

    # Set parameters from command line args
    Nf = args.Nfields
    nv_max = args.max_visits
    random.seed(args.seed)
    true_sequence = np.array([np.full(nv_max, i) for i in range(Nf)]).flatten()
    true_sequence = true_sequence.tolist()
    random.shuffle(true_sequence)

    # Set other parameters
    env_name = 'SimpleTel-v0'
    OUTDIR = f'results/{env_name}/'
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    # Register the environment so we can create it with gym.make()
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=SimpleTelEnv,
        max_episode_steps=300,  # Prevent infinite episodes. Here just set to 300 even though episode will terminate when stepping to last element of sequence
    )

    # create environment
    env = gym.make(f"gymnasium_env/{env_name}", Nf=Nf, target_sequence=true_sequence, nv_max=nv_max, off_by_lim=3)

    # Dictionary of models to train
    model_kwargs = {'policy': "MultiInputPolicy", "env": env, "verbose": args.verbosity}
    models = {'A2C': A2C, 'DQN':DQN, 'PPO':PPO} 
    results = {model_name: {'model': model(**model_kwargs)} for model_name, model in models.items()}

    # Train or load models
    for name in results.keys():
        file_path = OUTDIR + env_name + f'-Nf={Nf}-vmax={nv_max}-seed={args.seed}-nsteps={args.timesteps}_{name}'
    try:
        results[name]['model'] = results[name]['model'].load(file_path, env=env)
    except:
        t_start = time.time()
        results[name]['model'].learn(total_timesteps=args.timesteps, log_interval=10)
        t_stop = time.time()
        # results[name]['train_time'] = t_stop - t_start
        results[name]['model'].save(path=file_path)
    
    # Print train times
    # for name, model_dict in results.items():
    #     print(name, model_dict['train_time'])

    # Evaluate models
    observation_list = np.empty(shape=(3, len(true_sequence)), dtype=dict)
    reward_list = np.empty_like(observation_list)
    terminated_list = np.empty_like(observation_list)
    truncated_list = np.empty_like(observation_list)
    info_list = np.empty_like(observation_list, dtype=dict)

    for j, (name, model_dict) in enumerate(results.items()):
        obs, info = env.reset(seed=args.seed)
        observation_list[j, 0] = obs
        info_list[j, 0] = info
        for i in range(len(true_sequence)):
            action, _states = results[name]['model'].predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.tolist())
            observation_list[j,i+1] = obs
            info_list[j, i+1] = info
            reward_list[j, i] = reward
            terminated_list[j, i] = terminated
            truncated_list[j, i] = truncated
            if terminated or truncated:
                break

    # Extract proposed survey sequences
    proposed_survey = np.empty_like(observation_list)
    for j, obs_list in enumerate(observation_list):
        proposed_survey[j] = [observation_list[j, i]['field_id'] for i in range(len(observation_list[j]))]

    # Plot sequences
    fig, axs = plt.subplots(2, 3, figsize=(15,8), sharex='row')
    fig_hist, ax_hist = plt.subplots()
    for i, model_name in enumerate(results.keys()):
        axs[0,i].set_title(model_name, fontsize=20)
        axs[0,i].plot(np.array(true_sequence), marker='o', label='target', color='grey', linestyle='dashed', lw=3)
        axs[0,i].plot(np.array(proposed_survey[i]), marker='o', label='pred')
        axs[0,i].legend()
        axs[0,i].set_ylabel('Object number', fontsize=16)

        residuals = np.array(true_sequence) - np.array(proposed_survey[i])
        axs[1,i].plot(residuals, marker='o')
        axs[1, i].set_xlabel('sequence index', fontsize=16)
        axs[1,i].set_ylabel('residuals', fontsize=16)
        
        ax_hist.hist(residuals, label=model_name, alpha=.5)
        ax_hist.set_xlabel('Residual value')
        ax_hist.set_ylabel('Counts')
        ax_hist.legend(fontsize=16)
    fig.tight_layout()

    fig.savefig(OUTDIR + f'sequence_Nf={Nf}_vmax={nv_max}_seed={args.seed}-nsteps={args.timesteps}.png')
    fig_hist.savefig(OUTDIR + f'sequence_residuals-Nf={Nf}-vmax={nv_max}-seed={args.seed}-nsteps={args.timesteps}.png')

    # Evaluate policies
    n_episodes = 50
    rewards = np.empty(shape=(len(true_sequence), n_episodes))
    for i, model_name in enumerate(results.keys()):
        mean, std = evaluate_policy(results[model_name]['model'], Monitor(env), n_eval_episodes=n_episodes, deterministic=False)
        rewards[i], epsiodes = evaluate_policy(results[model_name]['model'], Monitor(env), n_eval_episodes=n_episodes, return_episode_rewards=True, deterministic=False)
        print(model_name, "mean =", f"{mean:.2f}", ', std = ', f"{std:.2f}")

    # Plot rewards
    fig, ax = plt.subplots()
    for i, model_name in enumerate(results.keys()):
        ax.plot(rewards[i], label=model_name, marker='o')
        ax.set_xlabel('Episode number', fontsize=16)
        ax.set_ylabel('Total reward', fontsize=16)
        ax.legend(fontsize=12)
    fig.savefig(OUTDIR + f'rewards-Nf={Nf}-vmax={nv_max}-seed={args.seed}-nsteps={args.timesteps}.png')