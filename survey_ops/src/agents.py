import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
import os
import pickle

class Agent:
    """
    A simple, generic agent/wrapper for fitting and evaluating Q-learning algorithms. 

    Args
    ----
        algorithm (Algorithm): The Q-learning algorithm
        env (gymnasium.Env): The environment in which the agent will act.
        name (str): File name prefix for policy net weights
        normalize_obs (bool): Whether or not to normalize observations
    """
    def __init__(
            self,
            algorithm,
            normalize_obs,
            outdir,
            env: gym.Env = None,
            ):
        self.algorithm = algorithm
        self.device = algorithm.device
        self.normalize_obs = normalize_obs
        self.env = env
        self.outdir = outdir

    def fit(self, dataset, num_epochs, batch_size):
        train_metrics = {
            'loss_history': [],
            'q_history': [],
            'test_acc_history': [] 
        }

        for i_epoch in tqdm(range(num_epochs)):
            batch = dataset.sample(batch_size)
            
            loss, q_val = self.algorithm.train_step(batch)
            train_metrics['loss_history'].append(loss)
            train_metrics['q_history'].append(q_val)

            if i_epoch % 100 == 0:
                with torch.no_grad():
                    # Test on a batch
                    obs, expert_actions, _, _, _, action_masks = dataset.sample(batch_size)
                    obs = torch.tensor(obs, device=self.device)
                    expert_actions = torch.tensor(expert_actions, device=self.device)
                    all_q_vals = self.algorithm.policy_net(obs)
                    all_q_vals[~action_masks] = float('-inf')
                    predicted_actions = all_q_vals.argmax(dim=1)
                    accuracy = (predicted_actions == expert_actions).float().mean()
                    train_metrics['test_acc_history'].append(accuracy.cpu().detach().numpy())
                    print(f"Epoch {i_epoch}: Accuracy = {accuracy:.3f}, Loss = {loss:.4f}")
            
        version_num = 0
        save_filepath = self.outdir + 'weights' + f'-v{version_num}.pt'
        self.save(save_filepath)
        train_metrics_filepath = self.outdir + 'train_metrics.pkl'
        with open(train_metrics_filepath, 'wb') as handle:
            pickle.dump(train_metrics, handle)

    
    def evaluate(self, env, num_episodes):
        #TODO eval metric updating as attribute
        # evaluation metrics
        observations = {}
        rewards = {}
        self.algorithm.policy_net.eval()
        episode_rewards = []
        eval_metrics = {}
        
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            obs_list = [obs]
            rewards_list = [0]
            i = 0

            while not (terminated or truncated):
                with torch.no_grad():
                    print(obs)
                    obs = obs / env.unwrapped.norm
                    action_mask = info.get('action_mask', None)
                    action = self.act(obs, action_mask, epsilon=None)  # greedy
                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_list.append(obs)
                    rewards_list.append(reward)
                    episode_reward += reward
                    i += 1
            print(f'terminated at {i}')
            observations.update({f'ep-{episode}': np.array(obs_list)})
            rewards.update({f'ep-{episode}': np.array(rewards_list)})
            episode_rewards.append(episode_reward)

        eval_metrics.update({
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
            'observations': observations,
            'rewards': rewards
        })

        with open(self.outdir + 'eval_metrics.pkl', 'wb') as handle:
            pickle.dump(eval_metrics, handle)
            print('dumped ')

    def act(self, obs, action_mask, epsilon):
        return self.algorithm.select_action(obs, action_mask, epsilon)
    
    def save(self, filepath):
        self.algorithm.save(filepath)
    
    def load(self, filepath):
        self.algorithm.load(filepath)

    # def get_next_available_version_filename(base_filename):
    #     """
    #     Finds the next available versioned filename (e.g., file_v1.txt, file_v2.txt).
    #     """
    #     name, ext = os.path.splitext(base_filename)
    #     version = 0
    #     while True:
    #         versioned_filename = f"{name}_v{version}{ext}"
    #         if not os.path.exists(versioned_filename):
    #             return versioned_filename
    #         version += 1