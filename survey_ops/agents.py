import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

class Agent:
    """
    A simple, generic agent for Q-learning. 

    Args
    ----
        algorithm (Algorithm): The Q-learning algorithm
        env (gymnasium.Env): The environment in which the agent will act.
    """
    def __init__(
            self,
            algorithm,
            name,
            env: gym.Env = None,
            ):
        self.algorithm = algorithm
        self.device = algorithm.device
        self.name = name
        self.loss_history = []
        self.q_history = []

    def fit(self, dataset, num_epochs, batch_size, outdir, version_num=0):

        for i_epoch in tqdm(range(num_epochs)):
            batch = dataset.sample(batch_size)
            
            loss, q_val = self.algorithm.train_step(batch)
            self.loss_history.append(loss)
            self.q_history.append(q_val)

        save_filepath = outdir + self.name + f'-v{version_num}.pt'
        self.save(save_filepath)


    def evaluate(self, env, num_episodes):
        # evaluation metrics
        episode_rewards = []
        observations = {}
        rewards = {}
        
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            obs_list = [obs[0]]
            rewards_list = []
            i = 0

            while not (terminated or truncated):
                action_mask = info.get('action_mask', None)
                action = self.act(obs, action_mask, epsilon=None)  # greedy
                obs, reward, terminated, truncated, info = env.step(action)
                obs_list.append(obs)
                if terminated:
                    print(f'terminated at step {i}')
                rewards_list.append(reward)
                episode_reward += reward
                i += 1
            print(obs_list)
            observations.update({f'ep-{episode}': np.array(obs_list)})
            rewards.update({f'ep-{episode}': np.array(rewards_list)})
            episode_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
            'observations': observations,
            'rewards': rewards
        }

    def act(self, obs, action_mask, epsilon):
        return self.algorithm.select_action(obs, action_mask, epsilon)
    
    def save(self, filepath):
        self.algorithm.save(filepath)
    
    def load(self, filepath):
        self.algorithm.load(filepath)