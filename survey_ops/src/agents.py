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
            normalize_obs,
            env: gym.Env = None,
            ):
        self.algorithm = algorithm
        self.device = algorithm.device
        self.name = name
        self.normalize_obs = normalize_obs
        self.loss_history = []
        self.q_history = []
        self.accuracies = []

    def fit(self, dataset, num_epochs, batch_size, outdir, version_num=0):

        for i_epoch in tqdm(range(num_epochs)):
            batch = dataset.sample(batch_size)
            
            loss, q_val = self.algorithm.train_step(batch)
            self.loss_history.append(loss)
            self.q_history.append(q_val)

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
                    self.accuracies.append(accuracy.cpu().detach().numpy())
                    print(f"Epoch {i_epoch}: Accuracy = {accuracy:.3f}, Loss = {loss:.4f}")
            
        save_filepath = outdir + self.name + f'-v{version_num}.pt'
        self.save(save_filepath)
    
    
    # def evaluate():
    #     states = torch.tensor(np.array([d['state'] for d in test_dataset]), dtype=torch.float32)
    #     expert_actions = torch.tensor([d['expert_action'] for d in test_dataset], dtype=torch.long)
        
    #     states = states.to(self.device)
    #     expert_actions = expert_actions.to(self.device)
        
    #     with torch.no_grad():
    #         action_logits = self.policy_net(states)
    #         predicted_actions = torch.argmax(action_logits, dim=1)
    #         accuracy = (predicted_actions == expert_actions).float().mean().item()

    def evaluate(self, env, num_episodes):
        # evaluation metrics
        episode_rewards = []
        observations = {}
        rewards = {}
        self.algorithm.policy_net.eval()
        
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