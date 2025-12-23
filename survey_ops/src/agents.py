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

    This class abstracts training loops, evaluation, saving/loading, and interaction with environment. It expects an underlying `algorithm` object that
    implements:
        - `algorithm.train_step(batch)`
        - `algorithm.select_action(obs, mask, epsilon)`
        - `algorithm.policy_net`
        - `algorithm.save(path)`
        - `algorithm.load(path)`

    Attributes
    ----------
        algorithm (Algorithm): Q-learning algorithm implementing train + act.
        device (str): Device used by the algorithm ('cpu' or 'cuda').
        normalize_obs (bool): Whether to normalize observations before acting.
        env (gym.Env | None): Optional environment for evaluation.
        outdir (str): Directory for saving weights and training/evaluation logs.

    """
    def __init__(
            self,
            algorithm,
            normalize_obs,
            outdir,
            env: gym.Env = None,
            ):
        """
        Args
        ----
            algorithm (Algorithm): The Q-learning algorithm
            env (gymnasium.Env): The environment in which the agent will act.
            outdir (str): directory to save results
            normalize_obs (bool): Whether or not to normalize observations
        """
        self.algorithm = algorithm
        self.device = algorithm.device
        self.normalize_obs = normalize_obs
        self.env = env
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.outdir = outdir
        

    def fit(self, num_epochs, dataset=None, batch_size=None, dataloader=None, eval_freq=100, patience=10):
        """Trains the agent on a transition dataset.

        Uses repeated sampling from a dataset that implements `sample(batch_size)`
        to perform Q-learning updates. Periodically evaluates accuracy on
        expert actions to monitor learning progress.

        Args:
            dataset (object):
                Must implement `sample(batch_size)` returning a full transition:
                (obs, actions, rewards, next_obs, dones, action_masks).
            num_epochs (int):
                Number of passes through the dataset (used to compute steps).
            batch_size (int):
                Number of transitions per optimization step.
            eval_freq (int):
                Frequency (in optimization steps) at which evaluation batches
                are sampled and accuracy is recorded.

        Saves:
            - `<outdir>/weights.pt`: Final model weights.
            - `<outdir>/train_metrics.pkl`: Dictionary containing:
                - `loss_history`
                - `q_history`
                - `test_acc_history`
        """
        # assert dataset is not None and dataloader is not None
        if dataloader is not None:
            assert batch_size is not None
        
        train_metrics = {
            'loss_history': [],
            'q_history': [],
            'test_acc_history': [] 
        }

        save_filepath = self.outdir + 'best_weights.pt'

        if dataloader is not None:
            # TODO for v0 only - remove when model is updated for release
            dataset_size = len(dataloader.dataset)
            total_steps = int(num_epochs * dataset_size)
            loader_iter = iter(dataloader)  # create iterator
        else:
            dataset_size = np.prod(dataset.obs.shape[1:])
            total_steps = int(num_epochs * dataset_size / batch_size)
            loader_iter = None  # not used for manual sampling

        best_val_loss = 1e5
        for i_step in tqdm(range(total_steps)):
            if dataloader is not None:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(dataloader)
                    batch = next(loader_iter)
            else:
                batch = dataset.sample(batch_size)

            loss, q_val = self.algorithm.train_step(batch)
            train_metrics['loss_history'].append(loss)
            train_metrics['q_history'].append(q_val)

            if i_step % eval_freq == 0:
                with torch.no_grad():
                    if dataloader is not None:
                        # --- evaluation from dataloader ---
                        try:
                            eval_batch = next(eval_iter)
                        except (NameError, StopIteration):
                            eval_iter = iter(dataloader)
                            eval_batch = next(eval_iter)
                        eval_obs, expert_actions, _, _, _, action_masks = eval_batch
                    else:
                        # --- old method fallback ---
                        eval_obs, expert_actions, _, _, _, action_masks = dataset.sample(batch_size)
                    
                    # Test on a batch
                    eval_obs = torch.tensor(eval_obs, device=self.device)
                    expert_actions = torch.tensor(expert_actions, device=self.device)
                    all_q_vals = self.algorithm.policy_net(eval_obs)
                    if self.algorithm.name != 'BehaviorCloning':
                        all_q_vals[~action_masks] = float('-inf')
                    eval_loss = self.algorithm.loss_fxn(all_q_vals, expert_actions)
                    predicted_actions = all_q_vals.argmax(dim=1)
                    accuracy = (predicted_actions == expert_actions).float().mean()
                    train_metrics['test_acc_history'].append(accuracy.cpu().detach().numpy())
                    print(f"Train step {i_step}: Accuracy = {accuracy:.3f}, Loss = {eval_loss.item():.4f}, Q-val={all_q_vals.mean().item():.3f}")
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        self.save(save_filepath)
        train_metrics_filepath = self.outdir + 'train_metrics.pkl'
        with open(train_metrics_filepath, 'wb') as handle:
            pickle.dump(train_metrics, handle)
    
    def evaluate(self, env, num_episodes):
        """Evaluates the agent in an environment for multiple episodes.

        Runs greedy (epsilon-free) policy evaluation using the current
        Q-network. Tracks rewards and observations per episode, as well as
        summary statistics.

        Args:
            env (gym.Env):
                Gymnasium environment with `.reset()` and `.step()` APIs.
                Must supply `info['action_mask']` if action masking is required.
            num_episodes (int):
                Number of rollout episodes to evaluate.

        Saves:
            `<outdir>/eval_metrics.pkl` containing:
                - `mean_reward`
                - `std_reward`
                - `min_reward`
                - `max_reward`
                - `episode_rewards` (list)
                - `observations` (dict of arrays per episode)
                - `rewards` (dict of reward arrays per episode)
        """
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
            print(f'eval_metrics.pkl saved in {self.outdir}')

    def act(self, obs, action_mask, epsilon):
        """Selects an action using the underlying algorithm.

        Args:
            obs (array-like):
                Current observation (normalized if applicable).
            action_mask (array-like | None):
                Boolean mask indicating which actions are legal.
            epsilon (float | None):
                Epsilon for epsilon-greedy exploration. If None, selects greedily.

        Returns:
            int: Selected action index.
        """
        return self.algorithm.select_action(obs, action_mask, epsilon)
    
    def save(self, filepath):
        """Saves algorithm parameters to a file.

        Args:
            filepath (str): Destination path for serialized model weights.
        """
        self.algorithm.save(filepath)
    
    def load(self, filepath):
        """Loads algorithm parameters from a file.

        Args:
            filepath (str): Path to previously saved model weights.
        """
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