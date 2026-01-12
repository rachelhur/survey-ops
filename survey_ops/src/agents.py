from random import random
import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
import os
import pickle
import random

from survey_ops.utils.interpolate import interpolate_on_sphere

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
            train_outdir,
            # env: gym.Env = None,
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
        if not os.path.exists(train_outdir):
            os.makedirs(train_outdir)
        self.train_outdir = train_outdir
        
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

        save_filepath = self.train_outdir + 'best_weights.pt'
        train_metrics_filepath = self.train_outdir + 'train_metrics.pkl'

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
                    eval_obs = torch.as_tensor(eval_obs, device=self.device, dtype=torch.float32)
                    expert_actions = torch.as_tensor(expert_actions, device=self.device, dtype=torch.long)
                    
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
                        with open(train_metrics_filepath, 'wb') as handle:
                            pickle.dump(train_metrics, handle)
        with open(train_metrics_filepath, 'wb') as handle:
            pickle.dump(train_metrics, handle)
    
    def evaluate(self, env, num_episodes, field_choice_method='interp', eval_outdir=None):
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
        eval_outdir = eval_outdir if eval_outdir is not None else self.train_outdir + 'evaluation/'
        if not os.path.exists(eval_outdir):
            os.makedirs(eval_outdir)
            
        # evaluation metrics
        self.algorithm.policy_net.eval()
        episode_rewards = []
        eval_metrics = {}
        field2nvisits, bin2fields_in_bin, field2radec = env.unwrapped.field2nvisits, env.unwrapped.bin2fields_in_bin, env.unwrapped.field2radec
        hpGrid = env.unwrapped.test_dataset.hpGrid
        
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            num_nights = env.unwrapped.max_nights
            observations = {f'night-{i}': [] for i in range(num_nights)}
            rewards = {f'night-{i}': [] for i in range(num_nights)}
            timestamps = {f'night-{i}': [] for i in range(num_nights)}
            field_ids = {f'night-{i}': [] for i in range(num_nights)}
            bin_nums = {f'night-{i}': [] for i in range(num_nights)}

            i = 0
            reward = 0
            night_idx = 0
            while not (terminated or truncated):
                with torch.no_grad():
                    observations[f'night-{night_idx}'].append(obs)
                    rewards[f'night-{night_idx}'].append(reward)
                    timestamps[f'night-{night_idx}'].append(info.get('timestamp'))
                    field_ids[f'night-{night_idx}'].append(info.get('field_id'))
                    bin_nums[f'night-{night_idx}'].append(info.get('bin'))

                    action_mask = info.get('action_mask', None)
                    action = self.act(obs, action_mask, epsilon=None)
                    field_id = self.choose_field(obs, action, info, field2nvisits, field2radec, bin2fields_in_bin, hpGrid, field_choice_method)

                    actions = np.array([action, field_id], dtype=np.int32)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    episode_reward += reward
                    night_idx = info.get('night_idx')
                    i += 1
            for night_idx in range(num_nights):
                observations[f'night-{night_idx}'] = np.array(observations[f'night-{night_idx}'])
                rewards[f'night-{night_idx}'] = np.array(rewards[f'night-{night_idx}'])
                timestamps[f'night-{night_idx}'] = np.array(timestamps[f'night-{night_idx}'])
                field_ids[f'night-{night_idx}'] = np.array(field_ids[f'night-{night_idx}'])
                bin_nums[f'night-{night_idx}'] = np.array(bin_nums[f'night-{night_idx}'])
            eval_metrics.update({f'ep-{episode}': {
                'observations': observations,
                'rewards': rewards,
                'timestamp': timestamps,
                'field_id': field_ids,
                'bin': bin_nums
            }})

            episode_rewards.append(episode_reward)
            print(f'terminated at {i}')

        eval_metrics.update({
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
        })

        with open(eval_outdir + 'eval_metrics.pkl', 'wb') as handle:
            pickle.dump(eval_metrics, handle)
            print(f'eval_metrics.pkl saved in {eval_outdir}')

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

    def choose_field(self, obs, bin_num, info, field2nvisits, field2radec, bin2fields_in_bin, hpGrid, field_choice_method): # az, el, az_data, el_data, values, field_ids_in_bin, bin_num
        """
        Choose field in bin based on interpolated Q-values
        """
        visited = info.get('visited', None)
        action_mask = info.get('action_mask', None)
        field_ids_in_bin = bin2fields_in_bin[bin_num]
        field_ids_in_bin = [fid for fid in field_ids_in_bin if visited.count(fid) < field2nvisits[fid]]

        if bin_num not in bin2fields_in_bin:
            return None, None
        
        if field_choice_method == 'interp':
            with torch.no_grad():
                obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                mask = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool)
                # obs = obs.to(self.device, dtype=torch.float32).unsqueeze(0)
                # mask = action_mask.to(self.device, dtype=torch.bool).unsqueeze(0)
                q_vals = self.algorithm.policy_net(obs).squeeze(0)
                q_vals = q_vals.cpu().detach().numpy()

            lon_data = hpGrid.lon
            lat_data = hpGrid.lat

            target_lonlats = np.array([field2radec[fid] for fid in field_ids_in_bin])
            
            q_interpolated = interpolate_on_sphere(target_lonlats[:, 0], target_lonlats[:, 1], lon_data, lat_data, q_vals)
            print(len(q_interpolated), len(field_ids_in_bin))
            if self.algorithm.name == 'BehaviorCloning':
                best_idx = np.argmin(q_interpolated)
            else:
                best_idx = np.argmax(q_interpolated)
            best_field = field_ids_in_bin[best_idx]

            return best_field
        elif field_choice_method == 'random':
            field_id = random.choice(field_ids_in_bin)
            return field_id
        

    #         lon_data = hpGrid.lon
            # lat_data = hpGrid.lat
            # q_interpolated = interpolate_on_sphere(az, el, lon_data, lat_data, q_vals)
            
            # with torch.no_grad():
            #     obs = obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            #     mask = mask.to(self.device, dtype=torch.bool).unsqueeze(0)
            #     action_logits = self.policy_net(obs)
            #     # mask invalid actions
            #     action_logits[~mask] = float('-inf')
            #     action = torch.argmax(action_logits, dim=1)
