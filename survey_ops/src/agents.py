from random import random
import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import time
from typing import Tuple
import os
import pickle
import random

from survey_ops.utils.interpolate import interpolate_on_sphere
import logging

# Get the logger associated with this module's name (e.g., 'my_module')
logger = logging.getLogger(__name__)
from tqdm.contrib.logging import logging_redirect_tqdm

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
        
    def fit(self, num_epochs, dataset=None, batch_size=None, trainloader=None, valloader=None, patience=10, train_log_freq=10):
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

        Saves:
            - `<outdir>/weights.pt`: Final model weights.
            - `<outdir>/train_metrics.pkl`: Dictionary containing:
                - `loss_history`
                - `q_history`
                - `test_acc_history`
        """
        # assert dataset is not None and dataloader is not None
        if trainloader is not None:
            assert batch_size is not None
        
        train_metrics = {
            'train_loss': [],
            'train_qvals': [],
            'lr': []
        }

        val_metrics = {metric: [] for metric in self.algorithm.val_metrics}
        val_train_metrics = {metric: [] for metric in self.algorithm.val_metrics}

        save_filepath = self.train_outdir + 'best_weights.pt'
        train_metrics_filepath = self.train_outdir + 'train_metrics.pkl'
        val_metrics_filepath = self.train_outdir + 'val_metrics.pkl'
        val_train_metrics_filepath = self.train_outdir + 'val_train_metrics.pkl'
        self.algorithm.policy_net.train()

        if trainloader is not None:
            dataset_size = len(trainloader.dataset)
            steps_per_epoch = dataset_size // batch_size
            total_steps = int(num_epochs * steps_per_epoch) # ie, total number of times dataset is sampled
            loader_iter = iter(trainloader)  # create iterator
        else:
            # TODO for v0 only - remove when model is updated for release
            dataset_size = np.prod(dataset.obs.shape[1:])
            total_steps = int(num_epochs * dataset_size / batch_size)
            loader_iter = None  # not used for manual sampling

        best_val_loss = 1e5
        patience_cur = patience
        i_epoch = 0

        steps_per_epoch = len(trainloader.dataset) // batch_size
        total_steps = int(num_epochs * steps_per_epoch)

        # total_lr_scheduler_steps = int(args.lr_scheduler_max_epochs * iterations_per_epoch // args.lr_scheduler_step_freq)
        logger.info(f"Total number of training steps: {total_steps}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.debug(f"Total number of lr scheduler steps: {self.algorithm.lr_scheduler_num_epochs if self.algorithm.lr_scheduler is not None else None}")
        logger.debug(f"Number of transitions in dataset: {len(trainloader.dataset)}")

        with logging_redirect_tqdm():
            pbar = tqdm(total=total_steps, dynamic_ncols=True, desc="Starting training")
            for i_step in range(total_steps):
                if trainloader is not None:
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(trainloader)
                        batch = next(loader_iter)
                else:
                    batch = dataset.sample(batch_size)

                if i_step % steps_per_epoch == 0:
                    i_epoch += 1
                    
                pbar.update(1)

                pbar.set_description(f"Epoch {i_epoch}/{int(num_epochs)} (step {i_step}/{total_steps})")
                loss, q_val = self.algorithm.train_step(batch, epoch_num=i_epoch, step_num=i_step)
                if i_step % train_log_freq == 0:
                    train_metrics['train_loss'].append(loss)
                    train_metrics['train_qvals'].append(q_val)
                    train_metrics['lr'].append(self.algorithm.optimizer.param_groups[0]["lr"])
                    logger.debug(f"current LR is {train_metrics['lr'][-1]}")

                # At end of each epoch, do validation check
                with torch.no_grad():
                    if i_step % steps_per_epoch == 0:
                        if trainloader is not None:
                            # --- evaluation from dataloader ---
                            try:
                                eval_batch = next(eval_iter)
                            except (NameError, StopIteration):
                                eval_iter = iter(valloader)
                                eval_batch = next(eval_iter)
                        else:
                            # --- old method fallback ---
                            eval_obs, expert_actions, _, _, _, action_masks = dataset.sample(batch_size)

                        val_metric_vals = self.algorithm.test_step(eval_batch)
                        val_train_metric_vals = self.algorithm.test_step(batch)

                        for metric_name, metric_val in zip(val_metrics.keys(), val_metric_vals):
                            val_metrics[metric_name].append(metric_val)
                        for metric_name, metric_val in zip(val_train_metrics.keys(), val_train_metric_vals):
                            val_train_metrics[metric_name].append(metric_val)

                        logger.info(
                            f"Validation check at train step {i_step} \n " + \
                                " ".join(
                                    f"{k} = {v:.3f} | " for k, v in zip(val_metrics.keys(), val_metric_vals)
                                ) + "\n" + \
                                " ".join(
                                    f"{k + ' (train batch)'} = {v:.3f} | " for k, v in zip(val_train_metrics.keys(), val_train_metric_vals)
                                )
                        )

                        val_loss_cur = val_metrics['val_loss'][-1]

                    if val_loss_cur < best_val_loss and best_val_loss != val_loss_cur and i_step % steps_per_epoch ==0:
                        best_val_loss = val_loss_cur
                        patience_cur = patience
                        logger.info(f'Improved model at step {i_step}. New best val loss is {val_loss_cur:.3f} Saving weights.')
                        self.save(save_filepath)
                        with open(train_metrics_filepath, 'wb') as handle:
                            pickle.dump(train_metrics, handle)
                        with open(val_metrics_filepath, 'wb') as handle:
                            pickle.dump(val_metrics, handle)
                        with open(val_train_metrics_filepath, 'wb') as handle:
                            pickle.dump(val_train_metrics, handle)
                    else:
                        patience_cur -= 1
                        logger.debug(f"Patience left: {patience_cur}")
                        if patience_cur == 0:
                            logger.info("No patience left. Ending training.")
                            break

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

        # TODO: save these somewhere as some config instead of using env.unwrapped
        get_fields_in_bin = env.unwrapped.get_fields_in_bin
        field2nvisits, field2radec, field_ids, field_radecs = env.unwrapped.field2nvisits, env.unwrapped.field2radec, env.unwrapped.field_ids, env.unwrapped.field_radecs
        bin2fields_in_bin = env.unwrapped.bin2fields_in_bin
        hpGrid = env.unwrapped.test_dataset.hpGrid # change to nside and recreate healpix
       
        with logging_redirect_tqdm():
            for episode in tqdm(range(num_episodes)):
                obs, info = env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                num_nights = env.unwrapped.max_nights
                observations = {f'night-{i}': [] for i in range(num_nights)}
                rewards = {f'night-{i}': [] for i in range(num_nights)}
                timestamps = {f'night-{i}': [] for i in range(num_nights)}
                fields = {f'night-{i}': [] for i in range(num_nights)}
                bins = {f'night-{i}': [] for i in range(num_nights)}
                                
                i = 0
                reward = 0
                night_idx = 0

                pbar = tqdm(total=250*num_nights, dynamic_ncols=True, desc=f"Rolling out policy for night {night_idx} step {i}")
                while not (terminated or truncated):
                    with torch.no_grad():
                        timestamp = info.get('timestamp')
                        observations[f'night-{night_idx}'].append(obs)
                        rewards[f'night-{night_idx}'].append(reward)
                        timestamps[f'night-{night_idx}'].append(info.get('timestamp'))
                        fields[f'night-{night_idx}'].append(info.get('field_id'))
                        bins[f'night-{night_idx}'].append(info.get('bin'))

                        action_mask = info.get('action_mask', None)
                        action = self.act(obs, action_mask, epsilon=None)
                        fields_in_bin = get_fields_in_bin(bin_num=action, timestamp=timestamp, field2nvisits=field2nvisits, field_ids=field_ids, field_radecs=field_radecs, hpGrid=hpGrid, visited=info.get('visited'), bin2fields_in_bin=bin2fields_in_bin)
                        field_id = self.choose_field(obs=obs, info=info, field2nvisits=field2nvisits, field2radec=field2radec, hpGrid=hpGrid, field_choice_method=field_choice_method, fields_in_bin=fields_in_bin)

                        actions = np.array([action, field_id], dtype=np.int32)
                        obs, reward, terminated, truncated, info = env.step(actions)
                        episode_reward += reward
                        night_idx = info.get('night_idx')
                        i += 1
                        pbar.update(1)
                        pbar.set_description(f"Rolling out policy for night {night_idx} step {i}")
            pbar.close()
            for night_idx in range(num_nights):
                observations[f'night-{night_idx}'] = np.array(observations[f'night-{night_idx}'])
                rewards[f'night-{night_idx}'] = np.array(rewards[f'night-{night_idx}'])
                timestamps[f'night-{night_idx}'] = np.array(timestamps[f'night-{night_idx}'])
                fields[f'night-{night_idx}'] = np.array(fields[f'night-{night_idx}'])
                bins[f'night-{night_idx}'] = np.array(bins[f'night-{night_idx}'])
            eval_metrics.update({f'ep-{episode}': {
                'observations': observations,
                'rewards': rewards,
                'timestamp': timestamps,
                'field_id': fields,
                'bin': bins
            }})

            episode_rewards.append(episode_reward)
            logger.info(f'terminated at step {i}')

        eval_metrics.update({
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
        })

        with open(eval_outdir + 'eval_metrics.pkl', 'wb') as handle:
            pickle.dump(eval_metrics, handle)
            logger.info(f'eval_metrics.pkl saved in {eval_outdir}')

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

    def choose_field(self, obs, info, field2nvisits, field2radec, hpGrid, field_choice_method, fields_in_bin): 
        """
        Choose field in bin based on interpolated Q-values
        """
        assert len(fields_in_bin) != 0, "The agent is receiving an empty list for `fields_in_bin`. "
        visited = info.get('visited', None)
        action_mask = info.get('action_mask', None)
        field_ids_in_bin = [fid for fid in fields_in_bin if visited.count(fid) < field2nvisits[fid]]

        if field_choice_method == 'interp':
            with torch.no_grad():
                obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                mask = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool)
                q_vals = self.algorithm.policy_net(obs).squeeze(0)
                q_vals = q_vals.cpu().detach().numpy()

            lon_data = hpGrid.lon
            lat_data = hpGrid.lat

            target_lonlats = np.array([field2radec[fid] for fid in field_ids_in_bin])
            
            q_interpolated = interpolate_on_sphere(target_lonlats[:, 0], target_lonlats[:, 1], lon_data, lat_data, q_vals)
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
