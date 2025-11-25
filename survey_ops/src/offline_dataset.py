from collections.abc import Callable
import numpy as np
import sys
sys.path.append('../survey_ops/utils')
import ephemerides

def reward_func_v0():
    raise NotImplementedError

class TelescopeDatasetv0:
    """
    First version of dataset class for telescope data. Designed for behavior cloning where the original
     schedule is mimicked exactly (no generalization). 

    Args
    ---- 
        schedule (pd.Dataframe): Pandas dataframe with columns 'field_id' and 'next_field_id'
        id2pos (dict): A dictionary mapping field_ids (keys) to positions (vals)
        n_nights
        normalize_obs (bool): Whether or not to normalize observations
        reward_func (Callable): Vectorized reward calculation
        device (str): cuda or CPU
    
    Attributes
    ----------
        obs (ndarray): The observations of shape (obs_dim, n_nights, n_transitions) and dtype float32
            to be input into the q-network.
        actions (ndarray): The array of field ids of shape (n_nights, n_transitions) which indicate the
            next field id to be observed
        next_obs (ndarray): The next observations given obs and action; shape (obs_dim, n_nights, n_transitions)
        rewards (ndarray): The reward of doing next_obs given obs and action; shape (n_nights, n_transitions)
        dones (ndarray): Boolean array indicating whether or not this action terminates the episode; 
            shape (n_nights, n_transitions)
        action_masks (ndarray): Boolean array indicating valid possible actions
        norm: if 
        _obs_indices (ndarray): A time index for each observation made in a single night (starts at 0)
        _schedule_field_id (ndarray): The actual schedule of shape (n_nights, n_obs_in_night) where
            n_obs_in_night = n_transitions + 1

    """
    def __init__(self, schedule, id2pos, reward_func=None, normalize_obs=True, device='cpu'):
        
        self._schedule_field_ids = schedule.field_id.values.astype(np.float32)
        if len(self._schedule_field_ids.shape) == 1:
            self._schedule_field_ids = self._schedule_field_ids[np.newaxis, :]
        assert len(self._schedule_field_ids.shape) == 2
        # self._schedule_field_ids = schedule.field_id.values[np.newaxis, :].astype(np.float32)
        self._radec = np.array([id2pos[field_id] for field_id in schedule.field_id.values], dtype=np.float32)
        self._unique_field_ids, counts = np.unique(self._schedule_field_ids, return_counts=True)
        self._max_visits = np.int32(np.max(counts))
        self._nfields = np.int32(len(self._unique_field_ids))
        self._n_obs_per_night = np.int32(self._schedule_field_ids.shape[-1])
        self._n_transitions = np.int32(len(self._radec))
        self._id2pos = id2pos
        self.device = device
        self.normalize_obs = normalize_obs
        self.norm = 1
        if self.normalize_obs:
            self.norm = np.array([self._n_obs_per_night, self._nfields])[:, np.newaxis]
        self.reward_func = reward_func

        self._obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, :-1]
        self._field_ids = schedule.field_id.values[:-1][np.newaxis, np.newaxis, :]
        self._next_obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, 1:]
        self._next_field_ids = schedule.field_id.values[1:][np.newaxis, np.newaxis, :]

        # normalize inputs to network
        self.obs = np.concatenate((self._field_ids, self._obs_indices), axis=0)
        self.next_obs = np.concatenate((self._next_field_ids, self._next_obs_indices), axis=0)

        self.obs_dim = self.obs.shape[0]
        self._n_nights = np.int32(self.obs.shape[1])
        self.num_actions = self._nfields 

        self.actions = schedule.field_id.values[1:][np.newaxis, :]
        self.rewards = self._get_rewards()
        self.dones = np.zeros_like(self.obs[0], dtype=np.bool_)
        self.dones[:, -1] = True
        self.action_masks = self._get_action_masks()


        # field_az, field_el = ephemerides.equatorial_to_topographic(field_ra, field_dec, time=timestamp)
        # sun_ra, sun_dec = ephemerides.get_source_ra_dec("sun", time=timestamp)
        # sun_az, sun_el = ephemerides.equatorial_to_topographic(sun_ra, sun_dec, time=timestamp)
        # moon_ra, moon_dec = ephemerides.get_source_ra_dec("moon", time=timestamp)
        # moon_az, moon_el = ephemerides.equatorial_to_topographic(moon_ra, moon_dec, time=timestamp)

    def _get_rewards(self):
        if self.reward_func is None:
            return np.ones_like(self.obs[0])
        return self._reward_func(self.obs, self.actions, self.next_obs)

    
    def _get_action_masks(self):
        nvisits_base = np.zeros(shape=(self._nfields), dtype=np.int32)
        full_nvisits = np.zeros(shape=(self._n_nights, self._n_obs_per_night, self._nfields), dtype=np.int32)

        for i, night_ids in enumerate(self._schedule_field_ids):
            for j, field_id in enumerate(night_ids):
                nvisits_base[np.int32(field_id)] += 1
                full_nvisits[i, j] = nvisits_base.copy()
        action_masks = full_nvisits != self._max_visits
        return action_masks

    def __len__(self):
        return self._n_nights

    def sample(self, batch_size):
        """
        Samples dataset for a random night, random observation in that night, and (optionally) normalizes observation
        """
        night_indices = np.random.choice(self._n_nights, batch_size, replace=True)
        obs_indices = np.random.choice(self._n_obs_per_night - 1, batch_size)
        return (
            np.array(self.obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T, # needs to be float for network
            np.array(self.actions[night_indices, obs_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, obs_indices], dtype=np.float32),
            np.array(self.next_obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T,
            np.array(self.dones[night_indices, obs_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, obs_indices], dtype=bool),
        )
    


class TelescopeDatasetv1:
    """
    First version of dataset class for telescope data. Designed for behavior cloning where the original
     schedule is mimicked exactly (no generalization). 

    Args
    ---- 
        schedule (pd.Dataframe): Pandas dataframe with columns 'field_id' and 'next_field_id'
        id2pos (dict): A dictionary mapping field_ids (keys) to positions (vals)
        n_nights
        normalize_obs (bool): Whether or not to normalize observations
        reward_func (Callable): Vectorized reward calculation
        device (str): cuda or CPU
    
    Attributes
    ----------
        obs (ndarray): The observations of shape (obs_dim, n_nights, n_transitions) and dtype float32
            to be input into the q-network.
        actions (ndarray): The array of field ids of shape (n_nights, n_transitions) which indicate the
            next field id to be observed
        next_obs (ndarray): The next observations given obs and action; shape (obs_dim, n_nights, n_transitions)
        rewards (ndarray): The reward of doing next_obs given obs and action; shape (n_nights, n_transitions)
        dones (ndarray): Boolean array indicating whether or not this action terminates the episode; 
            shape (n_nights, n_transitions)
        action_masks (ndarray): Boolean array indicating valid possible actions
        norm: if 
        _obs_indices (ndarray): A time index for each observation made in a single night (starts at 0)
        _schedule_field_id (ndarray): The actual schedule of shape (n_nights, n_obs_in_night) where
            n_obs_in_night = n_transitions + 1

    """
    def __init__(self, schedule, id2pos, reward_func=None, normalize_obs=True, device='cpu'):
        id2pos.update({-1: np.array([-1, -1])})
        self._schedule_field_ids = schedule.field_id.values.astype(np.float32)
        if len(self._schedule_field_ids.shape) == 1:
            self._schedule_field_ids = self._schedule_field_ids[np.newaxis, :]
        assert len(self._schedule_field_ids.shape) == 2
        if self._schedule_field_ids[0, 0] != -1:
            self._schedule_field_ids = np.concatenate(
                (-np.ones(self._schedule_field_ids.shape[:-1])[:, np.newaxis], self._schedule_field_ids),
                axis=-1
            )
        # self._schedule_field_ids = schedule.field_id.values[np.newaxis, :].astype(np.float32)
        self._radec = np.array([id2pos[field_id] for field_id in self._schedule_field_ids[0]], dtype=np.float32)
        self._unique_field_ids, counts = np.unique(self._schedule_field_ids[self._schedule_field_ids > -1], return_counts=True)
        self._max_visits = np.int32(np.max(counts))
        self._nfields = np.int32(len(self._unique_field_ids)) # subtract 1 for non-state
        self._n_obs_per_night = np.int32(self._schedule_field_ids.shape[-1] - 1)
        self._n_transitions = np.int32(len(self._radec))
        self._id2pos = id2pos
        self.device = device
        self.normalize_obs = normalize_obs
        self.norm = 1
        self.min = 0
        if self.normalize_obs:
            self.norm = np.array([self._n_obs_per_night, self._nfields])[:, np.newaxis]
            self.min = -1
        self.reward_func = reward_func

        obs_indices = np.arange(-1, self._n_obs_per_night)
        self._obs_indices = obs_indices[np.newaxis, np.newaxis, :-1]
        self._field_ids = self._schedule_field_ids[:, :-1][np.newaxis, :]
        self._next_obs_indices = obs_indices[np.newaxis, np.newaxis, 1:]
        self._next_field_ids = self._schedule_field_ids[:, 1:][np.newaxis, :]

        # normalize inputs to dqn network
        self.obs = np.concatenate((self._field_ids, self._obs_indices), axis=0)
        self.next_obs = np.concatenate((self._next_field_ids, self._next_obs_indices), axis=0)

        self.obs_dim = self.obs.shape[0]
        self._n_nights = np.int32(self.obs.shape[1])
        self.num_actions = self.actions.shape[-1]

        self.actions = np.arange(self._nfields)[np.newaxis, :]
        self.rewards = self._get_rewards()
        self.dones = np.zeros_like(self.obs[0], dtype=np.bool_)
        self.dones[:, -1] = True
        self.action_masks = self._get_action_masks()


        # field_az, field_el = ephemerides.equatorial_to_topographic(field_ra, field_dec, time=timestamp)
        # sun_ra, sun_dec = ephemerides.get_source_ra_dec("sun", time=timestamp)
        # sun_az, sun_el = ephemerides.equatorial_to_topographic(sun_ra, sun_dec, time=timestamp)
        # moon_ra, moon_dec = ephemerides.get_source_ra_dec("moon", time=timestamp)
        # moon_az, moon_el = ephemerides.equatorial_to_topographic(moon_ra, moon_dec, time=timestamp)

    def _get_rewards(self):
        if self.reward_func is None:
            return np.ones_like(self.obs[0])
        return self._reward_func(self.obs, self.actions, self.next_obs)

    
    def _get_action_masks(self):
        nvisits_base = np.zeros(shape=(self._nfields), dtype=np.int32)
        full_nvisits = np.zeros(shape=(self._n_nights, self._n_obs_per_night + 1, self._nfields), dtype=np.int32)

        for i, night_ids in enumerate(self._schedule_field_ids):
            for j, field_id in enumerate(night_ids):
                if field_id != -1:
                    nvisits_base[np.int32(field_id)] += 1
                full_nvisits[i, j] = nvisits_base.copy()
        action_masks = full_nvisits != self._max_visits
        return action_masks

    def __len__(self):
        return self._n_nights

    def sample(self, batch_size):
        """
        Samples dataset for a random night, random observation in that night, and (optionally) normalizes observation
        """
        night_indices = np.random.choice(self._n_nights, batch_size, replace=True)
        obs_indices = np.random.choice(self._n_obs_per_night, batch_size)
        return (
            np.array((self.obs[:, night_indices, obs_indices] - self.min)/self.norm, dtype=np.float32).T, # needs to be float for network
            np.array(self.actions[night_indices, obs_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, obs_indices], dtype=np.float32),
            np.array((self.next_obs[:, night_indices, obs_indices] - self.min)/self.norm, dtype=np.float32).T,
            np.array(self.dones[night_indices, obs_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, obs_indices], dtype=bool),
        )