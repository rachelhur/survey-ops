from collections.abc import Callable
import numpy as np
import sys
sys.path.append('../survey_ops/utils')
import ephemerides

class TelescopeDatasetv0:
    """
    First version of dataset for telescope data.

    Args
    ---- 
        schedule (pd.Dataframe): Pandas dataframe with columns 'field_id' and 'next_field_id'
    
    Attributes
    ----------
        _schedule_field_ids ()
    """
    def __init__(self, schedule, id2pos, n_nights=1, device=None, reward_config=0, normalize_obs=True):
        self._schedule_field_ids = schedule.field_id.values[np.newaxis, :].astype(np.float32)
        self._radec = np.array([id2pos[field_id] for field_id in schedule.field_id.values], dtype=np.float32)
        self._unique_field_ids, counts = np.unique(self._schedule_field_ids, return_counts=True)
        self._max_visits = np.int32(np.max(counts))
        self._nfields = np.int32(len(self._unique_field_ids))
        self._n_nights = np.int32(n_nights)
        self._n_obs_per_night = np.int32(len(self._schedule_field_ids[0]))
        self._n_transitions = np.int32(len(self._radec))
        self._id2pos = id2pos
        self.device = device
        self.reward_config = reward_config
        self.normalize_obs = normalize_obs

        self._obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, :-1]
        self._field_ids = schedule.field_id.values[:-1][np.newaxis, np.newaxis, :]
        self._next_obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, 1:]
        self._next_field_ids = schedule.field_id.values[1:][np.newaxis, np.newaxis, :]

        # normalize inputs to dqn network
        if normalize_obs:
            obs_indices_normed = self._obs_indices / self._n_obs_per_night
            next_obs_indices_normed = self._next_obs_indices / self._n_obs_per_night
            field_ids_normed = self._field_ids / self._nfields
            next_field_ids_normed = self._next_field_ids / self._nfields

            self.obs = np.concatenate((field_ids_normed, obs_indices_normed), axis=0)
            self.next_obs = np.concatenate((next_field_ids_normed, next_obs_indices_normed), axis=0)
        else:
            self.obs = np.concatenate((self._field_ids, self._obs_indices), axis=0)
            self.next_obs = np.concatenate((self._next_field_ids, self._next_obs_indices), axis=0)

        self.actions = schedule.field_id.values[1:][np.newaxis, :]
        self.rewards = self._get_rewards()
        self.dones = np.zeros_like(self.obs[0], dtype=np.bool_)
        self.dones[:, -1] = False
        self.action_masks = self._get_action_masks()

        self.obs_dim = len(self.obs)
        self.action_dim = self._nfields 
        # field_az, field_el = ephemerides.equatorial_to_topographic(field_ra, field_dec, time=timestamp)
        # sun_ra, sun_dec = ephemerides.get_source_ra_dec("sun", time=timestamp)
        # sun_az, sun_el = ephemerides.equatorial_to_topographic(sun_ra, sun_dec, time=timestamp)
        # moon_ra, moon_dec = ephemerides.get_source_ra_dec("moon", time=timestamp)
        # moon_az, moon_el = ephemerides.equatorial_to_topographic(moon_ra, moon_dec, time=timestamp)

    def _get_rewards(self):
        if self.reward_config == 0:
            # ones for every
            return np.ones_like(self.obs[0], dtype=np.float32) / 5
        # elif config == 1:
        #     return self._reward_func_1()
        
    # def _reward_func_1(self):
    #     rewards = np.ones_like(self.next_obs, dtype=np.float32) - .3 # TODO
    #     if self.get_distance(self.radec[:, 0]) <= 2:
    #         rewards[] 

    @staticmethod
    def _get_euclidean_dist():
        # np.linalg.norm(x1, x2)
        raise NotImplementedError
    
    def _get_action_masks(self,):
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
        night_indices = np.random.choice(self._n_nights, batch_size, replace=True)
        obs_indices = np.random.choice(self._n_obs_per_night - 1, batch_size)

        return (
            np.array(self.obs[:, night_indices, obs_indices], dtype=np.float32).T, # needs to be float for network
            np.array(self.actions[night_indices, obs_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, obs_indices], dtype=np.float32),
            np.array(self.next_obs[:, night_indices, obs_indices], dtype=np.float32).T,
            np.array(self.dones[night_indices, obs_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, obs_indices], dtype=bool),
        )