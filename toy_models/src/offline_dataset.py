from collections.abc import Callable
import numpy as np

class ToyOfflineDatasetv2:
    """
    Temporary class. Compatible specifically with data generator function defined in DQN_closest_distance_path.ipynb
    """

    def __init__(self, data_generator: Callable, train_size: int, **data_gen_args):
        self.data_generator = data_generator
        self.train_size = train_size
        self.obs, self.actions, self.rewards, self.next_obs, self.dones, self.action_masks, self._nfields, self._max_visits, self._full_separations, self._field_coord_mapping\
            = data_generator(train_size, **data_gen_args)
        
        self._obs_per_night = len(self.obs[0])
        self._obs_dim = 3 #(1 (coords) + 1 (field_id) )
        self._action_dim = self._nfields #doesn't include zenith
        # self.rewards = (self.rewards - np.mean(self.rewards)) / (np.std(self.rewards) + 1e-8)

    def __len__(self):
        return self.train_size

    def sample(self, batch_size):
        #TODO possible to have repeat night/step combos bc replace = True
        night_indices = np.random.choice(self.train_size, batch_size, replace=True)
        step_num_indices = np.random.choice(self._obs_per_night - 1, batch_size)

        # field_ids = self.obs[0][night_indices, step_num_indices]
        # coords = self.obs[1][night_indices, step_num_indices]
        # obs = np.concatenate((field_ids[:, np.newaxis], coords), axis=1)

        # next_field_ids = self.obs[0][night_indices, step_num_indices+1]
        # next_coords = self.obs[1][night_indices, step_num_indices+1]
        # next_obs = np.concatenate((next_field_ids[:, np.newaxis], next_coords), axis=1)
        
        return (
            np.array(self.obs[night_indices, step_num_indices], dtype=np.float32), # needs to be float for network
            np.array(self.actions[night_indices, step_num_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, step_num_indices], dtype=np.float32),
            np.array(self.next_obs[night_indices, step_num_indices], dtype=np.float32),
            np.array(self.dones[night_indices, step_num_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, step_num_indices], dtype=bool),
        )