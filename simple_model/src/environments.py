from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
from typing import Optional
import numpy as np


from typing import Optional
import numpy as np


class SimpleTelEnv(gym.Env):
    def __init__(self, Nf, target_sequence, nv_max, off_by_lim):
        self.Nf = Nf # number of fields
        # self.T = 1 # 28800sec = 8hrs
        self.nv_max = nv_max
        self.target_sequence = target_sequence
        self.off_by_lim = off_by_lim
        # "Teff_meas": Box(0, 1, shape=(Nf,), dtype=np.float32),

        #TODO
        # Initialize positions - will be set in reset()
        self._field_id = -1
        # self._t = -1
        self._nvisits = np.full(shape=(Nf,), fill_value=-1, dtype=np.int32)
        # self._Teff_pred = np.full(shape=(Nf,), fill_value=-1, dtype=np.float32)
        self._index = -1
        self._sequence = []

        # self._possible_actions = [i for i in range(Nf)]

        #TODO
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                # "t": Box(0, T, shape=None, dtype=np.float32),
                "field_id": Discrete(n=Nf, start=0),
                "nvisits": Box(0, 4, shape=(Nf,), dtype=np.int32),
                "index": Discrete(n=len(self.target_sequence), start=0)
                # "Teff_pred": Box(0, 1, shape=(Nf,), dtype=np.float32),
                    #filter
            }
        )
        
        # # Map action numbers to field
        self._action_to_field_id = {i:i for i in range(Nf)}

        self.action_space = gym.spaces.Discrete(self.Nf)

        # remove fields that have 
        
    def _get_obs(self):
        """Convert internal state to observation format.
    
        Returns:
            dict: Observation with agent and target positions
        """
        return {
            # "t": self._t,
            "field_id": self._field_id,
            "nvisits": self._nvisits,
            "index": self._index
            # "Teff_pred": self._Teff_pred,
        }

    def _get_info(self, chosen_field_id=None, correct=None):
        """Compute auxiliary information for debugging.

        Returns:
            
        """
        return {'chosen_field_id': chosen_field_id, 'correct': correct}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # initialize number of visits
        self._nvisits = np.full(shape=(self.Nf,), fill_value=0, dtype=np.int32)
        
        # Randomly choose initial field id and add 1 visit to nvisits list
        # self._field_id = int(self.np_random.integers(0, self.Nf, size=1, dtype=int)[0])
        self._field_id = self.target_sequence[0]
        # self._field_id = self.np_random.integers(0, self.Nf, size=1, dtype=int).tolist()[0]
        self._nvisits[self._field_id] += 1
        self._index = 0
        # self._t = np.array([0.0], dtype=np.float32)
        # self._Teff_pred = np.linspace(0.1, .98, num=self.Nf, dtype=np.float32)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:

        Returns:
        """
        # # choose random field for next observation
        # list_idx = self.np_random.integers(low=0, high=len(self._possible_actions), dtype=int)
        # proposed_field = self._possible_actions[list_idx]
        # self._nvisits[proposed_field] += 1
        # self._field_id = proposed_field
        self._index += 1            
        # get current field_id from action
        self._field_id = self._action_to_field_id[action]
        # add to nvisits
        self._nvisits[self._field_id] += 1

        # Simple reward structure: +1 for reaching target, 0 otherwise
        target_field = self.target_sequence[self._index]
        correct = self._field_id == target_field
        off_by_val = np.abs(self._field_id - target_field) <= 3
        if correct:
            reward = 1
        elif off_by_val:
            reward = .1
        else:
            reward = 0
            
        survey_complete = (self._index == len(self.target_sequence)-1)
        
        # end condition
        terminated = survey_complete
        truncated = False

        # get obs and info
        observation = self._get_obs()
        info = self._get_info(self._field_id, correct)

        return observation, reward, terminated, truncated, info



