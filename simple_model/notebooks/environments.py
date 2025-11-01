from gymnasium import gym
from gymnasium.spaces import Dict, Box, Discrete

from typing import Optional
import numpy as np


class SimpleTelEnv(gym.Env):
    def __init__(self, Nf: int = 1000, T=28800, sequence_true=None):
        self.Nf = Nf # number of fields
        self.T = T # 8hrs in sec
        self.sequence_true = sequence_true if sequence_true is not None else np.arange(0, Nf, 1, dtype=int)

        #TODO
        # Initialize positions - will be set in reset()
        self._field_id = -1
        self._t = -1

        #TODO
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "Teff_pred": Box(0, 1, shape=(Nf,), dtype=np.float32),
                "Teff_meas": Box(0, 1, shape=(Nf,), dtype=np.float32),
                "field_id": Discrete(0, Nf, shape=(1,), dtype=int),
                "nvisits": Discrete(0, 4, shape=(Nf,), dtype=int),
                    #filter
                
            }
        )

        self.action_space = gym.spaces.Discrete(self.Nf)

        """Convert internal state to observation format.
    
        Returns:
            dict: Observation with agent and target positions
        """
        return {"field_id": self._}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

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

        # Randomly choose initial field id and set time to 0
        self._field_id = self.np_random.integers(0, self.Nf, dtype=int)
        self._t = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info



