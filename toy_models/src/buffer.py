from collections import namedtuple, deque
import numpy as np


# Experience stores experience steps gathered in training
# essentially maps (current state, action) to (next state, reward)
Experience = namedtuple(
    "Experience",
    field_names=["obs", "action", "reward", "next_obs", "done", "action_mask", "next_action_mask"],
)

# Stores experiences
class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer) 
    
    def append(self, *args):
        """Save a transition"""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        batch = Experience(*zip(*batch))

        return (
            np.array(batch.obs, dtype=np.float32),
            np.array(batch.action, dtype=np.float32),
            np.array(batch.reward, dtype=np.float32),
            np.array(batch.next_obs, dtype=np.float32),
            np.array(batch.done, dtype=np.bool_),
            np.array(batch.action_mask, dtype=bool),
            np.array(batch.next_action_mask, dtype=bool)
        )
    
    def reset(self):
        self.buffer.clear()

