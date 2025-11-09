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
        self.buffer = deque([], maxlen=capacity)
        self.full = False
        self.device = device
        self.pos = 0

    def __len__(self):
        return len(self.buffer) 
    
    def append(self, *args):
        """Save a transition"""
        self.buffer.append(Experience(*args))
        self.pos += 1
        if self.pos == self.buffer.maxlen:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones, action_masks, next_action_masks = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=bool),
            np.array(action_masks),
            np.array(next_action_masks)
        )
    
    def reset(self):
        # self.buffer.clear()
        self.full = False
        self.pos = 0

