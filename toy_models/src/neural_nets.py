import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F


def linear_schedule(eps_start: float, eps_end: float, duration: int, t: int):
    """
    Decreases epsilon linearly as a function of time step.

    Args
    ----
    eps_start: float
        starting epsilon value
    eps_end: float
        final epsilon value
    duration: int
        Number of time steps it takes to move from eps_start to eps_end
    t: ind
        Current time step
    
    Returns
    ------
    epsilon: float
        Current epsilon value
        
    """
    try:
        slope = (eps_end - eps_start) / duration
        return max(slope * t + eps_start, eps_end)
    except:
        raise Exception("Error in linear_schedule")

def exponential_schedule(eps_start: float, eps_end: float, decay_rate: float, t: int):
    try:
        return eps_end + (eps_start - eps_end) * np.exp(-1. * t / decay_rate)
    except:
        raise Exception("Error in exponential_schedule")

class DQN(nn.Module):
    '''Sends observations to action-value space'''
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
