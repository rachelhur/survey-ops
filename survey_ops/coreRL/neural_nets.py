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
    """Deep Q-Network mapping observations to action-values.
    """
    def __init__(self, observation_dim, action_dim, hidden_dim=128, activation=None):
        super(DQN, self).__init__()
        self.activation = F.relu if activation is None else activation 
        self.layer1 = nn.Linear(observation_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, actions=None):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)
    
class BinEmbedding(nn.Module):
    def __init__(self, num_bins, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, embedding_dim)
        
    def forward(self, bin_ids):
        return self.embedding(bin_ids)
    
class BinEmbeddingDQN(nn.Module):
    """Deep Q-Network mapping observations to action-values.
    """
    def __init__(self, n_local_features, n_global_features, n_bin_features, action_dim, hidden_dim=128, activation=None, embedding_dim=None):
        super(BinEmbeddingDQN, self).__init__()

        self.activation = F.relu if activation is None else activation

        self.bin_embedding = nn.Embedding(action_dim, embedding_dim)
        
        input_dim = (n_bin_features + embedding_dim) * action_dim + n_local_features + n_global_features

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, actions):
        local_features, global_features, bin_features, bin_indices = state

        bin_embeddings = self.bin_embedding(actions) # [batch, n_bins, emb_dim]
        bin_input = torch.cat([bin_features, bin_embeddings], dim=-1)  # [batch, n_bins, n_features + emb_dim]
        
        bin_flat = bin_input.flatten(start_dim=1)  # [batch, n_bins * (n_features + emb_dim)]
        full_input = torch.cat([bin_flat, local_features, global_features], dim=-1)
        
        return self.policy_net(full_input)

class SpatialEncoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
    def forward(self, x):
        # x shape: (Batch, Features, Lat, Lon)
        return self.cnn(x)