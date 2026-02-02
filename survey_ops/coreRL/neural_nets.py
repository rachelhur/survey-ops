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
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)

class BinEmbeddingDQN(nn.Module):
    def __init__(self, num_bins, pointing_dim, dynamic_bin_feature_dim, hidden_dim, embedding_dim, activation=None):
        super().__init__()
        self.activation = F.relu if activation is None else activation 
        
        # Static bin positions — learned once, not fed through input
        self.bin_embedding = nn.Embedding(num_bins, embedding_dim)
        self.dqn = DQN(observation_dim=pointing_dim + embedding_dim + dynamic_bin_feature_dim, action_dim=1, hidden_dim=hidden_dim)
                
    # Single MLP scorer: takes (pointing_features, bin_embedding, dynamic_bin_features) → score
        
    def forward(self, pointing_features, dynamic_bin_features):
        # pointing_features: (batch, pointing_dim)
        # dynamic_bin_features: (batch, num_bins, dynamic_bin_feature_dim)
        
        batch_size = pointing_features.shape[0]
        num_bins = self.bin_embedding.num_embeddings
        
        # Get bin embeddings: (batch, num_bins, embedding_dim)
        bin_ids = torch.arange(num_bins, device=pointing_features.device)
        bin_emb = self.bin_embedding(bin_ids).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Broadcast pointing to all bins: (batch, num_bins, pointing_dim)
        pointing_expanded = pointing_features.unsqueeze(1).expand(-1, num_bins, -1)
        
        # Concatenate everything: (batch, num_bins, pointing_dim + embedding_dim + dynamic_bin_feature_dim)
        combined = torch.cat([pointing_expanded, bin_emb, dynamic_bin_features], dim=-1)
        
        # Score each bin: (batch, num_bins)
        scores = self.scorer(combined).squeeze(-1)
        
        return scores