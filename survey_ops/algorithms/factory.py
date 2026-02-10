import logging
logger = logging.getLogger(__name__)


import torch.nn as nn
import torch.nn.functional as F

from survey_ops.algorithms.algorithms import DDQN, BehaviorCloning


def setup_algorithm(algorithm_name=None, obs_dim=None, num_actions=None, loss_fxn=None, hidden_dim=None, lr=None, lr_scheduler=None, device=None, 
                    lr_scheduler_kwargs=None, gamma=None, tau=None, lr_scheduler_epoch_start=None, lr_scheduler_num_epochs=None, activation=None, 
                    grid_network=None, n_local_features=None, n_global_features=None, n_bin_features=None, embedding_dim=None):
    assert loss_fxn is not None
    if activation == 'relu':
        activation = F.relu
    elif activation == 'mish':
        activation = F.mish
    elif activation == 'swish':
        activation = F.silu
        
    # Set up model hyperparameters that are algorithm independent
    model_hyperparams = {
        'obs_dim': obs_dim,
        'num_actions': num_actions, 
        'hidden_dim': hidden_dim,
        'lr': lr,
        'lr_scheduler': lr_scheduler,
        'lr_scheduler_kwargs': lr_scheduler_kwargs,
        'lr_scheduler_epoch_start': lr_scheduler_epoch_start,
        'lr_scheduler_num_epochs': lr_scheduler_num_epochs,
        'loss_fxn': loss_fxn,
        'activation': activation
    }
        
    if algorithm_name == 'DDQN' or algorithm_name == 'DQN':
        assert gamma is not None, "Gamma (discount factor) must be specified for DDQN."
        assert tau is not None, "Tau (target network update rate) must be specified for DDQN."
        # assert loss_fxn in ['mse', 'huber'], "DDQN only supports mse or huber loss functions."
        
        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn
        elif loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        elif loss_fxn == 'huber':
            loss_fxn = nn.HuberLoss(reduction='mean')
        else:
            print(loss_fxn)
            raise NotImplementedError(f'Loss function {loss_fxn} not yet implemented for {algorithm_name}')

        model_hyperparams .update( {
            'gamma': gamma,
            'tau': tau,
            'use_double': algorithm_name == 'ddqn',
            'loss_fxn': loss_fxn
        } )

        algorithm = DDQN(
            device=device,
            **model_hyperparams
        )

    elif algorithm_name == 'BC':
        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn
        elif loss_fxn == 'cross_entropy':
            loss_fxn = nn.CrossEntropyLoss(reduction='mean')
        elif loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        else:
            print(loss_fxn)
            raise NotImplementedError(f'Loss function {loss_fxn} not yet implemented for {algorithm_name}')
        
        model_hyperparams.update({
        'loss_fxn': loss_fxn
        })
        algorithm = BehaviorCloning(
            device=device,
            **model_hyperparams
        )
    else:
        raise NotImplementedError
    return algorithm

