import numpy as np
import matplotlib.pyplot as plt
import random

import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C, DDPG, SAC, TD3, HER
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import optuna

import os
import time
import pickle
import sys
sys.path.append('../src/')

from utils import save_results, load_results

Nf = 20
nv_max = 2
random.seed(2)
true_sequence = np.array([np.full(nv_max, i) for i in range(Nf)]).flatten()
true_sequence = true_sequence.tolist()
random.shuffle(true_sequence)
print(true_sequence)


import sys
import argparse

def cli_args():
        # Make parser object
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("-Nf", "--Nfields", type=int)
    parser.add_argument("-vmax", "--max-visits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("-v", "--verbosity", type=bool)
                   
    return(parser.parse_args())


# Try running with these args
#
# "Hello" 123 --enable
if __name__ == '__main__':
        
    try:
        args = cli_args()
        print(args)
    except:
        print('Failed')

    print()