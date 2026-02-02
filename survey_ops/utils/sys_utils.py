import sys
import logging


import numpy as np
import random
import torch


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    
def setup_logger(save_dir, logging_filename, logging_level='debug'):
    # Create logger
    logger = logging.getLogger()
    if logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif logging_level == 'info':
        logger.setLevel(logging.INFO)
    else:
        raise NotImplementedError

    # Avoid duplicate handlers if called twice
    if logger.handlers:
        return logger
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(save_dir / logging_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add to handlers
    # console_format = logging.Formatter('%(levelname)s - %(message)s')
    format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(format)
    file_handler.setFormatter(format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"   
    )
    return device