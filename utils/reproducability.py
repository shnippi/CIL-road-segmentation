import torch
import os
import random
import numpy as np

def set_seed(seed):
    """Set ALL random seeds"""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def set_device(device):
    """Use GPU if available, otherwise cpu. Unless specified explicitly"""
    if device != None:
        device = device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nUsing device: {device}")
    return device
