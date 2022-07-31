"""
#
# This script is used together with a configuration file to train a model
#  
"""

import yaml
import argparse
import os
import torch
import sys
from utils.reproducability import set_seed, set_device
from utils.data_handling import load_model, get_dataloaders
from utils.test_functions import get_test_fn
from utils.wandb import initialize
from utils.create_submission import create_submission

def main():
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/gan_roadmap/test.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    yaml.dump(config, sys.stdout)

    # Wandb support
    initialize(config)

    if config['debug']:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.autograd.set_detect_anomaly(True)

    # Set seed and device
    set_seed(config['seed'])
    device = set_device(config['device'])

    # Create Dataset and Dataloader
    _, _, test_dataloader = get_dataloaders(config)

    # Load model
    models = load_model(config, device)

    # Get test function
    test_fn = get_test_fn(config)

    # Prediction
    test_fn(models, test_dataloader, config, device)

    # Create submission file
    create_submission(config)


if __name__ == "__main__":
    main()
