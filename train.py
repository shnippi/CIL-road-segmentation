"""
#
# This script creates predictions and a submission for a model loaded from a configuration file
#  
"""

import yaml
import argparse
import wandb
import os
import torch
import sys
from utils.reproducability import set_seed, set_device
from utils.data_handling import get_dataloaders, load_model, save_checkpoint
from utils.loss_functions import get_loss_function
from utils.training_functions import get_train_fn
from utils.validation_functions import get_val_fn
from utils.optimizers import get_optimizers
from utils.wandb import initialize

def main():
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/classic/train.yaml")
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
    train_dataloader, val_dataloader, _ = get_dataloaders(config)

    # Load model (Takes also care of: Continue training from checkpoint)
    models = load_model(config, device)
    
    # Get Loss function
    loss_fn = get_loss_function(config)

    # Set optimizer
    optimizers = get_optimizers(models, config)

    # Get training and validation function
    train_fn = get_train_fn(config)
    val_fn = get_val_fn(config)

    # Loop through the Epochs
    for epoch in range(config['epoch_count'], config['epochs']):
        # Run through the epoch
        train_fn(models, loss_fn, optimizers, train_dataloader, epoch, config, device)

        # save model
        save_checkpoint(models, optimizers, config, epoch)

        # Validate the current models
        val_fn(models, loss_fn, val_dataloader, epoch, config, device)

    wandb.finish()


if __name__ == "__main__":
    main()
