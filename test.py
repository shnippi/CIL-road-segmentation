import yaml
import argparse
import os
import torch
from utils.reproducability import set_seed, set_device
from utils.data_handling import load_model, get_dataloaders
from utils.test_functions import get_test_fn
from utils.wandb import initialize

def main():
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/pix2pix.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    seed = config['seed']

    # Wandb support
    initialize(config)

    if config['debug']:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.autograd.set_detect_anomaly(True)

    # Set seed and device
    set_seed(seed)
    device = set_device(config['device'])

    # Create Dataset and Dataloader
    _, _, test_dataloader = get_dataloaders(config)

    # Load model (Takes also care of: Continue training from checkpoint)
    models = load_model(config, device)

    # Get training and validation function
    test_fn = get_test_fn(config)

    # Do the actual testing
    test_fn(models, test_dataloader, config, device)



if __name__ == "__main__":
    main()
