import torch
import yaml
import argparse
import wandb
import torch.optim as optim
from utils.reproducability import set_seed, set_device
from utils.data_handling import get_dataloaders, load_model, save_checkpoint
from utils.loss_functions import get_loss_function
from utils.training_functions import get_train_fn
from utils.validation_functions import get_val_fn

def main():
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    seed = config['seed']
    epochs = config['epochs']

    # Wandb support
    mode = "online" if config['wandb_logging'] else "disabled"
    wandb.init(
        project="cil-road-segmentation", 
        entity="cil-road-segmentation", 
        config=config, 
        mode=mode
    )

    # Set seed and device
    set_seed(seed)
    device = set_device()

    # Create Dataset and Dataloader
    train_dataloader, val_dataloader = get_dataloaders(config)

    # Load model (Takes also care of: Continue training from checkpoint)
    model = load_model(config, device)
    wandb.watch(model)
    
    # Get Loss function
    loss_fn = get_loss_function(config['loss_function'])

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Get training and validation function
    train_fn = get_train_fn(config)
    val_fn = get_val_fn(config)

    # Before we start training we validate our model for a first time
    val_fn(config, model, val_dataloader, epoch, device)

    # Loop through the Epoch
    for epoch in range(epochs):
        try:
            train_fn(model, loss_fn, optimizer, train_dataloader, epoch, device)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            val_fn(config, model, val_dataloader, epoch, device)

        except:
            print("Exception occured. We skip to next epoch")
            continue


if __name__ == "__main__":
    main()
