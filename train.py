import yaml
import argparse
import wandb
from utils.reproducability import set_seed, set_device
from utils.data_handling import get_dataloaders, load_model, save_checkpoint
from utils.loss_functions import get_loss_function
from utils.training_functions import get_train_fn
from utils.validation_functions import get_val_fn, get_val_small_fn
from utils.optimizers import get_optimizers

def main():
    # Config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs/roadmap-gan.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    seed = config['seed']
    epochs = config['epochs']

    # Wandb support
    mode = "online" if (config['wandb_logging'] and not config['debug'])else "disabled"
    wandb.init(
        project="cil-road-segmentation", 
        entity="davincis", 
        config=config, 
        mode=mode
    )

    # Set seed and device
    set_seed(seed)
    device = set_device(config['device'])

    # Create Dataset and Dataloader
    train_dataloader, val_dataloader = get_dataloaders(config)

    # Load model (Takes also care of: Continue training from checkpoint)
    models = load_model(config, device)
    
    # Get Loss function
    loss_fn = get_loss_function(config['loss_function'])

    # Set optimizer
    optimizers = get_optimizers(models, config)

    # Get training and validation function
    train_fn = get_train_fn(config)
    val_fn = get_val_fn(config)
    val_small_fn = get_val_small_fn(config)

    # Before we start training we validate our model for a first time
    val_fn(models, loss_fn, val_dataloader, -1, config, device)

    # Loop through the Epochs
    for epoch in range(epochs):
        # Run through the epoch
        train_fn(models, loss_fn, optimizers, train_dataloader, val_small_fn, val_dataloader, epoch, config, device)

        # save model
        save_checkpoint(models, optimizers, config)

        # Validate the current models
        val_fn(models, loss_fn, val_dataloader, epoch, config, device)

    wandb.finish()


if __name__ == "__main__":
    main()
