import re
import wandb

def initialize(config):
    mode = "online" if (config['wandb_logging']) else "disabled"
    resume = False if config['load_from_checkpoint'] else False
    if resume:
        print("Resume logging in wandb")

    wandb.init(
        project="cil-road-segmentation", 
        entity="davincis", 
        config=config, 
        mode=mode,
        resume=resume
    )
    config['wandb_run_name'] = wandb.run.name
