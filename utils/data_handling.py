import torch
from dataset.map_dataset import MapDataset
from dataset.paired_dataset import PairedDataset
from torch.utils.data import DataLoader
from models.base_u_net import BASE_U_NET
from models.roadmap_gan import Pix2Pix_Generator
from models.roadmap_gan import Pix2Pix_Descriminator
from utils.transformations import get_transforms

def load_model(config, device):
    '''
    Return a dictionary of all the needed models. In Case of a GAN the dictionary contains the discriminator and generator.
    '''
    if config['model'] == "base-u-net":
        model = BASE_U_NET(in_channels=3, out_channels=1).to(device)
        models = {'unet': model}
        
    elif config['model'] == 'roadmap-gan':
        gen = Pix2Pix_Generator()
        disc = Pix2Pix_Descriminator()
        models = {'gen': gen, 'disc': disc}
    else:
        raise ValueError("Your specified model does not exist")

    if config['load_from_checkpoint']:
        print("Loading from checkpoint")
        load_checkpoint(torch.load(config['checkpoint_path']), model)

    return models
    

def save_checkpoint(models, optimizers, config):
    # TODO
    # Iterate through models and optimizers and save them
    return


def load_checkpoint(checkpoint, model):
    # TODO
    # Load from checkpoint files. Here we may need actually again thw switch statement with different logic
    # for different models
    return


def get_dataloaders(config):

    train_transform,val_transform = get_transforms(config)

    if config['dataset'] == 'MapDataset':
        train_dataset = MapDataset(
            image_dir=config['train_dir'],
            mask_dir=config['train_maskdir'],
            transform=train_transform,
        )

        val_dataset = MapDataset(
            image_dir=config['val_dir'],
            mask_dir=config['val_maskdir'],
            transform=val_transform,
        )
    elif config['dataset'] == 'PairedDataset':
        train_dataset = PairedDataset(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='train'
        )

        val_dataset = PairedDataset(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='val'
        )
    else:
        raise ValueError("Your specified dataset does not exist")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        shuffle=False,
    )

    return train_dataloader, val_dataloader
