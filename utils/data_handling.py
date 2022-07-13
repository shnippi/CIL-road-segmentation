import torch
import wandb
import os
from utils.transformations import get_transforms
from dataset.map_dataset import MapDataset
from dataset.paired_dataset import PairedDataset
from dataset.original_pix2pix import OriginalPix2Pix
from torch.utils.data import DataLoader
from models.base_u_net import BASE_U_NET
from models.pix2pix import Pix2Pix_Generator
from models.pix2pix import Pix2Pix_Descriminator
from models.pix2pixHD import Pix2PixHD_Generator
from models.pix2pixHD import Pix2PixHD_Descriminator

def load_model(config, device):
    '''
    Return a dictionary of all the needed models. In Case of a GAN the dictionary contains the discriminator and generator.
    '''
    if config['model'] == "base-u-net":
        model = BASE_U_NET(in_channels=3, out_channels=1).to(device)
        models = {'unet': model}
        wandb.watch(models['unet'])
    elif config['model'] == 'pix2pix':
        gen = Pix2Pix_Generator().to(device)
        disc = Pix2Pix_Descriminator().to(device)
        models = {'gen': gen, 'disc': disc}
        wandb.watch((models['gen'], models['disc']))
    elif config['model'] == 'pix2pixHD':
        gen = Pix2PixHD_Generator().to(device)
        disc = Pix2PixHD_Descriminator().to(device)
        models = {'gen': gen, 'disc': disc}
        wandb.watch((models['gen'], models['disc']))
    else:
        raise ValueError("Your specified model does not exist")

    if config['load_from_checkpoint']:
        models = load_checkpoint(config, models)

    return models
    

def save_checkpoint(models, optimizers, config, epoch):
    # If we don't explicitly say that we want to save we skip
    if not config['save_checkpoint']:
        return

    # Create directiroy 
    directory = os.path.join(config['checkpoint_root'], config['wandb_run_name'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if config['model'] == "base-u-net":
        return
    elif config['model'] == 'pix2pix':
        # Save generator
        filename =  'epoch_' + str(epoch) + '_' + config['model']+ '_gen' + '.pth.tar'
        path = os.path.join(directory, filename)
        checkpoint = {
            "state_dict": models['gen'].state_dict(),
            "optimizer": optimizers['opt_gen'].state_dict(),
        }
        torch.save(checkpoint, path)

        # Save discriminator
        filename =  'epoch_' + str(epoch) + '_' + config['model'] + '_disc' + '.pth.tar'
        path = os.path.join(directory, filename)
        checkpoint = {
            "state_dict": models['disc'].state_dict(),
            "optimizer": optimizers['opt_disc'].state_dict(),
        }
        torch.save(checkpoint, path)
    elif config['model'] == 'pix2pixHD':
        # Save generator
        filename =  'epoch_' + str(epoch) + '_' + config['model']+ '_gen' + '.pth.tar'
        path = os.path.join(directory, filename)
        checkpoint = {
            "state_dict": models['gen'].state_dict(),
            "optimizer": optimizers['opt_gen'].state_dict(),
        }
        torch.save(checkpoint, path)

        # Save discriminator
        filename =  'epoch_' + str(epoch) + '_' + config['model'] + '_disc' + '.pth.tar'
        path = os.path.join(directory, filename)
        checkpoint = {
            "state_dict": models['disc'].state_dict(),
            "optimizer": optimizers['opt_disc'].state_dict(),
        }
        torch.save(checkpoint, path)
    else:
        raise ValueError("Your specified model does not exist")



def load_checkpoint(config, models):
    if config['model'] == "base-u-net":
        return
    elif config['model'] in ['pix2pix', 'pix2pixHD']:
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' +  config['model'] + '_gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        models['gen'].load_state_dict(checkpoint["state_dict"])

        if config['mode'] == "train":
            # Load discriminator
            filename_disc =  'epoch_' + str(config['epoch_count']) + '_' +  config['model'] + '_disc' + '.pth.tar'
            path_disc = os.path.join(config['checkpoint_load_pth'], filename_disc)
            checkpoint = torch.load(path_disc, config['device'])
            models['disc'].load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError("Your specified model does not exist")

    return models


def get_dataloaders(config):

    train_transform, val_transform = get_transforms(config)

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

        test_dataset = MapDataset(
            image_dir=config['test_dir'],
            mask_dir=None,
            transform=val_transform,
        )

    elif config['dataset'] == 'PairedDataset':
        train_dataset = PairedDataset(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='train',
            transform=train_transform
        )

        val_dataset = PairedDataset(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='val',
            transform=val_transform
        )

        test_dataset = PairedDataset(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='test',
            transform=val_transform
        )

    elif config['dataset'] == 'OriginalPix2Pix':
        train_dataset = OriginalPix2Pix(
            root_dir=config['root_dir'],
            phase='train',
            transform=train_transform
        )

        val_dataset = OriginalPix2Pix(
            root_dir=config['root_dir'],
            phase='val',
            transform=val_transform
        )

        test_dataset = OriginalPix2Pix(
            root_dir=config['root_dir'],
            phase='test',
            transform=val_transform
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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader
