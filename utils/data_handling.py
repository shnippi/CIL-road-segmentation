import imp
import torch
import wandb
import os
from utils.transformations import get_transforms
from dataset.map_dataset import MapDataset
from dataset.paired_dataset import PairedDataset
from dataset.original_pix2pix import OriginalPix2Pix
from dataset.paired_dataset_label import PairedDatasetLabel
from torch.utils.data import DataLoader
from models.convnext_unet import ConvNext_Unet
from models.drnA import DRN_A
from models.drnD import DRN_D
from models.pathGAN_discriminator import PatchGAN_Descriminator
from models.pix2pix import Pix2Pix_Generator
from models.pix2pixHD import Pix2PixHD_Generator
from models.unet import UNet
from models.unet3plus import Unet3plus

def load_model(config, device):
    '''
    Return a dictionary of all the needed models. In Case of a GAN the dictionary contains the discriminator and generator.
    '''
    # Model selection
    if config['model'] == "unet":
        gen = UNet(in_channels=3, out_channels=3).to(device)
    elif config['model'] == 'drnA':
        gen = DRN_A(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'drnD':
        gen = DRN_D(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'pix2pix':
        gen = Pix2Pix_Generator(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'pix2pixHD':
        gen = Pix2PixHD_Generator(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'unet3plus':
        gen = Unet3plus(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'convnext_unet':
        gen = ConvNext_Unet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Your specified model does not exist")

    # Set the discriminator if we are in GAN-mode
    if config['generation_mode'] == 'classic':
        models = {'gen': gen}
        wandb.watch(models['gen'])
        print("Number of trainable Parameters: ", count_parameters(gen))
    elif config['generation_mode'] == 'gan':
        disc = PatchGAN_Descriminator(in_channels=6).to(device)
        models = {'gen': gen, 'disc': disc}
        wandb.watch((models['gen'], models['disc']))
        print("Number of trainable Parameters (gen): ", count_parameters(gen))
        print("Number of trainable Parameters (disc): ", count_parameters(disc))
    else:
        raise ValueError("Your specified model does not exist")

    # If load_from_checkpoint is enabled we load the model checkpoints
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
    
    if config['generation_mode'] == "classic":
        filename =  'epoch_' + str(epoch) + '_' + config['model']+ '_gen' + '.pth.tar'
        path = os.path.join(directory, filename)
        checkpoint = {
            "state_dict": models['gen'].state_dict(),
            "optimizer": optimizers['opt_gen'].state_dict(),
        }
        torch.save(checkpoint, path)
    elif config['generation_mode'] == 'gan':
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
    if config['generation_mode'] == 'classic':
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' +  config['model'] + '_gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        models['gen'].load_state_dict(checkpoint["state_dict"])
    elif config['generation_mode'] == 'gan':
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' +  config['model'] + '_gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        models['gen'].load_state_dict(checkpoint["state_dict"])

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

    elif config['dataset'] == 'PairedDatasetLabel':
        train_dataset = PairedDatasetLabel(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='train',
            transform=train_transform
        )

        val_dataset = PairedDatasetLabel(
            root_A=config['root_A'],
            root_B=config['root_B'],
            phase='val',
            transform=val_transform
        )

        test_dataset = PairedDatasetLabel(
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
