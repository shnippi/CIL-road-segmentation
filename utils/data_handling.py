import torch
from dataset.map_dataset import MapDataset
from dataset.paired_dataset import PairedDataset
from torch.utils.data import DataLoader
from utils.transformations import get_transforms
from models.base_u_net import BASE_U_NET
from models.roadmap_gan import ROADMAP_GAN

def load_model(config, device):

    if config['model'] == "base-u-net":
        model = BASE_U_NET(in_channels=3, out_channels=1).to(device)
    elif config['model'] == 'roadmap-gan':
        model = ROADMAP_GAN().to(device)
    else:
        raise ValueError("Your specified model does not exist")

    if config['load_from_checkpoint']:
        print("Loading from checkpoint")
        load_checkpoint(torch.load(config['checkpoint_path']), model)

    return model
    

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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
