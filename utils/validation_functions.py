from tqdm import tqdm
from utils.evaluations import check_accuracy, save_predictions_as_imgs

def get_val_fn(config):
    if config['model'] == "base-u-net":
        val_fn = unet_val_fn
    elif config['model'] == 'roadmap-gan':
        val_fn = roadmap_gan_val_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return val_fn

def unet_val_fn(config, model, val_dataloader, epoch, device):
    # Check accuracy
    check_accuracy(val_dataloader, model, device=device)

    # print some examples to a folder
    save_predictions_as_imgs(val_dataloader, model, folder="data/saved_images/", device=device)
    return


def roadmap_gan_val_fn(config, model, val_dataloader, epoch, device):

    return
