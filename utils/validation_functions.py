"""
#
# This file contains functions to validate the different approaches, gan or cloassic
#  
"""

import torch
import wandb
from tqdm import tqdm
from utils.evaluations import check_accuracy, save_predictions_as_imgs
from torchvision import transforms


def get_val_fn(config):
    if config['generation_mode'] == "classic":
        val_fn = classic_val_fn
    elif config['generation_mode'] == "gan":
        val_fn = gan_val_fn
    else:
        raise ValueError("Your specified model mode function does not exist")
    
    return val_fn



def classic_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
    gen = models['gen']

    with torch.no_grad():
        with tqdm(val_dataloader) as tepoch:
            num_samples = 8
            generated = []
            label =[]
            sattelite = []

            for batch, data in enumerate(tepoch):
                # X:= Sattelite, Y:= Roadmap
                A = data['A']
                B = data['B']
                A, B = A.to(device), B.to(device)

                # Generate Fake-image and calculate loss
                B_fake = gen(A)
                loss = loss_fn(B_fake, B)

                # Track and log the loss
                config['val_loss_track'] = (9*config['val_loss_track'] + loss.item())/10
                wandb.log({"val-loss": config['val_loss_track']})
                tepoch.set_postfix(loss = config['val_loss_track'])

                # We save some images for logging purposes
                if len(generated) < num_samples:
                    generated.append(B_fake[0])
                    sattelite.append(A[0])
                    label.append(B[0])

            # Generated Roadmap/Mask Examples
            caption = "epoch_" + str(epoch)
            image_tensors = [generated[i]*0.5+0.5 for i in range(num_samples)]
            images = [transforms.ToPILImage()(image) for image in image_tensors]
            wandb.log({"val-generated": [wandb.Image(image, caption=caption) for image in images]})

            # We only need to log the real images once
            if epoch == config['epoch_count']:
                # Sattelite Images
                image_tensors = [sattelite[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-sattelite": [wandb.Image(image, caption=caption) for image in images]})

                # Real Roadmap/Mask images
                image_tensors = [label[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-real": [wandb.Image(image, caption=caption) for image in images]})



def gan_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
    gen = models['gen']

    with torch.no_grad():
        with tqdm(val_dataloader) as tepoch:
            num_samples = 8
            generated = []
            label =[]
            sattelite = []

            for batch, data in enumerate(tepoch):

                # X:= Sattelite, Y:= Roadmap
                A = data['A']
                B = data['B']
                A, B = A.to(device), B.to(device)

                # Generate Fake-image and calculate loss
                B_fake = gen(A)
                loss = loss_fn(B_fake, B)

                # Update Progressbar and log to wandb
                config['val_loss_track'] = (9*config['val_loss_track'] + loss.item())/10
                wandb.log({"loss-val": config['val_loss_track']})
                tepoch.set_postfix(loss = config['val_loss_track'])

                if len(generated) < num_samples:
                    generated.append(B_fake[0])
                    sattelite.append(A[0])
                    label.append(B[0])


            # Generated Roadmap Examples
            caption = "epoch_" + str(epoch)
            image_tensors = [generated[i]*0.5+0.5 for i in range(num_samples)]
            images = [transforms.ToPILImage()(image) for image in image_tensors]
            wandb.log({"val-generated": [wandb.Image(image, caption=caption) for image in images]})

            if epoch == config['epoch_count']:
                # Sattelite Images
                image_tensors = [sattelite[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-original": [wandb.Image(image, caption=caption) for image in images]})

                # Real Roadmap images
                image_tensors = [label[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-real": [wandb.Image(image, caption=caption) for image in images]})
