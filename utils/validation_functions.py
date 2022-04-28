import torch
import wandb
from tqdm import tqdm
from utils.evaluations import check_accuracy, save_predictions_as_imgs
from torchvision import transforms


def get_val_fn(config):
    if config['model'] == "base-u-net":
        val_fn = unet_val_fn
    elif config['model'] == 'roadmap-gan':
        val_fn = roadmap_gan_val_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return val_fn


def get_val_small_fn(config):
    if config['model'] == "base-u-net":
        raise ValueError("Not implemented yet")
    elif config['model'] == 'roadmap-gan':
        val_fn = roadmap_gan_val_small_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return val_fn

def unet_val_fn(models, val_dataloader, epoch, config, device):
    model = models['model']
    # Check accuracy
    check_accuracy(val_dataloader, model, device=device)

    # print some examples to a folder
    save_predictions_as_imgs(val_dataloader, model, folder="data/saved_images/", device=device)
    return


def roadmap_gan_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
    gen = models['gen']
    disc  =models['disc']

    l1 = torch.nn.L1Loss()
    loss_sum = 0

    with torch.no_grad():
        with tqdm(val_dataloader) as tepoch:
            for batch, data in enumerate(tepoch):
                # X:= Sattelite, Y:= Roadmap
                A = data['A']
                B = data['B']
                A, B = A.to(device), B.to(device)

                # Run Discriminator
                B_fake = gen(A)
                D_real = disc(A, B)
                D_fake = disc(A, B_fake.detach())
                D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
                D_fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss)/2

                # Run Generator
                D_fake = disc(A, B_fake)
                G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
                L1_loss = l1(B_fake, B) * config['l1_lambda']
                G_loss = G_fake_loss + L1_loss
                
                # Update Progressbar and log to wandb
                loss_sum += G_loss.item() + D_loss.item()
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss = loss_sum/(batch+1))
                wandb.log({"epoch": epoch})
                wandb.log({"val-loss": loss_sum/(batch+1)})
                wandb.log({"val-loss-gen": G_loss})
                wandb.log({"val-loss-gen-fake": G_fake_loss})
                wandb.log({"val-loss-gen-l1": L1_loss})
                wandb.log({"val-loss-disc": D_loss})

                # Log first 4 images of each epoch
                if batch == 0:
                    num_samples = min(4,B_fake.shape[0])
                    caption = "epoch_" + str(epoch)

                    # Generated Roadmap Examples
                    image_tensors = [B_fake[i]*0.5+0.5 for i in range(num_samples)]
                    images = [transforms.ToPILImage()(image) for image in image_tensors]
                    wandb.log({"val-generated": [wandb.Image(image, caption=caption) for image in images]})

                    if epoch == 0:
                        # Sattelite Images
                        image_tensors = [A[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-original": [wandb.Image(image, caption=caption) for image in images]})

                        # Real Roadmap images
                        image_tensors = [B[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-real": [wandb.Image(image, caption=caption) for image in images]})


def roadmap_gan_val_small_fn(models, loss_fn, val_dataloader, epoch, batch_nr, config, device):
    gen = models['gen']
    disc  =models['disc']
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            if batch == 0:
                # X:= Sattelite, Y:= Roadmap
                A = data['A']
                B = data['B']
                A, B = A.to(device), B.to(device)

                # Run Discriminator
                B_fake = gen(A)

                num_samples = min(4,B_fake.shape[0])
                caption = "epoch_" + str(epoch) + "_batch_" + str(batch_nr)

                # Sattelite Images
                image_tensors = [A[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-small-original": [wandb.Image(image, caption=caption) for image in images]})

                # Real Roadmap images
                image_tensors = [B[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"vals-small-real": [wandb.Image(image, caption=caption) for image in images]})

                # Generated Roadmap Examples
                image_tensors = [B_fake[i]*0.5+0.5 for i in range(num_samples)]
                images = [transforms.ToPILImage()(image) for image in image_tensors]
                wandb.log({"val-small-generated": [wandb.Image(image, caption=caption) for image in images]})

