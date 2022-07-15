import torch
import wandb
from tqdm import tqdm
from utils.evaluations import check_accuracy, save_predictions_as_imgs
from torchvision import transforms


def get_val_fn(config):
    if config['model'] == "baseunet":
        val_fn = baseunet_val_fn
    elif config['model'] == "pix2pix":
        val_fn = pix2pix_val_fn
    elif config['model'] == "pix2pixHD":
        val_fn = pix2pixHD_val_fn
    else:
        raise ValueError("Your specified model's validation function does not exist")
    
    return val_fn


def get_val_small_fn(config):
    if config['model'] == "baseunet":
        val_fn = baseunet_val_small_fn
    elif config['model'] == 'pix2pix':
        val_fn = pix2pix_val_small_fn
    elif config['model'] == "pix2pixHD":
        val_fn = pix2pixHD_val_small_fn
    else:
        raise ValueError("Your specified model's small validation function does not exist")
    
    return val_fn


def baseunet_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
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

                # Run Discriminator
                B_fake = gen(A)

                generated.append(B_fake[0])
                sattelite.append(A[0])
                label.append(B[0])

                if len(generated) >= num_samples:

                    caption = "epoch_" + str(epoch)

                    # Generated Roadmap Examples
                    image_tensors = [generated[i]*0.5+0.5 for i in range(num_samples)]
                    images = [transforms.ToPILImage()(image) for image in image_tensors]
                    wandb.log({"val-generated": [wandb.Image(image, caption=caption) for image in images]})

                    if epoch == 0:
                        # Sattelite Images
                        image_tensors = [sattelite[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-original": [wandb.Image(image, caption=caption) for image in images]})

                        # Real Roadmap images
                        image_tensors = [label[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-real": [wandb.Image(image, caption=caption) for image in images]})
                        
                    return

def baseunet_val_small_fn(models, loss_fn, val_dataloader, epoch, batch_nr, config, device):
    gen = models['gen']
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


def pix2pix_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
    gen = models['gen']
    disc  =models['disc']

    l1 = torch.nn.L1Loss()
    loss_sum = 0

    with torch.no_grad():
        with tqdm(val_dataloader) as tepoch:
            num_samples = 8
            generated = []
            label =[]
            sattelite = []
            loss_sum = 0

            for batch, data in enumerate(tepoch):
                # X:= Sattelite, Y:= Roadmap
                A = data['A']
                B = data['B']
                A, B = A.to(device), B.to(device)

                # Run Discriminator
                B_fake = gen(A)
                loss = loss_fn(B_fake, B)

                loss_sum += float(loss.item())
                wandb.log({"loss-val": loss_sum/(batch+1)})
                tepoch.set_postfix(loss = loss_sum/(batch+1))

                if len(generated) < num_samples:
                    generated.append(B_fake[0])
                    sattelite.append(A[0])
                    label.append(B[0])


            caption = "epoch_" + str(epoch)

            # Generated Roadmap Examples
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
                
            return


def pix2pix_val_small_fn(models, loss_fn, val_dataloader, epoch, batch_nr, config, device):
    gen = models['gen']
    disc = models['disc']
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



def pix2pixHD_val_fn(models, loss_fn, val_dataloader, epoch, config, device):
    gen = models['gen']
    disc  =models['disc']

    l1 = torch.nn.L1Loss()
    loss_sum = 0

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

                # Run Discriminator
                B_fake = gen(A)

                generated.append(B_fake[0])
                sattelite.append(A[0])
                label.append(B[0])

                if len(generated) >= num_samples:

                    caption = "epoch_" + str(epoch)

                    # Generated Roadmap Examples
                    image_tensors = [generated[i]*0.5+0.5 for i in range(num_samples)]
                    images = [transforms.ToPILImage()(image) for image in image_tensors]
                    wandb.log({"val-generated": [wandb.Image(image, caption=caption) for image in images]})

                    if epoch == 0:
                        # Sattelite Images
                        image_tensors = [sattelite[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-original": [wandb.Image(image, caption=caption) for image in images]})

                        # Real Roadmap images
                        image_tensors = [label[i]*0.5+0.5 for i in range(num_samples)]
                        images = [transforms.ToPILImage()(image) for image in image_tensors]
                        wandb.log({"val-real": [wandb.Image(image, caption=caption) for image in images]})
                        
                    return

def pix2pixHD_val_small_fn(models, loss_fn, val_dataloader, epoch, batch_nr, config, device):
    gen = models['gen']
    disc = models['disc']
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
