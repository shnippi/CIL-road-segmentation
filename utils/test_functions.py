from torchvision import transforms
import torch
import wandb
from tqdm import tqdm
from PIL import Image

def get_test_fn(config):
    if config['model'] == "base-u-net":
        train_fn = None
    elif config['model'] == 'pix2pix':
        train_fn = pix2pix_test_fn
    elif config['model'] == 'pix2pixHD':
        train_fn = None
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return train_fn


def pix2pix_test_fn(
        models, 
        test_dataloader, 
        config, 
        device):

    gen = models['gen']
    #gen.eval()


    with tqdm(test_dataloader) as tepoch:
        count = 144
        trans_tensToPil = transforms.ToPILImage()
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            A = A.to(device)

            # Train Discriminator
            B_fake = gen(A)
            
            pil_img = trans_tensToPil(B_fake.squeeze())
            pil_img = pil_img.resize((400, 400))

            path = "results/roadmaps/" + str(count) + ".png"
            pil_img.save(path)
            count += 1
