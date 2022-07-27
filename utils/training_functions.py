import torch
import wandb
from tqdm import tqdm

def get_train_fn(config):
    if config['generation_mode'] == "classic":
        train_fn = classic_train_fn
    elif config['generation_mode'] == 'gan':
        train_fn = gan_train_fn
    else:
        raise ValueError("Your specified geneartion mode does not exist")
    
    return train_fn

def classic_train_fn(models, loss_fn, optimizers, train_dataloader, epoch, config, device):
    model = models['gen']
    optimizer = optimizers['opt_gen']
    model.train()

    with tqdm(train_dataloader) as tepoch:
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            B = data['B']
            A, B = A.to(device), B.to(device)

            # Train the geneator
            out = model(A)
            loss = loss_fn(out, B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Progressbar and log to wandb
            config['train_loss_track'] = (9*config['train_loss_track'] + loss)/10
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = config['train_loss_track'])
            wandb.log({"epoch": epoch})
            wandb.log({"loss": config['train_loss_track']})


def gan_train_fn(models, loss_fn, optimizers, train_dataloader, epoch, config, device):
    gen = models['gen']
    disc  =models['disc']
    opt_gen = optimizers['opt_gen']
    opt_disc = optimizers['opt_disc']
    gen.train()
    disc.train()

    bceLogitLoss = torch.nn.BCEWithLogitsLoss()

    with tqdm(train_dataloader) as tepoch:
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            B = data['B']
            A, B = A.to(device), B.to(device)

            B_fake = gen(A)

            # Train Discriminator
            D_real = disc(A, B)
            D_fake = disc(A, B_fake.detach())
            D_real_loss = bceLogitLoss(D_real, torch.ones_like(D_real))
            D_fake_loss = bceLogitLoss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)
            opt_disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            # Train Generator
            D_fake = disc(A, B_fake)
            G_fake_loss = bceLogitLoss(D_fake, torch.ones_like(D_fake))
            Normal_loss = loss_fn(B_fake, B)
            G_loss = G_fake_loss + config['disc_lambda']*Normal_loss
            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            # Update Progressbar and log to wandb
            config['train_loss_track'] = (9*config['train_loss_track'] + (G_loss + D_loss))/10
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = config['train_loss_track'])
            wandb.log({"epoch": epoch})
            wandb.log({"loss": config['train_loss_track']})
            wandb.log({"loss-normal": Normal_loss*100})
            wandb.log({"loss-genfake": G_fake_loss})
            wandb.log({"loss-disc": D_loss})
