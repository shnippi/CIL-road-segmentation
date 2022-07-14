import torch
import wandb
from tqdm import tqdm

def get_train_fn(config):
    if config['model'] == "baseunet":
        train_fn = baseunet_train_fn
    elif config['model'] == 'pix2pix':
        train_fn = pix2pix_train_fn
    elif config['model'] == 'pix2pixHD':
        train_fn = pix2pixHD_train_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return train_fn

def baseunet_train_fn(
        models, 
        loss_fn, 
        optimizers, 
        train_dataloader, 
        val_small_fn,
        val_small_dataloader,
        epoch, 
        config, 
        device):

    model = models['gen']
    optimizer = optimizers['opt_gen']

    model.train()
    loss_sum = 0

    with tqdm(train_dataloader) as tepoch:
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            B = data['B']
            A, B = A.to(device), B.to(device)

            out = model(A)

            loss = loss_fn(out, B)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))


def pix2pix_train_fn(
        models, 
        loss_fn, 
        optimizers, 
        train_dataloader, 
        val_small_fn,
        val_small_dataloader,
        epoch, 
        config, 
        device):

    gen = models['gen']
    disc  =models['disc']
    opt_gen = optimizers['opt_gen']
    opt_disc = optimizers['opt_disc']
    gen.train()
    disc.train()

    l1 = torch.nn.L1Loss()
    loss_sum = 0

    with tqdm(train_dataloader) as tepoch:
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            B = data['B']
            A, B = A.to(device), B.to(device)

            # Train Discriminator
            B_fake = gen(A)
            D_real = disc(A, B)
            D_fake = disc(A, B_fake.detach())
            D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
            D_fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)

            opt_disc.zero_grad()
            D_loss.backward()
            # Maybe this is actually wrong? Because we need disc in the discriminator step
            # Since in the next step we update the Generator (but already with the updated Disc)
            # The actual opt_step should parobably done after the GAN pass.
            opt_disc.step()

            # Train Generator
            D_fake = disc(A, B_fake)
            G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
            L1_loss = l1(B_fake, B) * config['l1_lambda']
            G_loss = G_fake_loss + L1_loss

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            # Update Progressbar and log to wandb
            loss_sum += float(G_loss.item()) + float(D_loss.item())
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))
            wandb.log({"epoch": epoch})
            wandb.log({"loss": loss_sum/(batch+1)})
            wandb.log({"loss-gen": G_loss})
            wandb.log({"loss-gen-fake": G_fake_loss})
            wandb.log({"loss-gen-l1": L1_loss})
            wandb.log({"loss-disc": D_loss})

            # Some small ev
            # if batch % 1000 == 0:
            #    val_small_fn(models, loss_fn, val_small_dataloader, epoch, batch, config, device)




def pix2pixHD_train_fn(
        models, 
        loss_fn, 
        optimizers, 
        train_dataloader, 
        val_small_fn,
        val_small_dataloader,
        epoch, 
        config, 
        device):

    gen = models['gen']
    disc  =models['disc']
    opt_gen = optimizers['opt_gen']
    opt_disc = optimizers['opt_disc']
    gen.train()
    disc.train()

    l1 = torch.nn.L1Loss()
    loss_sum = 0

    with tqdm(train_dataloader) as tepoch:
        for batch, data in enumerate(tepoch):
            # A:= Sattelite, B:= Roadmap
            sattelite = data['A']
            roadmap = data['B']
            sattelite, roadmap = sattelite.to(device), roadmap.to(device)

            # Generate fake roadmap
            roadmap_fake = gen(sattelite)

            # Descriminator fake loss
            input_concat = torch.cat((roadmap, roadmap_fake.detach()), dim=1)
            D_fake = disc(input_concat)
            D_fake_loss = loss_fn(D_fake, False)

            # Descriminator real loss
            input_concat = torch.cat((roadmap, roadmap.detach()), dim=1)
            D_real = disc(input_concat)
            D_real_loss = loss_fn(D_real, True)

            # GAN loss (Here we don't detach())
            input_concat = torch.cat((roadmap, roadmap_fake), dim=1)
            D_fake_undetached = disc(input_concat)
            G_loss = loss_fn(D_fake_undetached, True)

            # Feature Matching loss
            loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (disc.n_layers + 1)
            D_weights = 1.0 / disc.num_D
            for i in range(disc.num_D):
                for j in range(len(D_fake_undetached[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        l1(D_fake_undetached[i][j], D_real[i][j].detach()) * 10

            # Final losses
            loss_D = (D_fake_loss + D_real_loss)*0.5
            loss_G = G_loss + loss_G_GAN_Feat

            # Backward Pass and update parameters
            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            if batch%10 == 0:
                opt_disc.zero_grad()
                loss_D.backward()
                opt_disc.step()

            
            # Update Progressbar and log to wandb
            loss_sum += float(G_loss.item()) + float(loss_D.item())
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))
            wandb.log({"epoch": epoch})
            wandb.log({"loss": loss_sum/(batch+1)})
            wandb.log({"loss-gen": loss_G})
            wandb.log({"loss-disc": loss_D})

            # Some small ev
            if batch % 50 == 0:
                val_small_fn(models, loss_fn, val_small_dataloader, epoch, batch, config, device)

