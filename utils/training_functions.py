import torch
from tqdm import tqdm

def get_train_fn(config):
    if config['model'] == "base-u-net":
        train_fn = unet_train_fn
    elif config['model'] == 'roadmap-gan':
        train_fn = roadmap_gan_train_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return train_fn

def unet_train_fn(models, loss_fn, optimizers, train_dataloader, epoch, device):
    model = models.model
    optimizer = optimizers.opt

    model.train()
    loss_sum = 0

    with tqdm(train_dataloader) as tepoch:
        for batch, (X, y) in enumerate(tepoch):
            X, y = X.to(device), y.to(device)
            y = y.float().unsqueeze(1).to(device)

            out = model(X)

            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))


def roadmap_gan_train_fn(models, loss_fn, optimizers, train_dataloader, epoch, device):
    gen = models.gen
    disc  =models.disc
    opt_gen = optimizers.opt_gen
    opt_disc = optimizers.opt_disc
    gen.train()
    disc.train()

    loss_sum = 0

    with tqdm(train_dataloader) as tepoch:
        for batch, (X, Y) in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            X, Y = X.to(device), Y.to(device)

            # Train Discriminator
            Y_fake = gen(X)
            D_real = disc(X, Y)
            D_fake = disc(X, Y_fake.detach())
            D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
            D_fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake))
            D_loss = D_real_loss + D_fake_loss

            opt_disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

            # Train Generator
            D_fake = disc(X, Y_fake)
            G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
            L1 = torch.nn.L1loss(Y_fake, Y) * 100
            G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()
            
            # Update Progressbar
            loss_sum += G_loss.item()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss_sum/(batch+1))
