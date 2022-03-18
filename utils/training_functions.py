from tqdm import tqdm

def get_train_fn(config):
    if config['model'] == "base-u-net":
        train_fn = unet_train_fn
    elif config['model'] == 'roadmap-gan':
        train_fn = roadmap_gan_train_fn
    else:
        raise ValueError("Your specified model's training function does not exist")
    
    return train_fn

def unet_train_fn(model, loss_fn, optimizer, train_dataloader, epoch, device):
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


def roadmap_gan_train_fn(model, loss_fn, optimizer, train_dataloader, epoch, device):
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
