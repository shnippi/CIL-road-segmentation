import torch.optim as optim

def get_optimizers(models, config):
    '''
    Returns a dictionary with the optimizers for the models
    '''
    if config['model'] == "base-u-net":
        opt = optim.Adam(models.unet.parameters(), lr=config['learning_rate'])
        optimizers = {'opt': opt}
    elif config['model'] == 'roadmap-gan':
        opt_gen = optim.Adam(models.gen.parameters(), lr=config['learning_rate'])
        opt_disc = optim.Adam(models.disc.parameters(), lr=config['learning_rate'])
        optimizers = {'opt_gen': opt_gen, 'opt_disc': opt_disc}
    else:
        raise ValueError("Your specified model does not exist (get_optimizers)")
    
    return optimizers
