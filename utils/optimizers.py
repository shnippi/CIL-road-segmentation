import torch.optim as optim
import torch
import os

def get_optimizers(models, config):
    '''
    Returns a dictionary with the optimizers for the models
    '''
    if config['model'] == "base-u-net":
        opt = optim.Adam(models['unet'].parameters(), lr=config['learning_rate'])
        optimizers = {'opt': opt}
    elif config['model'] == 'roadmap-gan':
        opt_gen = optim.Adam(models['gen'].parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
        opt_disc = optim.Adam(models['disc'].parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
        optimizers = {'opt_gen': opt_gen, 'opt_disc': opt_disc}
    else:
        raise ValueError("Your specified model does not exist (get_optimizers)")
    
    if config['load_from_checkpoint']:
        optimizers = load_checkpoint(config, optimizers)

    return optimizers


def load_checkpoint(config, optimizers):
    if config['model'] == "base-u-net":
        return
    elif config['model'] == 'roadmap-gan':
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' + 'gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        optimizers['opt_gen'].load_state_dict(checkpoint["optimizer"])

        # Load discriminator
        filename_disc =  'epoch_' + str(config['epoch_count']) + '_' + 'disc' + '.pth.tar'
        path_disc = os.path.join(config['checkpoint_load_pth'], filename_disc)
        checkpoint = torch.load(path_disc, config['device'])
        optimizers['opt_disc'].load_state_dict(checkpoint["optimizer"])
    else:
        raise ValueError("Your specified model does not exist")

    return optimizers
