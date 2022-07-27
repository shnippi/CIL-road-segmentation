import torch.optim as optim
import torch
import os

def get_optimizers(models, config):
    '''
    Returns a dictionary with the optimizers for the models
    '''
    if config['generation_mode'] == 'classic':
        opt = optim.Adam(models['gen'].parameters(), lr=config['learning_rate'])
        optimizers = {'opt_gen': opt}
    elif config['generation_mode'] == 'gan':
        opt_gen = optim.Adam(models['gen'].parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
        opt_disc = optim.Adam(models['disc'].parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
        optimizers = {'opt_gen': opt_gen, 'opt_disc': opt_disc}       
    else:
        raise ValueError("Your specified model mode does not exist (get_optimizers)")
    
    if config['load_from_checkpoint']:
        optimizers = load_optimizer_checkpoint(config, optimizers)

    return optimizers


def load_optimizer_checkpoint(config, optimizers):
    if config['generation_mode'] == 'classic':
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' + config['model'] + '_gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        optimizers['opt_gen'].load_state_dict(checkpoint["optimizer"])
    elif config['generation_mode'] == 'gan':
        # Load generator
        filename_gen =  'epoch_' + str(config['epoch_count']) + '_' + config['model'] + '_gen' + '.pth.tar'
        path_gen = os.path.join(config['checkpoint_load_pth'], filename_gen)
        checkpoint = torch.load(path_gen, config['device'])
        optimizers['opt_gen'].load_state_dict(checkpoint["optimizer"])

        # Load discriminator
        filename_disc =  'epoch_' + str(config['epoch_count']) + '_' + config['model'] + '_disc' + '.pth.tar'
        path_disc = os.path.join(config['checkpoint_load_pth'], filename_disc)
        checkpoint = torch.load(path_disc, config['device'])
        optimizers['opt_disc'].load_state_dict(checkpoint["optimizer"])
    else:
        raise ValueError("Your specified model's optimizer checkpoint don't exist")

    return optimizers


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
