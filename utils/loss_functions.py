import torch.nn as nn
import torch
from torch.autograd import Variable


def get_loss_function(config):
    loss_fn_name = config['loss_function']
    if loss_fn_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss()
    elif loss_fn_name == 'BCEWithLogitsLossOverList':
        return GANLoss(config)
    elif loss_fn_name == "BCELoss":
        return BCELoss()
    else:
        raise ValueError("Your specified loss does not exist.")

def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()

def BCELoss():
    return nn.BCELoss()


class GANLoss(nn.Module):
    def __init__(self, config, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = torch.cuda.FloatTensor if config['device']=='cuda' else torch.Tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def dice_loss(self, y_true, y_pred):
    loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
    return loss