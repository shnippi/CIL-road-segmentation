import torch.nn as nn

def BCEWithLogitsLoss():
    return nn.BCEWithLogitsLoss()


def get_loss_function(loss_fn):
    if loss_fn == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss()
    else:
        raise ValueError("Your specified loss does not exist.")
