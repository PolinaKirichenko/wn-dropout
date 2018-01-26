import numpy as np


# first half of epochs -- lr, the second half linearly decreases to 0
def lr_linear(num_epochs, epoch, lr):
    return max(0, lr * np.minimum(2. - 2. * epoch / (1. * num_epochs), 1.))

def lr_linear_to0(num_epochs, epoch, lr):
    return lr * max(0, (1. * num_epochs - epoch) / (1. * num_epochs))

def lr_const(epoch):
    return 1e-3
