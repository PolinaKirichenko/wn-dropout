from __future__ import print_function
import sys
import warnings
from nets import optpolicy
import experiments.utils
from experiments.utils import run_experiment
import numpy as np
import os
from experiments.lenet.lenet import FC_Net, FC_BinDO_Net
from experiments.lenet.lenet import WN_BinDO_Net, WN_Net
import torch
import torch.nn as nn


dataset = 'mnist'
criterion = nn.CrossEntropyLoss().cuda()
num_epochs, batch_size, verbose = 200, 100, 1

optpol_linear = lambda epoch: optpolicy.lr_linear(num_epochs, epoch, 1e-3)

arch = WN_Net
noise_type = None
alpha = None
#alphas = np.logspace(np.log10(0.01), np.log10(3), 8)
noise_magnitude = True
magn_vars = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

folder_name = 'wn_magn_batch'
filename = 'wn_magn_batch'
trainset_sizes = [100, 500, 1000, 5000, 10000, 50000]
ave_times = 50


if not os.path.exists('./experiments/logs'):
        os.mkdir('./experiments/logs')

if not os.path.exists('./experiments/logs/' + folder_name):
    os.mkdir('./experiments/logs/' + folder_name)


for trainset_size in trainset_sizes:
#    for alpha in alphas:
    for magn_var in magn_vars:
        log_fname = folder_name + '/' + filename + '-' + str(trainset_size) + '-' + str(alpha) + '-' + str(magn_var)
        run_experiment(
            dataset, num_epochs, batch_size, arch, criterion, verbose,
            optpol_linear, params=None, optimizer='adam',
            trainset_size=trainset_size, log_fname=log_fname,
            noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var,
            noise_ave_times=ave_times,
        )
