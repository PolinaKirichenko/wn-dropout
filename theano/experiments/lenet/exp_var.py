from __future__ import print_function

import sys
import warnings
from nets import objectives
from nets import optpolicy, layers
from experiments.utils import run_experiment, apply_net
import numpy as np
import os
from experiments.lenet.lenet300_100 import net_lenet5_wn_proj_do, net_lenet5_wn_gaus

warnings.simplefilter("ignore")


dataset = 'mnist'
iparam = str(sys.argv[1]) if len(sys.argv) > 1 else None
print('dataset = ', dataset, 'params = ', iparam)

num_epochs, batch_size, verbose = 1, 100, 1
optpol_linear = lambda epoch: optpolicy.lr_linear(num_epochs, epoch, 1e-3)

# maybe should sample multiple times at one point?
# eps_i (ccordinate-wise noise) is not independent 
# separately for layers and for all at the same time
updates_per_epoch = 1
arch = net_lenet5_wn_gaus
folder_name = 'wn_var'
filename = 'wn_gaus_var'
trainset_sizes = [10000]
alphas = [1]
ave_times = 50


if not os.path.exists('./experiments/logs/' + folder_name):
    os.mkdir('./experiments/logs/' + folder_name)


for trainset_size in trainset_sizes:
    for alpha in alphas:
        log_fname = folder_name + '/' + filename + '-' + str(trainset_size) + '-' + str(alpha)
        net = run_experiment(
            dataset, num_epochs, batch_size, arch, objectives.nll, verbose,
            optpol_linear, optpolicy.rw_linear, params=None, optimizer='adam', train_clip=False,
            trainset_size=trainset_size, log_fname=log_fname, alpha=alpha,
            noise_ave_times=ave_times
        )
