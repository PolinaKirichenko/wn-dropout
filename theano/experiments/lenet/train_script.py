from __future__ import print_function
import sys
import warnings
from nets import objectives
from nets import optpolicy, layers
from experiments.utils import run_experiment, apply_net
import numpy as np
import os
from experiments.lenet.lenet300_100 import net_lenet5_wn_proj_do, net_lenet5_wn_gaus, net_lenet5_wn_proj_do_tangent


dataset = 'mnist'
iparam = str(sys.argv[1]) if len(sys.argv) > 1 else None
print('dataset = ', dataset, 'params = ', iparam)

num_epochs, batch_size, verbose = 200, 100, 1
optpol_linear = lambda epoch: optpolicy.lr_linear(num_epochs, epoch, 1e-3)

arch = net_lenet5_wn_proj_do_tangent
folder_name = 'wn_tangent'
filename = 'wn_tang'
trainset_sizes = [100]
alphas = [0.01]
ave_times = 0


if not os.path.exists('./experiments/logs/' + folder_name):
    os.mkdir('./experiments/logs/' + folder_name)


for trainset_size in trainset_sizes:
    for alpha in alphas:
        log_fname = folder_name + '/' + filename + '-' + str(trainset_size) + '-' + str(alpha)
        net = run_experiment(
            dataset, num_epochs, batch_size, arch, objectives.nll, verbose,
            optpol_linear, optpolicy.rw_linear, params=None, optimizer='adam', train_clip=False,
            trainset_size=trainset_size, log_fname=log_fname, alpha=alpha,
            noise_ave_times=ave_times,
        )
