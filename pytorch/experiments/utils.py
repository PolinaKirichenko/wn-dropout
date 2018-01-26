from __future__ import print_function

import os
import sys
import random
import datetime
import numpy as np

from nets import utils
from data import reader
from time import gmtime, strftime
import torch
import torch.optim as optim


def get_logging_print(fname):
    cur_time = strftime("%m-%d_%H:%M:%S", gmtime())

    def prin(*args):
        str_to_write = ' '.join(map(str, args))
        filename = fname % cur_time if '%s' in fname else fname
        with open(filename, 'a') as f:
            f.write(str_to_write + '\n')
            f.flush()

        print(str_to_write)
        sys.stdout.flush()

    return prin


def experiment_info(dataset, num_epochs, batch_size, arch, criterion, optpolicy_lr,
                    train_size, test_size, **params):
    info = [
        '\n', '=' * 80, '\n>> Experiment parameters:\n',
        'dataset:       ', dataset, '\n',
        'train size:    ', train_size, '\n',
        'test  size:    ', test_size, '\n',
        'num_epochs:    ', num_epochs, '\n',
        'batch_size:    ', batch_size, '\n',
        'arch:          ', arch.__name__, '\n',
        'objective:     ', criterion, '\n',
        'optpolicy_lr:     ', optpolicy_lr.__name__, '\n',
        'Commandline:         ', ' '.join(sys.argv) + '\n']

    return ''.join([str(x) for x in info])


def run_experiment(dataset, num_epochs, batch_size, arch, criterion, verbose, optpolicy_lr, log_fname,
                   params=None, optimizer='adam', trainset_size=None, p=None,
                   noise_type=None, alpha=None, noise_magnitude=False, magn_var=None,
                   noise_ave_times=0, updates_per_epoch=None):
    train_loader, test_loader, train_size, test_size, input_size, nclass = reader.load(dataset, batch_size, trainset_size)
    
    if noise_type is not None or noise_magnitude:
        net = arch(input_size, nclass, p=p, noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var)
    else:
        net = arch(input_size, nclass)

    base_fname = './experiments/logs/{fname}-%s.txt'
    print = get_logging_print(base_fname.format(fname=log_fname))
    print(experiment_info(**locals()))
    print(">> Net Architecture")
    print(net)

    if optimizer == 'adam':
        optimizer_fn = optim.Adam(net.parameters())
    else:
        raise Exception('unknown optimizer:', optimizer)

    def up_opt(lr):
        for param_group in optimizer_fn.param_groups:
            param_group['lr'] = lr

    utils.train(net, train_loader, test_loader, train_size, num_epochs, batch_size, nclass, criterion, optimizer_fn, up_opt, optpolicy_lr,
                printf=print, noise_ave_times=noise_ave_times, updates_per_epoch=updates_per_epoch)

    print(save_net(net, dataset, log_fname))
    print(utils.test_net(net, train_loader, test_loader, nclass, noise_ave_times))

    return net


def save_net(net, dataset, log_fname):
    hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])

    if not os.path.exists('./experiments/weights'):
        os.mkdir('./experiments/weights')

    base_fname = './experiments/weights/{fname}-%s.txt'.format(fname=log_fname.split('/')[-1])
    name = base_fname % (hash)
    print('save model: ' + name)
    torch.save(net.state_dict(), name)
    return name
