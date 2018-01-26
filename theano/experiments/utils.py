from __future__ import print_function

import os
import sys
import lasagne
import random
import datetime
import numpy as np

from nets import utils
from data import reader
from lasagne import layers as ll
from time import gmtime, strftime


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


def experiment_info(dataset, num_epochs, batch_size, arch, obj, optpolicy_lr, optpolicy_rw,
                    train_size, test_size, **params):
    info = [
        '\n', '=' * 80, '\n>> Experiment parameters:\n',
        'dataset:       ', dataset, '\n',
        'train size:    ', train_size, '\n',
        'test  size:    ', test_size, '\n',
        'num_epochs:    ', num_epochs, '\n',
        'batch_size:    ', batch_size, '\n',
        'arch:          ', arch.__name__, '\n',
        'objective:     ', obj.__name__, '\n',
        'optpolicy_lr:     ', optpolicy_lr.__name__, '\n',
        'optpolicy_rw:     ', optpolicy_rw.__name__, '\n',
        'Commandline:         ', ' '.join(sys.argv) + '\n']

    return ''.join([str(x) for x in info])


def run_experiment(dataset, num_epochs, batch_size, arch, obj, verbose,
                   optpolicy_lr, optpolicy_rw, log_fname=None, params=None,
                   train_clip=False, thresh=3, optimizer='adam', da=False, trainset_size=None,
                   alpha=None, noise_ave_times=0, updates_per_epoch=None):
    data, train_size, test_size, input_shape, nclass = reader.load(dataset, trainset_size)
    # k will be in filename for weights of this net
    if alpha:
        net, input_x, target_y, k = arch(input_shape, nclass, alpha=alpha)
    else:
        net, input_x, target_y, k = arch(input_shape, nclass)

    #if num_epochs == 0:
    #    return net

    if params is not None:
        ll.set_all_param_values(net, params)

    # Default log file name = experiment script file name
    if log_fname is None:
        log_fname = sys.argv[0].split('/')[-1][:-3]

    if not os.path.exists('./experiments/logs'):
        os.mkdir('./experiments/logs')

    # base_fname = './experiments/logs/{fname}-{dataset}-%s.txt'
    # print = get_logging_print(base_fname.format(dataset=dataset, fname=log_fname))
    base_fname = './experiments/logs/{fname}-%s.txt'
    print = get_logging_print(base_fname.format(fname=log_fname))
    print(experiment_info(**locals()))
    print(utils.net_configuration(net, short=(not verbose)))

    print('start compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))
    trainf, testf, up_opt, up_rw, get_output = utils.get_functions(**locals())
    print('finish compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))

    net, tr_info, te_info = utils.train(
        net, trainf, testf, get_output, up_opt, optpolicy_lr, up_rw, optpolicy_rw,
        data, num_epochs, batch_size, nclass, verbose, printf=print, thresh=thresh, da=da,
        noise_ave_times=noise_ave_times, updates_per_epoch=updates_per_epoch)

    print(save_net(net, dataset, k, log_fname))
    print(utils.test_net(testf, get_output, data, nclass, noise_ave_times))

    return net


def save_net(net, dataset, k, log_fname):
    params = ll.get_all_param_values(net)
    hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])

    if not os.path.exists('./experiments/weights'):
        os.mkdir('./experiments/weights')

    base_fname = './experiments/weights/{fname}-%s-%s.txt'.format(fname=log_fname.split('/')[-1])
    name = base_fname % (k, hash)
    print('save model: ' + name)
    np.save(name, params)
    return name + '.npy'


def apply_net(dataset, arch, log_fname, test_idx, params_file, alpha):
    data, train_size, test_size, input_shape, nclass = reader.load(dataset)
    # k will be in filename for weights of this net
    net, input_x, target_y, k = arch(input_shape, nclass, alpha=alpha)

    if params_file is not None:
        ll.set_all_param_values(net, np.load(params_file))

    print(utils.net_configuration(net, short=0))

    print('start compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))
    net_output = utils.get_output_score(net, input_x, target_y)
    print('finish compile', datetime.datetime.now().isoformat()[:16].replace('T', ' '))

    base_fname = './experiments/logs/{fname}.txt' 
    printf = get_logging_print(base_fname.format(fname=log_fname))
    utils.test_output(net_output, data, test_idx, alpha, printf)


def build_params_from_init(net, init_name, lsinit=-15, verbose=False):
    init_paramsv = list(np.load(init_name))
    params, paramsv, ardi = ll.get_all_params(net), [], 0

    for i in range(len(params)):
        if params[i].name in ['W', 'beta', 'gamma', 'mean', 'inv_std', 'b']:
            paramsv.append(init_paramsv[ardi])
            ardi += 1
        elif params[i].name == 'ls2':
            sh = paramsv[-1] if len(paramsv[-1].shape) == 4 else paramsv[-2]
            init_ = np.zeros_like(sh)
            paramsv.append(init_ + lsinit)
        else:
            raise Exception('wtf' + params[i].name)

        if verbose:
            print(params[i].name, paramsv[-1].shape)

    return paramsv
