from __future__ import print_function
from __future__ import division

import sys
import time
import random
import lasagne
import numpy as np
import theano

from nets.objectives import *
from tabulate import tabulate
import sklearn
from sklearn.metrics import accuracy_score

np.set_printoptions(precision=4, linewidth=150)


def get_output_score(net, input_x, target_y):
    predict_nondet = lasagne.layers.get_output(net, deterministic=False)
    accuracy_nondet = lasagne.objectives.categorical_accuracy(predict_nondet, target_y).mean()
    func = theano.function([input_x, target_y], [predict_nondet, accuracy_nondet])
    return func


def get_functions(net, obj, input_x, target_y, batch_size, train_size, test_size, optimizer='nesterov',
                  train_clip=False, thresh=3, **params):
    predict_nondet = lasagne.layers.get_output(net, deterministic=False)
    accuracy_nondet = lasagne.objectives.categorical_accuracy(predict_nondet, target_y).mean()

    predict_det = lasagne.layers.get_output(net, deterministic=True)
    accuracy_det = lasagne.objectives.categorical_accuracy(predict_det, target_y).mean()
    
    loss_train, rw = obj(predict_nondet, target_y, 0, 0, 0)
    nll_train = ell(predict_nondet, target_y)
    reg_train = rw*reg(net)

    loss_test, _ = obj(predict_det, target_y, net, batch_size=batch_size, num_samples=test_size, rw=rw, **params)
    nll_test = ell(predict_det, target_y)
    reg_test = rw*reg(net)

    weights = lasagne.layers.get_all_params(net, trainable=True)

    lr, beta = theano.shared(np.cast[theano.config.floatX](0)), theano.shared(np.cast[theano.config.floatX](0))
    if optimizer == 'nesterov':
        updates = lasagne.updates.nesterov_momentum(loss_train, weights, learning_rate=lr, momentum=beta)
    elif optimizer == 'adam':
        updates = lasagne.updates.adam(loss_train, weights, learning_rate=lr, beta1=beta)
    else:
        raise Exception('opt wtf')

    train_func = theano.function(
        [input_x, target_y], [loss_train, accuracy_nondet, accuracy_det],
        allow_input_downcast=True, updates=updates)
    test_func = theano.function(
        [input_x, target_y], [loss_test,  accuracy_nondet, accuracy_det],
        allow_input_downcast=True)
    get_output = theano.function([input_x], predict_nondet)

    def update_optimizer(new_lr, new_beta):
        lr.set_value(np.cast[theano.config.floatX](new_lr))
        beta.set_value(np.cast[theano.config.floatX](new_beta))

    def update_regweight(new_rw):
        rw.set_value(np.cast[theano.config.floatX](new_rw))

    return train_func, test_func, update_optimizer, update_regweight, get_output


def net_configuration(net, short=False):
    if short:
        nl = net.input_layer.nonlinearity.func_name if hasattr(net.input_layer, 'nonlinearity') else 'linear'
        return "%s, %s, %s:" % (net.input_layer.name, net.input_layer.input_layer.output_shape, nl)

    table = []
    header = ['Layer', 'output_shape', 'parameters', 'nonlinearity']
    while hasattr(net, 'input_layer'):
        if hasattr(net, 'nonlinearity') and hasattr(net.nonlinearity, 'func_name'):
            nl = net.nonlinearity.func_name
        else:
            nl = 'linear'

        if net.name is not None:
            table.append((net.name, net.output_shape, net.params.keys(), nl))
        else:
            table.append((str(net.__class__).split('.')[-1][:-2],
                          net.output_shape, net.params.keys(), nl))
        net = net.input_layer

    if hasattr(net, 'nonlinearity') and hasattr(net.nonlinearity, 'func_name'):
        nl = net.nonlinearity.func_name
    else:
        nl = 'linear'
    table.append((net.name, net.output_shape, net.params.keys(), nl))

    return ">> Net Architecture\n" + tabulate(reversed(table), header, floatfmt=u'.3f') + '\n'


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def batch_iterator_train_crop_flip(data, y, batchsize, shuffle=False):
    PIXELS = 28
    PAD_CROP = 4
    n_samples = data.shape[0]
    # Shuffles indicies of training data, so we can draw batches from random indicies instead of shuffling whole data
    indx = np.random.permutation(range(n_samples))
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[indx[sl]]
        y_batch = y[indx[sl]]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # flip left-right choice
        flip_lr = random.randint(0,1)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                # pad and crop images
                img_pad = np.pad(
                    X_batch_aug[j, k], pad_width=((PAD_CROP, PAD_CROP), (PAD_CROP, PAD_CROP)), mode='constant')
                X_batch_aug[j, k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # flip left-right if chosen
                if flip_lr == 1:
                    X_batch_aug[j, k] = np.fliplr(X_batch_aug[j,k])

        # fit model on each batch
        yield X_batch_aug, y_batch


def iter_info(verbose, epoch, start_time, num_epochs, updates, train_info, test_info,
              train_acc_noise_ave, test_acc_noise_ave, printf, net, optpolicy_lr, optpolicy_rw,
              thresh=3, **params):
    if verbose and epoch % verbose == 0:
        train_loss, train_acc_noise_one, train_acc_det = train_info[-1]
        test_loss,  test_acc_noise_one, test_acc_det  = test_info[-1]
        epoch_time, start_time = int(time.time() - start_time), time.time()


        ard_layers = map(lambda l: l.get_ard(thresh=thresh) if 'reg' in l.__dict__ else None,
                 lasagne.layers.get_all_layers(net))

        he = ['epo', 'upd', 'tr_loss', 'tr_acc_noise_one',
              'tr_acc_noise_ave', 'tr_acc_det',
              'te_loss', 'te_acc_noise_one', 'te_acc_noise_ave',
              'te_acc_det', 'lr', 'sec']
        info = ('%s/%s' % (str(epoch).zfill(3) , num_epochs),
                updates, '\'%.3f' % train_loss,
                '\'%.3f' % train_acc_noise_one,
                '\'%.3f' % train_acc_noise_ave,
                '\'%.3f' % train_acc_det,
                '\'%.3f' % test_loss,
                '\'%.3f' % test_acc_noise_one,
                '\'%.3f' % test_acc_noise_ave,
                '\'%.3f' % test_acc_det,
                optpolicy_lr(epoch)[0],
                epoch_time)

        if epoch == 0:
            printf(">> Start Learning")
            printf(tabulate([info], he, floatfmt='1.1e'))
        else:
            printf(tabulate([info], he, tablefmt="plain", floatfmt='1.1e').split('\n')[1])

    return start_time


def train(net, train_fun, test_fun, get_output, up_opt, optpolicy_lr, up_rw, optpolicy_rw, data, num_epochs, batch_size,
          nclass, verbose=1, printf=print, thresh=3, da=False, noise_ave_times=0, updates_per_epoch=None):
    """
    da: whether to perform data augmentation
    """
    sys.stdout.flush()
    train_info, test_info = [], []
    # updates: how many updates were performed during learning overall
    start_time, updates = time.time(), 0
    X_train, y_train, X_test, y_test = data

    if not updates_per_epoch:
        updates_per_epoch = int(X_train.shape[0] / batch_size)

    try:
        for epoch in range(num_epochs):
            up_opt(*optpolicy_lr(epoch))
            up_rw(optpolicy_rw(epoch))

            itera = batch_iterator_train_crop_flip if da else iterate_minibatches
            
            batches, info = 0, np.zeros(3)

            while batches < updates_per_epoch:
                for inputs, targets in itera(X_train, y_train, batch_size, shuffle=True):
                    info += train_fun(inputs, targets)
                    batches += 1
                    updates += 1
                    if batches == updates_per_epoch:
                        break

            train_info.append(info/batches)

            batches, info = 0, np.zeros(3)
            for inputs, targets in itera(X_test, y_test, batch_size, shuffle=False):
                info += test_fun(inputs, targets)
                batches += 1

            test_info.append(info/batches)
            
            train_acc_noise_ave = 0
            test_acc_noise_ave = 0
            if noise_ave_times > 0:
                train_output = np.zeros((X_train.shape[0], nclass))
                test_output = np.zeros((X_test.shape[0], nclass))
                for i in range(noise_ave_times):
                    train_output += get_output(X_train)
                    test_output += get_output(X_test)
                train_acc_noise_ave = accuracy_score(y_train, np.argmax(train_output, 1))
                test_acc_noise_ave = accuracy_score(y_test, np.argmax(test_output, 1))

            start_time = iter_info(**locals())

    except KeyboardInterrupt:
        print('stop train')

    return net, train_info, test_info


def test_output(net_output, data, test_idx, alpha, printf=print):
    X_train, y_train, X_test, y_test = data
    #_, score = net_output(X_test, y_test)
    # bad_idxs = []
    # for i in range(len(y_test)):
    #     _, score_ = net_output(X_test[i:i+1], y_test[i:i+1])
    #     if score_ < 1:
    #         bad_idxs.append(i)
    #         if len(bad_idxs) == 20:
    #             break         

    one_x = X_test[test_idx:test_idx+1]
    one_y = y_test[test_idx:test_idx+1]

    probs = []
    for i in range(50):
        out, sc = net_output(one_x, one_y)
        probs.append(out)

    probs = np.array(probs)
    mean = probs.mean(axis=0)
    std = probs.std(axis=0)

    printf(np.array_str(mean), np.array_str(std))


def test_net(test_fn, get_output, data, nclass, noise_ave_times=0):
    X_train, y_train, X_test, y_test = data

    batches = 0
    train_info = np.zeros(3)
    train_acc_noise_ave = 0
    for inputs, targets in iterate_minibatches(X_train, y_train, 100, shuffle=False):
        train_info += test_fn(inputs, targets)
        batches += 1

        if noise_ave_times > 0:
            output = np.zeros((inputs.shape[0], nclass))
            for i in range(noise_ave_times):
                output += get_output(inputs)
            train_acc_noise_ave += accuracy_score(targets, np.argmax(output, 1))
    train_info /= batches
    train_acc_noise_ave /= batches
    
    batches = 0
    test_info = np.zeros(3)
    test_acc_noise_ave = 0
    for inputs, targets in iterate_minibatches(X_test, y_test, 100, shuffle=False):
        test_info += test_fn(inputs, targets)
        batches += 1

        if noise_ave_times > 0:
            output = np.zeros((inputs.shape[0], nclass))
            for i in range(noise_ave_times):
                output += get_output(inputs)
            test_acc_noise_ave += accuracy_score(targets, np.argmax(output, 1))
    test_info /= batches
    test_acc_noise_ave /= batches

    return 'tr_det: %s tr_noise_one: %s tr_noise_ave: %s\nte_det: %s te_noise_one: %s te_noise_ave: %s' % (
        train_info[2], train_info[1], train_acc_noise_ave, test_info[2], test_info[1], test_acc_noise_ave
        )
