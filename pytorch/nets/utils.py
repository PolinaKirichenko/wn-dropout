from __future__ import print_function
from __future__ import division
import sys
import time
import numpy as np
from tabulate import tabulate
import sklearn
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable


class AccCounter:
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        self.__n_objects = 0
        self.__sum = 0


def to_np(x):
    return x.data.cpu().numpy()


def iter_info(verbose, epoch, start_time, num_epochs, updates, train_info, test_info,
              printf, net, optpolicy_lr, thresh=3, **params):
    if verbose and epoch % verbose == 0:
        train_loss, train_acc_det, train_acc_nondet_1, train_acc_nondet_ave = train_info[-1]
        test_loss, test_acc_det, test_acc_nondet_1, test_acc_nondet_ave  = test_info[-1]
        epoch_time, start_time = int(time.time() - start_time), time.time()

        he = ['epo', 'upd', 'tr_loss',
              'tr_acc_det', 'tr_acc_nondet_1', 'tr_acc_nondet_ave', 
              'te_loss', 'te_acc_det', 'te_acc_nondet_1', 'te_acc_nondet_ave',
              'lr', 'sec']
        info = ('%s/%s' % (str(epoch).zfill(3) , num_epochs),
                updates, '\'%.3f' % train_loss,
                '\'%.3f' % train_acc_det,
                '\'%.3f' % train_acc_nondet_1,
                '\'%.3f' % train_acc_nondet_ave,
                '\'%.3f' % test_loss,
                '\'%.3f' % test_acc_det,
                '\'%.3f' % test_acc_nondet_1,
                '\'%.3f' % test_acc_nondet_ave,
                optpolicy_lr(epoch),
                epoch_time)

        if epoch == 0:
            printf(">> Start Learning")
            printf(tabulate([info], he, floatfmt='1.1e'))
        else:
            printf(tabulate([info], he, tablefmt="plain", floatfmt='1.1e').split('\n')[1])

    return start_time


def train(net, train_loader, test_loader, train_size, num_epochs, batch_size, nclass, criterion, optimizer, up_opt, optpolicy_lr,
          verbose=1, printf=print, noise_ave_times=0, updates_per_epoch=None):
    sys.stdout.flush()
    train_info, test_info = [], []
    # updates: how many updates were performed during learning overall
    start_time, updates = time.time(), 0

    if not updates_per_epoch:
        updates_per_epoch = int(train_size / batch_size)

    counter = AccCounter()

    test_size = test_loader.dataset.test_data.shape[0]
    X_train = Variable(train_loader.dataset.train_data.cuda())
    X_test = Variable(test_loader.dataset.test_data.cuda())
    y_train = Variable(train_loader.dataset.train_labels.cuda())
    y_test = Variable(test_loader.dataset.test_labels.cuda())

    net.cuda()
    try:
        for epoch in range(num_epochs):

            up_opt(optpolicy_lr(epoch))
            batches, train_loss = 0, 0

            # non-deterministic mode
            net.train()
            while batches < updates_per_epoch:
                for inputs, labels in train_loader:
                    # Convert torch tensor to Variable
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                    # zero the gradient buffer
                    optimizer.zero_grad()  

                    # Forward + Backward + Optimize
                    outputs = net(inputs)
                    counter.add(to_np(outputs), to_np(labels))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += to_np(loss)[0] * float(batch_size)

                    batches += 1
                    updates += 1
                    if batches == updates_per_epoch:
                        break
            
            train_acc_nondet_1 = counter.acc()
            counter.flush()

            # TODO: check what works faster: loaders or applying net to the whole set
            # this is no good for big datasets, it can cause memory problems
            output_test_nondet = net(X_test)
            y_test_nondet_1 = np.argmax(to_np(output_test_nondet), axis=1)
            test_acc_nondet_1 = accuracy_score(to_np(y_test), y_test_nondet_1)

            # averaged from noise_ave_times random models

            # if not feed whole dataste to net, but use train_loader/test_loader instead,
            # it is not clear how to fix model's weight for all batches (test one random model)
            train_acc_nondet_ave = 0
            test_acc_nondet_ave = 0
            if noise_ave_times > 0:
                train_output = np.zeros((train_size, nclass))
                test_output = np.zeros((test_size, nclass))
                for i in range(noise_ave_times):
                    train_output += to_np(net(X_train))
                    test_output += to_np(net(X_test))
                train_acc_nondet_ave = accuracy_score(to_np(y_train), np.argmax(train_output, axis=1))
                test_acc_nondet_ave = accuracy_score(to_np(y_test), np.argmax(test_output, axis=1))

            # deterministic mode
            net.eval()
            output_train_det = net(X_train)
            output_test_det = net(X_test)
            # test loss in det mode
            test_loss = to_np(criterion(output_test_det, y_test))[0]
            y_train_det = np.argmax(to_np(output_train_det), axis=1)
            y_test_det = np.argmax(to_np(output_test_det), axis=1)
            train_acc_det = accuracy_score(to_np(y_train), y_train_det)
            test_acc_det = accuracy_score(to_np(y_test), y_test_det)

            train_info.append([train_loss, train_acc_det, train_acc_nondet_1, train_acc_nondet_ave])
            test_info.append([test_loss, test_acc_det, test_acc_nondet_1, test_acc_nondet_ave])

            start_time = iter_info(**locals())

    except KeyboardInterrupt:
        print('stop train')

    return net


def test_net(net, train_loader, test_loader, nclass, noise_ave_times=0):
    X_train = Variable(train_loader.dataset.train_data.cuda())
    X_test = Variable(test_loader.dataset.test_data.cuda())
    y_train = to_np(Variable(train_loader.dataset.train_labels))
    y_test = to_np(Variable(test_loader.dataset.test_labels))

    train_size = X_train.size()[0]
    test_size = X_test.size()[0]

    # stichastc mode
    net.train()
    output_train_nondet = net(X_train)
    y_train_nondet_1 = np.argmax(to_np(output_train_nondet), axis=1)
    train_acc_nondet_1 = accuracy_score(y_train, y_train_nondet_1)

    output_test_nondet = net(X_test)
    y_test_nondet_1 = np.argmax(to_np(output_test_nondet), axis=1)
    test_acc_nondet_1 = accuracy_score(y_test, y_test_nondet_1)

    # averaged from noise_ave_times random models

    # if not feed whole dataste to net, but use train_loader/test_loader instead,
    # it is not clear how to fix model's weight for all batches (test one random model)
    train_acc_nondet_ave = 0
    test_acc_nondet_ave = 0
    if noise_ave_times > 0:
        train_output = np.zeros((train_size, nclass))
        test_output = np.zeros((test_size, nclass))
        for i in range(noise_ave_times):
            train_output += to_np(net(X_train))
            test_output += to_np(net(X_test))
        train_acc_nondet_ave = accuracy_score(y_train, np.argmax(train_output, axis=1))
        test_acc_nondet_ave = accuracy_score(y_test, np.argmax(test_output, axis=1))

    # deterministic mode
    net.eval()
    output_train_det = net(X_train)
    output_test_det = net(X_test)
    y_train_det = np.argmax(to_np(output_train_det), axis=1)
    y_test_det = np.argmax(to_np(output_test_det), axis=1)
    train_acc_det = accuracy_score(y_train, y_train_det)
    test_acc_det = accuracy_score(y_test, y_test_det)

    return 'tr_det: %s tr_nondet_1: %s tr_nondet_ave: %s\nte_det: %s te_nondet_1: %s te_nondet_ave: %s' % (
        train_acc_det, train_acc_nondet_1, train_acc_nondet_ave, test_acc_det, test_acc_nondet_1, test_acc_nondet_ave,
        )

