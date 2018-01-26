#import cPickle as pickle
import pickle
import gzip
import os

import numpy as np
import theano
from scipy import linalg
from theano import tensor as T

from data.downloader import download_mnist#, download_cifar10, download_cifar100


def load(dataset, trainset_size=None, classes=None):
    if dataset == 'mnist':
        return load_mnist(trainset_size, classes)
    if dataset == 'mnist-random':
        return load_mnist_random()
    if dataset == 'cifar10':
        return load_cifar10()
    if dataset == 'cifar10-random':
        return load_cifar10_random()
    if dataset == 'cifar100':
        return load_cifar100()

    raise Exception('Load of %s not implimented yet' % dataset)


def maybe_get_less_data_and_classes(X_train, y_train, X_test, y_test, trainset_size, classes):
    train_idxs = np.arange(y_train.size)
    np.random.shuffle(train_idxs)
    X_train_ = X_train[train_idxs]
    y_train_ = y_train[train_idxs]
    X_test_ = X_test.copy()
    y_test_ = y_test.copy()

    # filter classes
    labels = np.arange(classes)
    X_train_ = X_train_[np.in1d(y_train_, labels)]
    y_train_ = y_train_[np.in1d(y_train_, labels)]
    X_test_ = X_test_[np.in1d(y_test_, labels)]
    y_test_ = y_test_[np.in1d(y_test_, labels)]

    # take less training data
    train_idxs = np.arange(y_train_.size)
    np.random.shuffle(train_idxs)
    train_idxs = train_idxs[:trainset_size]
    X_train_ = X_train_[train_idxs]
    y_train_ = y_train_[train_idxs]

    return X_train_, y_train_, X_test_, y_test_


def load_mnist(trainset_size=None, classes=None):
    """
    load_mnist taken from https://github.com/Lasagne/Lasagne/blob/master/examples/images.py
    :param base: base path to images dataset
    """
    if not trainset_size:
        trainset_size = 60000
    if not classes:
        classes = 10

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    base = './data/mnist'

    if not os.path.exists(base):
        download_mnist()

    # We can now download and read the training and test set image and labels.
    X_train = load_mnist_images(base + '/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(base + '/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(base + '/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(base + '/t10k-labels-idx1-ubyte.gz')

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    X_train, y_train, X_test, y_test = maybe_get_less_data_and_classes(
       X_train, y_train, X_test, y_test, trainset_size, classes
    )
    
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 1, 28, 28), np.unique(y_test).size

def load_mnist_random(base='./data/mnist'):
    X_train, y_train, X_test, y_test = load_mnist(base)[0]
    np.random.seed(74632)
    y_train = np.random.choice(10, len(y_train))
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 1, 28, 28), 10
