import torch
import torchvision.datasets as dsets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from data.mnist import NormMNIST


def load(dataset, batch_size, trainset_size=None):
    if dataset == 'mnist':
        return load_mnist(batch_size, trainset_size)

    raise Exception('Load of %s not implimented yet' % dataset)


def load_mnist(batch_size, trainset_size=None):
    train_dataset = NormMNIST(root='data/mnist',
                              train=True,
                              trainset_size=trainset_size,
                              download=True)

    mean_img = train_dataset.mean_img

    test_dataset = NormMNIST(root='data/mnist',
                             mean_img=mean_img,
                             train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Data Loader (Input Pipeline)
    x_train_shape = train_dataset.train_data.shape
    x_test_shape = test_dataset.test_data.shape

    return train_loader, test_loader, x_train_shape[0], x_test_shape[0], int(np.prod(train_dataset.train_data.shape[1:])), 10
