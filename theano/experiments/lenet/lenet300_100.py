from __future__ import print_function

import sys
import warnings
from nets import objectives
from nets import optpolicy, layers
from theano import tensor as T
from lasagne import init, nonlinearities as nl, layers as ll
from experiments.utils import run_experiment, apply_net
import numpy as np
import os

warnings.simplefilter("ignore")

def net_lenet5(input_shape, nclass):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.DenseLayer(net, 1024, Wfc=init.Normal())
    net = layers.DenseLayer(net, 1024, Wfc=init.Normal())
    net = layers.DenseLayer(net, 2048, Wfc=init.Normal())
    net = layers.DenseLayer(net, nclass, Wfc=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, 1


def net_lenet5_do(input_shape, nclass):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.DenseBinaryDropOut(net, 1024, Wfc=init.Normal(), p=0.5)
    net = layers.DenseBinaryDropOut(net, 1024, Wfc=init.Normal(), p=0.5)
    net = layers.DenseBinaryDropOut(net, 2048, Wfc=init.Normal(), p=0.5)
    net = layers.DenseBinaryDropOut(net, nclass, Wfc=init.Normal(), p=0.5, nonlinearity=nl.softmax)

    return net, input_x, target_y, 2


def net_lenet5_wn(input_shape, nclass):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.WeightNormLayer(net, 1024, Wfc=init.Normal())
    net = layers.WeightNormLayer(net, 1024, Wfc=init.Normal())
    net = layers.WeightNormLayer(net, 2048, Wfc=init.Normal())
    net = layers.WeightNormLayer(net, nclass, Wfc=init.Normal(), nonlinearity=nl.softmax)

    return net, input_x, target_y, 3


def net_lenet5_wn_do(input_shape, nclass):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.WeightNormBinaryDropOut(net, 1024, Wfc=init.Normal(), p=0.5)
    net = layers.WeightNormBinaryDropOut(net, 1024, Wfc=init.Normal(), p=0.5)
    net = layers.WeightNormBinaryDropOut(net, 2048, Wfc=init.Normal(), p=0.5)
    net = layers.WeightNormBinaryDropOut(net, nclass, Wfc=init.Normal(), p=0.5, nonlinearity=nl.softmax)

    return net, input_x, target_y, 4


def net_lenet5_wn_proj_do(input_shape, nclass, alpha):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.WeightNormProjectedDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormProjectedDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormProjectedDropOut(net, 2048, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormProjectedDropOut(net, nclass, Wfc=init.Normal(), alpha=alpha, nonlinearity=nl.softmax)

    return net, input_x, target_y, 5

def net_lenet5_wn_proj_do_tangent(input_shape, nclass, alpha):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.WeightNormProjectedDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha, tangent=True)
    net = layers.WeightNormProjectedDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha, tangent=True)
    net = layers.WeightNormProjectedDropOut(net, 2048, Wfc=init.Normal(), alpha=alpha, tangent=True)
    net = layers.WeightNormProjectedDropOut(net, nclass, Wfc=init.Normal(), alpha=alpha, tangent=True, nonlinearity=nl.softmax)

    return net, input_x, target_y, 5

def net_lenet5_wn_gaus(input_shape, nclass, alpha=1):
    # tensor 4 because (n objects, n channels, pic size, pic size)
    input_x, target_y, Winit = T.tensor4("input"), T.vector("target", dtype='int32'), init.Normal()

    net = ll.InputLayer(input_shape, input_x)
    net = layers.WeightNormGausDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormGausDropOut(net, 1024, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormGausDropOut(net, 2048, Wfc=init.Normal(), alpha=alpha)
    net = layers.WeightNormGausDropOut(net, nclass, Wfc=init.Normal(), alpha=alpha, nonlinearity=nl.softmax)

    return net, input_x, target_y, 6

