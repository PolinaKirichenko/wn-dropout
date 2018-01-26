import theano.tensor as T
from lasagne import nonlinearities, updates
from lasagne import init
from lasagne.random import get_rng
from lasagne.nonlinearities import rectify, identity
from lasagne.layers import Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import numpy as np


class DenseLayer(Layer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify,
                 mnc=False, b=init.Constant(0.), **kwargs):
        super(DenseLayer, self).__init__(incoming)
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        # what is srng? 
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.W = self.add_param(Wfc, (self.num_inputs, self.num_units), name="W")
        # max norm constraint
        if mnc:
            self.W = updates.norm_constraint(self.W, mnc)

        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        return self.get_output_for_(input, deterministic, **kwargs)

    def get_output_for_(self, input, deterministic, **kwargs):
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)
    
    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return None


class DenseBinaryDropOut(DenseLayer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify, p=0.5, **kwargs):
        super(DenseBinaryDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, mnc=3, **kwargs)
        self.p = p
        self.reg = True
        self.num_updates = 0
        self.name = 'BDropOut'

    def get_output_for_(self, input, deterministic, **kwargs):
        input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape

        if not (deterministic or self.p == 0):
            input /= (1 - self.p)
            input *= self._srng.binomial(input_shape, p=1 - self.p, dtype=input.dtype)

        return self.nonlinearity(T.dot(input, self.W) + self.b)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return str(self.p / (1 - self.p))


class DenseGausDropOut(DenseLayer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify, alpha=1, **kwargs):
        super(DenseBinaryDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, mnc=3, **kwargs)
        # alpha is the variance of Gaussian noise
        self.alpha = alpha
        self.reg = True
        self.num_updates = 0
        self.name = 'GausDropOut'

    def get_output_for_(self, input, deterministic, **kwargs):
        input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape

        if not (deterministic or self.alpha == 0):
            input *= self._srng.normal(input_shape, avg=1, std=np.sqrt(self.alpha), dtype=input.dtype)

        return self.nonlinearity(T.dot(input, self.W) + self.b)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return str(self.alpha)
    

class WeightNormLayer(Layer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify,
                 mnc=False, g=init.Constant(1.), b=init.Constant(0.), **kwargs):
        super(WeightNormLayer, self).__init__(incoming)
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        
        self.W_norm = self.add_param(Wfc, (self.num_inputs, self.num_units), name="W_norm")
        self.g = self.add_param(g, (self.num_units, ), name="g")
        self.b = self.add_param(b, (self.num_units, ), name="b", regularizable=False)
        
        W_axes_to_sum = 0
        W_dimshuffle_args = ['x', 0]
        
        self.W = self.W_norm * (
            self.g / T.sqrt(T.sum(T.square(self.W_norm), axis=W_axes_to_sum))
        )

        # max norm constraint
        if mnc:
            self.W = updates.norm_constraint(self.W, mnc)
        

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        return self.get_output_for_(input, deterministic, **kwargs)

    def get_output_for_(self, input, deterministic, **kwargs):
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)
    
    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return None
    

class WeightNormBinaryDropOut(WeightNormLayer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify, p=0.5, **kwargs):
        super(WeightNormBinaryDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, **kwargs)
        self.p = p
        self.reg = True
        self.num_updates = 0
        self.name = 'WNBDropOut'        

    def get_output_for_(self, input, deterministic, **kwargs):
        # input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape
        input_shape = (self.num_inputs, )

        if not (deterministic or self.p == 0):
            input /= (1 - self.p)
            input *= self._srng.binomial(input_shape, p=1 - self.p, dtype=input.dtype)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return str(self.p / (1 - self.p))


class WeightNormGausDropOut(WeightNormLayer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify, alpha=1, **kwargs):
        """alpha: variance of gaussian noise"""
        super(WeightNormGausDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, **kwargs)
        self.alpha = alpha
        self.reg = True
        self.num_updates = 0
        self.name = 'WNGausDropOut'

    def get_output_for_(self, input, deterministic, **kwargs):
        input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape

        if not (deterministic or self.alpha == 0):
            input *= self._srng.normal(input_shape, avg=1, std=np.sqrt(self.alpha), dtype=input.dtype)

        return self.nonlinearity(T.dot(input, self.W) + self.b)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return str(self.alpha)
    
    
class WeightNormProjectedDropOut(WeightNormLayer):
    def __init__(self, incoming, num_units, Wfc=init.Normal(), nonlinearity=rectify, alpha=0.01, method=='gaus', **kwargs):
        """alpha: scalar, covariance matrix of the normal distribution for weight noise is alpha * I"""
        super(WeightNormProjectedDropOut, self).__init__(incoming, num_units, Wfc, nonlinearity, **kwargs)
        self.alpha = alpha
        self.reg = True
        self.num_updates = 0
        if method == 'gaus' or method == 'tangent' or method == 'rot':
            raise('Unknown directional dropout', self.method)
        self.method = method
        self.name = 'WNProjDropOut'

    def perturb_on_sphere(self, W_norm):
        # column vectors are weight vectors for units
        W_axes_to_sum = 0

        sampled = self._srng.normal((self.num_inputs, self.num_units), avg=0, std=np.sqrt(self.alpha))

        if method == 'tangent':
            coef = T.sum(W_norm * sampled, axis=1)
            sampled -= W_norm * coef[:, None]
        
        # This way W_noisy[:, i] will be sampled from multivariate Gaussian centered at W[:, i]
        W_noisy = W_norm + sampled
        W_noisy = W_noisy / T.sqrt(T.sum(T.square(W_noisy), axis=W_axes_to_sum))

        return W_noisy

    def rotate(self, W_norm):
        pass


    def get_output_for(self, input, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        return self.get_output_for_(input, deterministic, **kwargs)

    def get_output_for_(self, input, deterministic, **kwargs):
        input_shape = input.shape if any(s is None for s in self.input_shape) else self.input_shape
        W_axes_to_sum = 0

        if (self.alpha == 0 or deterministic):
            W_norm = self.W_norm / T.sqrt(T.sum(T.square(self.W_norm), axis=W_axes_to_sum))
            activation = T.dot(input, self.g * W_norm) + self.b
        else:
            # inject noise into weights
            W_norm_old = self.W / T.sqrt(T.sum(T.square(self.W), axis=W_axes_to_sum))
            W_norm_new = self.perturb_on_sphere(W_norm_old)
            activation = T.dot(input, self.g * W_norm_new) + self.b

        return self.nonlinearity(activation)

    def eval_reg(self, **kwargs):
        return 0

    def get_ard(self, **kwargs):
        return None

    def get_reg(self):
        return None
