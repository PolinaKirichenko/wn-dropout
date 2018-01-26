import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy
from scipy.stats import truncnorm


class Linear(nn.Linear):
    def __repr__(self):
        params = ''
        if self._parameters:
            params = ', '.join(list(self._parameters.keys()))
            params = '\t' + params
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' + str(self.out_features) + ')' + params


class WeightNorm(nn.Module):
    def __init__(self, in_features, out_features, p=None, noise_type=None, alpha=None, noise_magnitude=False, magn_var=None):
        super(WeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.noise_magnitude = noise_magnitude
        
        if noise_type is not None and noise_type not in ['bin', 'gaus', 'project', 'tangent', 'rotate-beta', 'rotate-norm']:
            raise ValueError('Unknown noise type: ' + str(noise_type))
        
        # noise can be 'bin', 'gaus', project', 'tangent', 'rotate-beta', 'rotate-norm' or None
        self.noise_type = noise_type
        # noise var for 'project', 'tangent' or 'rotate-norm'
        # shape of beta distribution for 'rotate-beta'
        self.alpha = alpha
        # dropout rate for 'bin'
        self.p = p
        # magnitude noise variance
        self.magn_var = magn_var
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).cuda())
        self.weight.data.normal_(0, 0.01)
        self.magnitude = nn.Parameter(torch.FloatTensor(out_features).cuda())
        self.magnitude.data.fill_(1)
        self.bias = nn.Parameter(torch.FloatTensor(out_features).cuda())
        self.bias.data.fill_(0)
        
        # noise for direction is sampled once for the whole mini-batch
        self.dir_noise = Variable(torch.FloatTensor(self.weight.data.shape).cuda(), requires_grad=False)
        self.dir_noise.data.fill_(0)

    def forward(self, input):
        # try dim=1 ?
        # output units, not input, are regularized
        # columns are weight vectors (with norm 1)
        W_norm = self.weight / self.weight.norm(dim=0)

        # no regularization or
        # not self.training (deterministic mode)
        if (self.noise_type is None and not self.noise_magnitude) or not self.training:
            return input.mm(W_norm * self.magnitude).add(self.bias)

        # self.training (non-deterministic mode) further
        # we ignore magnitude noise for these standard regularization techniques

        if self.noise_type == 'bin':
            sampled_for_data = Variable(torch.cuda.FloatTensor(input.size()))
            sampled_for_data.data.bernoulli_(1 - self.p)
            return torch.mm(input * sampled_for_data / (1 - self.p), W_norm * self.magnitude).add(self.bias)

        if self.noise_type == 'gaus':
            sampled_for_data = Variable(torch.cuda.FloatTensor(input.size()))
            sampled_for_data.data.normal_(1, np.sqrt(self.alpha))
            return torch.mm(input * sampled_for_data, W_norm * self.magnitude).add(self.bias)

        # only magnitude noise
        # sampled independently for objects in mini-batch
        if self.noise_type is None and self.noise_magnitude:
            bs = input.size(0)
            magnitude_noise = Variable(torch.cuda.FloatTensor(bs, self.out_features))
            magnitude_noise.data.normal_(1, np.sqrt(self.magn_var))
            preactivation = torch.mm(input, W_norm) * self.magnitude * magnitude_noise
            return preactivation.add(self.bias)

        # directional noise, or both directional and magnitude further

        self.dir_noise.data.normal_(0, np.sqrt(self.alpha))

        if self.noise_type == 'project':
            add_noise = self.dir_noise
        
        elif self.noise_type == 'tangent':
            coef = (W_norm * self.dir_noise).sum(dim=1)
            add_noise = self.dir_noise - W_norm * coef.view(-1, 1)
        
        elif self.noise_type == 'rotate-beta':
            coef = (W_norm * self.dir_noise).sum(dim=1)
            projected = self.dir_noise - W_norm * coef.view(-1, 1)
            noise_norm = projected / projected.norm(dim=0)

            eps = (np.random.beta(self.alpha, self.alpha, self.out_features) - 0.5) * (np.pi - 0.001)
            add_noise = noise_norm * Variable(torch.cuda.FloatTensor(np.tan(eps)))

        # 'rotate-norm'
        else:
            coef = (W_norm * self.dir_noise).sum(dim=1)
            projected = self.dir_noise - W_norm * coef.view(-1, 1)
            noise_norm = projected / projected.norm(dim=0)

            clip = (np.pi / 2) / np.sqrt(self.alpha)
            eps = truncnorm.rvs(-clip, clip, loc=0, scale=np.sqrt(self.alpha), size=self.out_features)
            add_noise = noise_norm * Variable(torch.cuda.FloatTensor(np.tan(eps)))

        W_noisy = W_norm + add_noise
        W_noisy_norm = W_noisy / W_noisy.norm(dim=0)

        if self.noise_magnitude:
            bs = input.size(0)
            magnitude_noise = Variable(torch.cuda.FloatTensor(bs, self.out_features))
            magnitude_noise.normal_(1, np.sqrt(self.magn_var))
            preactivation = torch.mm(input, W_noisy_norm) * self.magnitude * magnitude_noise
        else:
            preactivation = torch.mm(input, W_noisy_norm) * self.magnitude

        return preactivation.add(self.bias)

    
    def __repr__(self):
        params = ''
        if self._parameters:
            params = ', '.join(list(self._parameters.keys()))
            params = '\t' + params
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' + str(self.out_features) + ')' + \
             params + '\tnoise ' + str(self.noise_type) + '\tdir_alpha ' + str(self.alpha) + '\tmagn_var ' + str(self.magn_var)
