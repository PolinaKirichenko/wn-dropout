import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from nets.layers import Linear, WeightNorm


# standard parametrization

class FC_Net(nn.Module):
    def __init__(self, input_size, num_classes, nonlinearity=nn.ReLU()):
        super(FC_Net, self).__init__()
        self.nonlinearity = nonlinearity

        self.fc1 = Linear(input_size, 1024)
        self.fc1.weight.data.normal_(mean=0, std=0.01)
        
        self.fc2 = Linear(1024, 1024)
        self.fc2.weight.data.normal_(mean=0, std=0.01)
        
        self.fc3 = Linear(1024, 2048)
        self.fc3.weight.data.normal_(mean=0, std=0.01)
        
        self.fc4 = Linear(2048, num_classes) 
        self.fc4.weight.data.normal_(mean=0, std=0.01)
    
    def forward(self, x):
        out = self.nonlinearity(self.fc1(x))
        out = self.nonlinearity(self.fc2(out))
        out = self.nonlinearity(self.fc3(out))
        out = self.fc4(out)
        return out


class FC_BinDO_Net(nn.Module):
    def __init__(self, input_size, num_classes, p=0.5, nonlinearity=nn.ReLU()):
        super(FC_BinDO_Net, self).__init__()
        self.nonlinearity = nonlinearity

        self.fc1_drop = nn.Dropout(p=p)
        self.fc1 = Linear(input_size, 1024)
        self.fc1.weight.data.normal_(mean=0, std=0.01)
        
        self.fc2_drop = nn.Dropout(p=p)
        self.fc2 = Linear(1024, 1024)
        self.fc2.weight.data.normal_(mean=0, std=0.01)
        
        self.fc3_drop = nn.Dropout(p=p)
        self.fc3 = Linear(1024, 2048)
        self.fc3.weight.data.normal_(mean=0, std=0.01)
        
        self.fc4_drop = nn.Dropout(p=p)
        self.fc4 = Linear(2048, num_classes) 
        self.fc4.weight.data.normal_(mean=0, std=0.01)
    
    def forward(self, x):
        out = self.fc1_drop(x)
        out = self.nonlinearity(self.fc1(out))

        out = self.fc2_drop(out)
        out = self.nonlinearity(self.fc2(out))

        out = self.fc3_drop(out)
        out = self.nonlinearity(self.fc3(out))

        out = self.fc4_drop(out)
        out = self.fc4(out)
        return out


# weight normalization parametrization

class WN_Net(nn.Module):
    def __init__(self, input_size, num_classes, nonlinearity=nn.ReLU(),
                 p=None, noise_type=None, alpha=None, noise_magnitude=False, magn_var=None):
        super(WN_Net, self).__init__()
        self.nonlinearity = nonlinearity

        self.fc1 = WeightNorm(input_size, 1024, p=p,
                              noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var)
        self.fc2 = WeightNorm(1024, 1024, p=p,
                              noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var)
        self.fc3 = WeightNorm(1024, 2048, p=p,
                              noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var)
        self.fc4 = WeightNorm(2048, num_classes, p=p,
                              noise_type=noise_type, alpha=alpha, noise_magnitude=noise_magnitude, magn_var=magn_var)

    def forward(self, x):
        out = self.nonlinearity(self.fc1(x))
        out = self.nonlinearity(self.fc2(out))
        out = self.nonlinearity(self.fc3(out))
        out = self.fc4(out)
        return out


class WN_BinDO_Net(nn.Module):
    def __init__(self, input_size, num_classes, p=0.5, nonlinearity=nn.ReLU(), alpha=0.01):
        super(WN_BinDO_Net, self).__init__()
        self.nonlinearity = nonlinearity

        self.fc1_drop = nn.Dropout(p=p)
        self.fc1 = WeightNorm(input_size, 1024, noise_type=None, alpha=None, p=None)
        
        self.fc2_drop = nn.Dropout(p=p)
        self.fc2 = WeightNorm(1024, 1024, noise_type=None, alpha=None, p=None)
        
        self.fc3_drop = nn.Dropout(p=p)
        self.fc3 = WeightNorm(1024, 2048, noise_type=None, alpha=None, p=None)

        self.fc4_drop = nn.Dropout(p=p)
        self.fc4 = WeightNorm(2048, num_classes, noise_type=None, alpha=None, p=None)

    def forward(self, x):
        out = self.fc1_drop(x)
        out = self.nonlinearity(self.fc1(out))

        out = self.fc2_drop(out)
        out = self.nonlinearity(self.fc2(out))

        out = self.fc3_drop(out)
        out = self.nonlinearity(self.fc3(out))

        out = self.fc4_drop(out)
        out = self.fc4(out)
        return out

