import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import truncnorm
import time
# Hyper Parameters
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.CIFAR10(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.CIFAR10(root='../data',
                           train=False,
                           transform=transforms.ToTensor())


train_data = torch.transpose(torch.FloatTensor(train_dataset.train_data).contiguous(), 1, 3)
test_data = torch.transpose(torch.FloatTensor(test_dataset.test_data).contiguous(), 1, 3)
mean = train_data.mean(dim=0)
train_data = ((train_data - mean) / 256.).cuda()
train_labels = torch.LongTensor(train_dataset.train_labels).cuda()
test_data = ((test_data - mean) / 256.).cuda()
test_labels = torch.LongTensor(test_dataset.test_labels).cuda()
import os

my_dir = 'convgauss'
if not os.path.exists(my_dir):
    os.makedirs(my_dir)

data_sizes = [10000, 50000]


def compute_forward(net, data, deterministic):
    output = np.zeros((data.shape[0], 10))
    for i in range(data.shape[0]//1000):
        l = 1000*i
        r = min(1000*(i+1), data.shape[0])
        output[l:r] = net.forward(data[l:r], deterministic=deterministic).data.cpu().numpy()
    return output

class WeightNormFC:
    def __init__(self, in_features, out_features, alpha=0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.internal_weight = Variable(torch.FloatTensor(in_features, out_features).cuda(), requires_grad=True)
        self.internal_weight.data.normal_(0, 0.01)
        self.magnitude = Variable(torch.FloatTensor(out_features).cuda(), requires_grad=True)
        self.magnitude.data.fill_(1)
        self.bias = Variable(torch.FloatTensor(out_features).cuda(), requires_grad=True)
        self.bias.data.fill_(0)
        self.sampled = Variable(torch.FloatTensor(self.internal_weight.data.shape).cuda(), requires_grad=False)
        self.sampled.data.fill_(0)
        self.sampled_inputs = []
        for size in [100, 1000]:
            self.sampled_inputs.append(Variable(torch.FloatTensor(np.zeros((size, in_features))).cuda(), requires_grad=False))
            self.sampled_inputs[-1].data.fill_(0)
        self.parameters = [self.internal_weight, self.magnitude, self.bias]

    def forward(self, input, deterministic=False, tangent=False,
                super_sampling_beta=False, super_sampling_truncn=False, gauss=False, gauss_objects=True):
        if gauss:
            self.sampled.data.normal_(1, np.sqrt(self.alpha))
            W_noisy = self.internal_weight * self.sampled
            result = input.mm(W_noisy)
            return result.add(self.bias)
        elif gauss_objects:
            size = input.shape[0]
            for i in range(len(self.sampled_inputs)):
                if self.sampled_inputs[i].shape[0] == size:
                    self.sampled_inputs[i].data.normal_(1, np.sqrt(self.alpha))
                    noised_input = input * self.sampled_inputs[i]

                    W_norm = self.internal_weight / self.internal_weight.norm(dim=0)
                    self.weight = (self.magnitude * W_norm)
                    result = noised_input.mm(self.weight)
                    return result.add(self.bias)

        W_norm = self.internal_weight / self.internal_weight.norm(dim=0)
        if not deterministic:
            self.sampled.data.normal_(0, np.sqrt(self.alpha))
            if tangent:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
            elif super_sampling_beta:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
                projected = projected / projected.norm(dim=0)
                tt = time.time()
                epsilon = (np.random.beta(self.alpha, self.alpha, self.out_features) - 0.5) * (np.pi-0.001)
                projected = projected * Variable(torch.cuda.FloatTensor(np.tan(epsilon)))
            elif super_sampling_truncn:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
                projected = projected / projected.norm(dim=0)
                alpha = (np.pi / 2) / self.alpha
                epsilon = truncnorm.rvs(-alpha, alpha, scale=np.pi / (2 * alpha), size=self.out_features)
                projected = projected * Variable(torch.cuda.FloatTensor(np.tan(epsilon)))
            else:
                projected = self.sampled

            W_noisy = W_norm + projected
            W_noisy_normed = W_noisy / W_noisy.norm(dim=0)
        else:
            W_noisy_normed = W_norm
        self.weight = (self.magnitude * W_noisy_normed)
        result = input.mm(self.weight)
        return result.add(self.bias)

class WeightNormConv:
    def __init__(self, in_features, out_features, pixels_in=32, alpha=0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.internal_weight = Variable(torch.FloatTensor(out_features, in_features, 3, 3).cuda(), requires_grad=True)
        self.internal_weight.data.normal_(0, 0.01)
        self.magnitude = Variable(torch.FloatTensor(out_features, 1, 1, 1).cuda(), requires_grad=True)
        self.magnitude.data.fill_(1)
        self.bias = Variable(torch.FloatTensor(out_features).cuda(), requires_grad=True)
        self.bias.data.fill_(0)
        self.sampled = Variable(torch.FloatTensor(self.internal_weight.data.shape).cuda(), requires_grad=False)
        self.sampled.data.fill_(0)
        self.sampled_inputs = []
        for size in [100, 1000]:
            self.sampled_inputs.append(Variable(torch.FloatTensor(np.zeros((size, in_features, pixels_in, pixels_in))).cuda(), requires_grad=False))
            self.sampled_inputs[-1].data.fill_(0)
        self.parameters = [self.internal_weight, self.magnitude, self.bias]

    def forward(self, input, deterministic=False, tangent=False,
                super_sampling_beta=False, super_sampling_truncn=False, gauss=False, gauss_objects=True):
        if gauss:
            self.sampled.data.normal_(1, np.sqrt(self.alpha))
            W_noisy = self.internal_weight * self.sampled
            result = torch.nn.functional.conv2d(input, W_noisy, bias=self.bias, padding=1)
            return result
        elif gauss_objects:
            size = input.shape[0]
            for i in range(len(self.sampled_inputs)):
                if self.sampled_inputs[i].shape[0] == size:
                    self.sampled_inputs[i].data.normal_(1, np.sqrt(self.alpha))
                    noised_input = input * self.sampled_inputs[i]
                    W_norm = self.internal_weight / (self.internal_weight.pow(2).sum(1, True).sum(2, True).sum(3, True).sqrt())
                    self.weight = (self.magnitude * W_norm)
                    result = torch.nn.functional.conv2d(noised_input, self.weight, bias=self.bias, padding=1)
                    return result

        W_norm = self.internal_weight / (self.internal_weight.pow(2).sum(1, True).sum(2, True).sum(3, True).sqrt())
        if not deterministic:
            self.sampled.data.normal_(0, np.sqrt(self.alpha))
            if tangent:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
            elif super_sampling_beta:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
                projected = projected / (projected.pow(2).sum(1, True).sum(2, True).sum(3, True).sqrt())
                tt = time.time()
                epsilon = (np.random.beta(self.alpha, self.alpha, self.out_features) - 0.5) * (np.pi-0.001)
                projected = projected * Variable(torch.cuda.FloatTensor(np.tan(epsilon)))
            elif super_sampling_truncn:
                coef = W_norm.mul(self.sampled).sum(dim=1)
                projected = self.sampled - W_norm * coef.view(-1, 1)
                projected = projected / (projected.pow(2).sum(1, True).sum(2, True).sum(3, True).sqrt())
                alpha = (np.pi / 2) / self.alpha
                epsilon = truncnorm.rvs(-alpha, alpha, scale=np.pi / (2 * alpha), size=self.out_features)
                projected = projected * Variable(torch.cuda.FloatTensor(np.tan(epsilon)))
            else:
                projected = self.sampled

            W_noisy = W_norm + projected
            W_noisy_normed = W_noisy / (W_noisy.pow(2).sum(1, True).sum(2, True).sum(3, True).sqrt())
        else:
            W_noisy_normed = W_norm
        self.weight = (self.magnitude * W_noisy_normed)
        result = torch.nn.functional.conv2d(input, self.weight, bias=self.bias, padding=1)
        return result


class Net(nn.Module):
    def __init__(self, num_classes, alpha=0.01):
        super(Net, self).__init__()
        self.parameters = []
        self.fc1 = WeightNormConv(3, 6, 32, alpha)
        self.parameters += self.fc1.parameters
        self.fc2 = WeightNormConv(6, 16, 32, alpha)
        self.parameters += self.fc2.parameters
        self.fc3 = WeightNormConv(16, 16, 16, alpha)
        self.parameters += self.fc3.parameters
        self.fc4 = WeightNormConv(16, 32, 8, alpha)
        self.parameters += self.fc4.parameters
        self.fc5 = WeightNormFC(512, 100, alpha)
        self.parameters += self.fc5.parameters
        self.fc6 = WeightNormFC(100, num_classes, alpha)
        self.parameters += self.fc6.parameters

    def forward(self, x, deterministic=False):
        hid1 = self.fc1.forward(x, deterministic=deterministic)
        out1 = nn.ReLU()(hid1)
        hid2 = self.fc2.forward(out1, deterministic=deterministic)
        hid2 = torch.nn.functional.max_pool2d(hid2, 2)
        out2 = nn.ReLU()(hid2)
        hid3 = self.fc3.forward(out2, deterministic=deterministic)
        hid3 = torch.nn.functional.max_pool2d(hid3, 2)
        out3 = nn.ReLU()(hid3)
        hid4 = self.fc4.forward(out3, deterministic=deterministic)
        hid4 = torch.nn.functional.max_pool2d(hid4, 2)
        out4 = nn.ReLU()(hid4).view(-1, 512)
        hid5 = self.fc5.forward(out4, deterministic=deterministic)
        out5 = nn.ReLU()(hid5)
        hid6 = self.fc6.forward(out5, deterministic=deterministic)
        return hid6


def make_experiment(alpha, data_size, train_data, train_labels, test_data, test_labels, num_epochs):
    net = Net(num_classes, alpha)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters, lr=learning_rate, betas=(0.9, 0.999))

    train_nondet, train_det, test_nondet, test_det, train_nondet_1, test_nondet_1 = [], [], [], [], [], []

    # Train the Model
    import time
    idxs = np.arange(50000, dtype=np.int32)
    np.random.shuffle(idxs)
    idxs = idxs[:data_size]
    train_data = train_data[torch.cuda.LongTensor(idxs.tolist())]
    train_labels = train_labels[torch.cuda.LongTensor(idxs.tolist())]

    n_samples = 30
    for epoch in range(num_epochs):
        start = time.time()
        idxs_data = np.arange(data_size)
        np.random.shuffle(idxs_data)
        for i in range(train_data.shape[0] // batch_size):
            idx_now = torch.cuda.LongTensor(idxs_data[i * batch_size:(i + 1) * batch_size].tolist())
            images = train_data[idx_now]
            labels = train_labels[idx_now]
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % (num_epochs // 200) == 0:
            print("Time to test!")
            outputs_train_nondet = compute_forward(net, Variable(train_data), False)
            outputs_test_nondet = compute_forward(net, Variable(test_data), False)

            y_train_nondet_1 = np.argmax(outputs_train_nondet, axis=1)
            y_test_nondet_1 = np.argmax(outputs_test_nondet, axis=1)
            train_nondet_acc_1 = accuracy_score(train_labels.cpu().numpy(), y_train_nondet_1)
            test_nondet_acc_1 = accuracy_score(test_labels.cpu().numpy(), y_test_nondet_1)
            train_nondet_1.append(train_nondet_acc_1)
            test_nondet_1.append(test_nondet_acc_1)
            np.save(my_dir + '/train_nondet_1_' + str(data_size) + '_' + str(alpha), train_nondet_1)
            np.save(my_dir + '/test_nondet_1_' + str(data_size) + '_' + str(alpha), test_nondet_1)
            n_samples = 10
            for i in range(n_samples - 1):
                outputs_train_nondet += compute_forward(net, Variable(train_data), False)
                outputs_test_nondet += compute_forward(net, Variable(test_data), False)
            print("Nondet test is completed")

            outputs_train_det = compute_forward(net, Variable(train_data), True)
            outputs_test_det = compute_forward(net, Variable(test_data), True)
            print("Det test is completed")

            y_train_nondet = np.argmax(outputs_train_nondet, axis=1)
            y_train_det = np.argmax(outputs_train_det, axis=1)
            y_test_nondet = np.argmax(outputs_test_nondet, axis=1)
            y_test_det = np.argmax(outputs_test_det, axis=1)
            train_nondet_acc = accuracy_score(train_labels.cpu().numpy(), y_train_nondet)
            train_det_acc = accuracy_score(train_labels.cpu().numpy(), y_train_det)
            test_nondet_acc = accuracy_score(test_labels.cpu().numpy(), y_test_nondet)
            test_det_acc = accuracy_score(test_labels.cpu().numpy(), y_test_det)
            train_nondet.append(train_nondet_acc)
            train_det.append(train_det_acc)
            test_nondet.append(test_nondet_acc)
            test_det.append(test_det_acc)
            np.save(my_dir + '/train_nondet_' + str(data_size) + '_' + str(alpha), train_nondet)
            np.save(my_dir + '/train_det_' + str(data_size) + '_' + str(alpha), train_det)
            np.save(my_dir + '/test_nondet_' + str(data_size) + '_' + str(alpha), test_nondet)
            np.save(my_dir + '/test_det_' + str(data_size) + '_' + str(alpha), test_det)

            print(
                'Epoch [%d/%d], Step [%d/%d], train_nondet %f, train_det %f, test_nondet %f, test_det %f, train_nondet_1 %f, test_nondet_1 %f,time %f'
                % (epoch + 1, num_epochs, train_data.shape[0] // batch_size, train_data.shape[0] // batch_size,
                   train_nondet_acc, train_det_acc,
                   test_nondet_acc, test_det_acc, train_nondet_acc_1, test_nondet_acc_1, time.time() - start))


# alphas = [0.02]
alphas = [0.00001]
data_sizes = [10000, 50000]
for alpha in alphas:
    for data_size in data_sizes:
        make_experiment(alpha, data_size, train_data, train_labels, test_data, test_labels,
                        num_epochs=int(200 * (50000 / data_size)))
