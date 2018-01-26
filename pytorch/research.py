import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 200
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data',
                           train=False,
                           transform=transforms.ToTensor())
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

mean = train_dataset.train_data.type(torch.FloatTensor).view(-1, 784).mean(dim=0)
train_data = ((train_dataset.train_data.view(-1, 784).type(torch.FloatTensor) - mean) / 256.).cuda()
train_labels = train_dataset.train_labels.cuda()
test_data = ((test_dataset.test_data.view(-1, 784).type(torch.FloatTensor) - mean) / 256.).cuda()
test_labels = test_dataset.test_labels.cuda()

import os
if not os.path.exists('results'):
    os.makedirs('results')

class WeightNorm:
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

        self.parameters = [self.internal_weight, self.magnitude, self.bias]

    def forward(self, input, deterministic=False, tangent=False, gauss=True):
        W_norm = self.internal_weight / self.internal_weight.norm(dim=0)
        normed_acts = []
        if not gauss:
            for i in range(500):
                if not deterministic:
                    self.sampled.data.normal_(0, np.sqrt(self.alpha))
                else:
                    self.sampled.data.fill_(0)
                if tangent:
                    coef = W_norm.mul(self.sampled).sum(dim=1)
                    projected = self.sampled - W_norm * coef.view(-1, 1)
                else:
                    projected = self.sampled
                W_noisy = W_norm + projected
                W_noisy_normed = W_noisy / W_noisy.norm(dim=0)
                normed_activations = input.mm(W_noisy_normed - W_norm) / ((input ** 2).mm(W_norm ** 2)) ** 0.5
                normed_acts += [normed_activations.data.cpu().numpy()]
        if gauss:
            for i in range(500):
                if not deterministic:
                    self.sampled.data.normal_(1, np.sqrt(self.alpha))
                else:
                    self.sampled.data.fill_(1)
                W_noisy = W_norm * self.sampled
                normed_activations = input.mm(W_noisy - W_norm) / ((input ** 2).mm(W_norm ** 2)) ** 0.5
                normed_acts += [normed_activations.data.cpu().numpy()]
        normed_activations = np.concatenate(normed_acts, axis=0)
        print(normed_activations.shape)
        print(normed_activations)
        if gauss:
            np.save('means_gauss', normed_activations)
        else:
            np.save('means', normed_activations)
        1/0

class Net(nn.Module):
    def __init__(self, input_size, num_classes, alpha=0.01):
        super(Net, self).__init__()
        self.parameters = []
        self.fc1 = WeightNorm(input_size, 1024, alpha)
        self.parameters += self.fc1.parameters
        self.fc2 = WeightNorm(1024, 1024, alpha)
        self.parameters += self.fc2.parameters
        self.fc3 = WeightNorm(1024, 2048, alpha)
        self.parameters += self.fc3.parameters
        self.fc4 = WeightNorm(2048, num_classes, alpha)
        self.parameters += self.fc4.parameters

    def forward(self, x, deterministic=False):
        hid1 = self.fc1.forward(x, deterministic=deterministic)
        out1 = nn.ReLU()(hid1)
        hid2 = self.fc2.forward(out1, deterministic=deterministic)
        out2 = nn.ReLU()(hid2)
        hid3 = self.fc3.forward(out2, deterministic=deterministic)
        out3 = nn.ReLU()(hid3)
        hid4 = self.fc4.forward(out3, deterministic=deterministic)
        return hid4


def make_experiment(alpha, data_size, train_data, train_labels, test_data, test_labels):
    net = Net(input_size, num_classes, alpha)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters, lr=learning_rate, betas=(0.9, 0.999))

    train_nondet, train_det, test_nondet, test_det, train_nondet_1, test_nondet_1 = [], [], [], [], [], []

    # Train the Model
    import time
    idxs = np.arange(60000, dtype=np.int32)
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

        outputs_train_nondet = net.forward(Variable(train_data), deterministic=False).data.cpu().numpy() / n_samples
        outputs_test_nondet = net.forward(Variable(test_data), deterministic=False).data.cpu().numpy() / n_samples
        y_train_nondet_1 = np.argmax(outputs_train_nondet, axis=1)
        y_test_nondet_1 = np.argmax(outputs_test_nondet, axis=1)
        train_nondet_acc_1 = accuracy_score(train_labels.cpu().numpy(), y_train_nondet_1)
        test_nondet_acc_1 = accuracy_score(test_labels.cpu().numpy(), y_test_nondet_1)
        train_nondet_1.append(train_nondet_acc_1)
        test_nondet_1.append(test_nondet_acc_1)
        np.save('results/train_nondet_1_' + str(data_size) + '_' + str(alpha), train_nondet_1)
        np.save('results/test_nondet_1_' + str(data_size) + '_' + str(alpha), test_nondet_1)
        for _ in range(n_samples - 1):
            outputs_train_nondet += net.forward(Variable(train_data),
                                                deterministic=False).data.cpu().numpy() / n_samples
            outputs_test_nondet += net.forward(Variable(test_data), deterministic=False).data.cpu().numpy() / n_samples
        outputs_train_det = net.forward(Variable(train_data), deterministic=True).data.cpu().numpy()
        outputs_test_det = net.forward(Variable(test_data), deterministic=True).data.cpu().numpy()
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
        np.save('results/train_nondet_' + str(data_size) + '_' + str(alpha), train_nondet)
        np.save('results/train_det_' + str(data_size) + '_' + str(alpha), train_det)
        np.save('results/test_nondet_' + str(data_size) + '_' + str(alpha), test_nondet)
        np.save('results/test_det_' + str(data_size) + '_' + str(alpha), test_det)

        print(
            'Epoch [%d/%d], Step [%d/%d], train_nondet %f, train_det %f, test_nondet %f, test_det %f, train_nondet_1 %f, test_nondet_1 %f,time %f'
            % (epoch + 1, num_epochs, train_data.shape[0] // batch_size, train_data.shape[0] // batch_size,
               train_nondet_acc, train_det_acc,
               test_nondet_acc, test_det_acc, train_nondet_acc_1, test_nondet_acc_1, time.time() - start))


alphas = [1.0]
data_sizes = [50000]
for alpha in alphas:
    for data_size in data_sizes:
        make_experiment(alpha, data_size, train_data, train_labels, test_data, test_labels)
