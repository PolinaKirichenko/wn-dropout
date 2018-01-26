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


def make_experiment(alpha, data_size, train_data, train_labels, test_data, test_labels):
    net = Net(input_size, num_classes, alpha)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters, lr=learning_rate, betas=(0.9, 0.999))

    train_nondet, train_det, test_nondet, test_det, train_nondet_1, test_nondet_1 = [], [], [], [], [], []

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

