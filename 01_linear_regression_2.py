import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# set input feature number
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)         # [1000, 2]
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b     # [1000,]
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)     # [1000,]

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break


w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):      # [10, 2], [2, 1], [1]
    return torch.mm(X, w) + b      # [10, 1]

def squared_loss(y_hat, y):     # [10, 1], [10,]
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
    # return (y_hat.view(-1) - y) ** 2 / 2
    # return (y_hat - y.view(-1)) ** 2 / 2
    # return (y_hat - y.view(y_hat.shape)) ** 2 / 2
    # return (y_hat - y.view(-1, 1)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track


# super parameters init
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once

    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):        # [10, 2], [10,]
        l = loss(net(X, w, b), y).sum()
        # calculate the gradient of batch sample loss
        l.backward()
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

