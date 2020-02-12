# 加载各种包或者模块
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import utils as d2l

print(torch.__version__)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='input/FashionMNIST2065')

num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)      # [784, 10]

    def forward(self, x):  # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# from collections import OrderedDict
#
# net = nn.Sequential(
#     # FlattenLayer(),
#     # LinearNet(num_inputs, num_outputs)
#     OrderedDict([
#         ('flatten', FlattenLayer()),
#         ('linear', nn.Linear(num_inputs, num_outputs))])  # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
# )

# init.normal_(net.linear.weight, mean=0, std=0.01)
# init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss() # 下面是他的函数原型

optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # 下面是函数原型

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)








