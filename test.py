import torch
import torch.nn as nn
import numpy as np


X = torch.rand(4, 3, 100, 100)
print(X.shape)

conv2d = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=2)
Y = conv2d(X)
print('Y.shape: ', Y.shape)
print('weight.shape: ', conv2d.weight.shape)
print('bias.shape: ', conv2d.bias.shape)