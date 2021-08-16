# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


class CenteredLayer(nn.Module):
    def __init__(self, mean=0):
        super().__init__()
        self.mean = mean

    def forward(self, X):
        return X - X.mean() + self.mean


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == '__main__':
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())

    linear = MyLinear(5, 3)
    print(linear.weight)
    print(linear(torch.rand(2, 5)))

    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))
