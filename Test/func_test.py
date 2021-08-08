import torch
import numpy as np
import matplotlib.pylab as plt
from MyUtils.utils import *

if __name__ == '__main__':
    # 测试relu()函数
    # x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    # y = x.relu()
    # xy_plot(x, y, 'relu')
    # y.sum().backward()
    # xy_plot(x, x.grad, 'grad of relu')

    # 测试sigmod()函数
    # x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    # y = x.sigmoid()
    # xy_plot(x, y, 'sigmoid')
    # y.sum().backward()
    # xy_plot(x, x.grad, 'grad of sigmoid')

    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = x.tanh()
    xy_plot(x, y, 'tanh')
    y.sum().backward()
    xy_plot(x, x.grad, 'grad of tanh')

