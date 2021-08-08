# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import random
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from IPython import display
import math
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量,控制是否显示绘图
is_show_figure = True


# 记录多次运行时间。
class Timer:
    def __init__(self):
        self.times = []
        self.start()
        self.tik = 0

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


# # 用矢量图显示
# def useSvgDisplay():
#     display.set_matplotlib_formats('svg')


def testTimer():
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')
    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')


# 设置图的尺寸
def set_figsize(figureSize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figureSize


# 根据全局变量is_show_figure的值来决定是否显示matplotlib库中的绘图函数的绘图
def show_figure():
    if is_show_figure:
        plt.show()


# @Save
# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise。"""
    d = len(w)
    X = torch.normal(mean=0, std=1, size=(num_examples, d))
    y = torch.matmul(X, w) + b
    noise = torch.normal(mean=0, std=0.01, size=y.shape)
    y += noise
    return X, y.reshape((-1, 1))  # 将y转换为为列向量


# @Save
# 小批量数据产生器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# @save
# 线性回归模型函数
def linear_regression(X, w, b):
    assert X.shape[1] == len(w)
    return torch.matmul(X, w) + b


# @save
# 均方损失函数
def squared_loss(y_hat, y_true):
    return (y_hat - y_true.reshape(y_hat.shape)) ** 2 / 2


# @save
# 小批量随机梯度下降函数
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# @save
# 构造一个torch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    testTimer()
