# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


# @save
# 计算二维互相关运算
def corr2d(X, K):
    assert X.dim() == 2 and K.dim() == 2
    h_n_in, w_n_in = X.shape
    h_k_in, w_k_in = K.shape
    h_out, w_out = h_n_in - h_k_in + 1, w_n_in - w_k_in + 1
    Y = torch.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            Y[i, j] = (X[i: i + h_k_in, j: j + w_k_in] * K).sum()
    return Y


# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


if __name__ == '__main__':
    # 测试corr2d函数
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))  # expected [[19,25],[38,43]]

    # 垂直边缘检测示例
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    Z = corr2d(X.t(), K)
    print(Z)

    # 构造一个二维卷积层，它具有1个输入通道，1个输出出通道和形状为(1，2)的卷积核，不带bias
    # 这个二维卷积层使用四维输入和输出格式(批量大小、通道、高度、宽度)
    # 其中批量大小和通道数都为1
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        lr = 3e-2
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {l.sum():.3f}')
    print("learning weight:", conv2d.weight.data)
