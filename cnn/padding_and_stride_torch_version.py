# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


# 为了方便起见，我们定义了一个计算卷积层的函数
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的(1，1)示批量大小和通道数都是1
    # (a,b) + (c,d) = (a,b,c,d)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


if __name__ == '__main__':
    # 填充 padding:
    # 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列，此时恰好可以使得输入输出张量的大小一样
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    # 请注意，上下各填充了2行，左右各填充了1列，因此总共添加了4行和2列，此时恰好可以使得输入输出张量的大小一样
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)

    # 步幅 stride:
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)
