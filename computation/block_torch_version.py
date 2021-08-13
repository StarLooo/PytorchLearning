# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Block的构造函数来执行必要的初始化
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params(稍后将介绍)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意,这里我们使用ReLU的函数版本,其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # 这里，block是Module子类的一个实例,我们把它保存在Module类的成员变量
            # _modules的类型是OrderedDict。
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    # 自定义更复杂的前向传播
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和dot函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层,这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == '__main__':
    net = MLP()
    X = torch.rand(size=(2, 20))
    print(X)
    print(net(X))
    print("************************************************************")
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X))
    print("************************************************************")
    net = FixedHiddenMLP()
    print(net(X))
    print("************************************************************")
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    chimera(X)
