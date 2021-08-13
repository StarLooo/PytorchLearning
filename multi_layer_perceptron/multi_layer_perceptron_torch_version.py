# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn

import Utils.utils_torch_version as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 256
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.1


# 定义relu激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def multi_layer_perceptron_wheel():
    # 读数据
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    # 定义输入层、输出层和单隐藏层的宽度：
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    # 初始化权重和偏置
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    # 定义我们的单隐藏层MLP网络
    def mlp_net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
        return H @ W2 + b2

    # 使用交叉熵损失函数
    loss = nn.CrossEntropyLoss()

    # 使用SGD优化
    updater = torch.optim.SGD(params, lr=lr)

    # 训练
    utils.train_ch3(mlp_net, train_iter, test_iter, loss, num_epochs, updater)

    # 预测
    utils.predict_ch3(mlp_net, test_iter)


def multi_layer_perceptron_easy():
    # 读数据
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    # 定义网络
    mlp_net = nn.Sequential(nn.Flatten(),
                            nn.Linear(784, 256),
                            nn.ReLU(),
                            nn.Linear(256, 10))

    # 定义初始化器
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    # 初始化参数
    mlp_net.apply(init_weights)

    # 定义损失
    loss = nn.CrossEntropyLoss()

    # 定义优化器
    trainer = torch.optim.SGD(mlp_net.parameters(), lr=lr)

    # 训练
    utils.train_ch3(mlp_net, train_iter, test_iter, loss, num_epochs, trainer)

    # 预测
    utils.predict_ch3(mlp_net, test_iter)


if __name__ == '__main__':
    multi_layer_perceptron_wheel()
    print("*******************************")
    multi_layer_perceptron_easy()
