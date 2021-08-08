# -*- coding: utf-8 -*-
import torch
import random
import os
import Utils.utils_torch_version as utils
import warnings
from torch.utils import data
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 10
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.03


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


# 线性回归从零开始实现
def linear_regression_net_wheel():
    # 设置真实值
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成数据集
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 查看标签与第二维特征的散点关系图
    print('features shape:', features.shape, '\nlabel shape:', labels.shape)
    utils.set_figsize()
    utils.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    utils.show_figure()

    # 初始化参数
    w = torch.normal(mean=0, std=0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 定义模型
    net = linear_regression
    loss = squared_loss

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            # 因为l的形状是(batch_size, 1)，而不是一个标量。
            # 所以将`l`中的所有元素被加到一起，并以此计算关于[`w`, `b`]的梯度
            l.sum().backward()  # 反向传播
            sgd([w, b], lr, batch_size)  # 小批量随机梯度下降优化参数
        with torch.no_grad():
            train_l = loss(y_hat=net(features, w, b), y_true=labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    # 总结训练结果
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    print('b的估计误差：', true_b - b)


# @save
# 构造一个torch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 线性回归简洁实现
def linear_regression_net_easy():
    # 设置真实值
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成数据集
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 生成数据集迭代器
    data_iter = load_array((features, labels), batch_size)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(params=net.parameters(), lr=0.03)

    # 初始化参数

    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    # 总结训练结果
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)


if __name__ == '__main__':
    linear_regression_net_wheel()
    print("***************************************")
    linear_regression_net_easy()
