# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from matplotlib import pyplot as plt
import Utils.utils_torch_version as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 10
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.1


# @save
# 将Fashion-MNIST数据集的数字标签映射为文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# @save
# 绘制一系列图片
def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    utils.show_figure()


# @save
# 使用4个进程来读取数据。
def get_data_loader_workers():
    return 1


# @save
# 加载Fashion_MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):
    # 读取Fashion-MNIST数据集并进行预处理
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    # 测试读取全部训练集数据所需时间
    # timer = utils.Timer()
    # timer.start()
    # for X, y in train_iter:
    #     continue
    # print("read all train data cost:", f'{timer.stop():.2f} sec')

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_data_loader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_data_loader_workers()))


# @save
# 计算分类准确的类的个数
def accuracy(y_hat, y_true):
    return (y_hat.argmax(dim=1) == y_true).float().sum().item()


# @save
# 在n个变量上累加
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# @save
# 计算在指定数据集上模型的分类精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# @save
# 训练模型一个迭代周期(定义见第3章)
def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()  # 注意：torch自带的loss是求平均后的标量
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(batch_size=X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练总损失率和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


# @save
# 训练模型(定义见第3章)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    train_metrics = None
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        print("epoch:", epoch + 1, "train loss:", round(train_loss, 6), "train_acc", round(train_acc, 6))
    test_acc = evaluate_accuracy(net, test_iter)
    print("train finished, test_acc", test_acc)


# @save
# 预测标签
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        trues = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        break


# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 定义softmax回归
def softmax_regression(X, W, b):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 定义交叉熵损失
def cross_entropy(y_hat, y_true):
    return - torch.log(y_hat[range(len(y_hat)), y_true])


# softmax回归从零开始实现
def softmax_regression_net_wheel():
    # 获取数据迭代器
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    for X, y in train_iter:
        print("train data shape:", X.shape,
              "train data type:", X.dtype)
        print("test data shape:", y.shape,
              "test data type:", y.dtype)
        break

    # 设置输入和输出层数
    num_inputs = 784
    num_outputs = 10

    # 初始化模型参数
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 定义模型和损失
    def net(X):
        return softmax_regression(X, W, b)

    loss = cross_entropy

    # 定义updater
    def updater(batch_size):
        return utils.sgd([W, b], lr, batch_size)

    # 初始分类精度，应该接近0.1
    evaluate_accuracy(net, test_iter)

    # 开始训练
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # 部分预测展示
    predict_ch3(net, test_iter)


if __name__ == '__main__':
    softmax_regression_net_wheel()
