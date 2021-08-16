# -*- coding: utf-8 -*-
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt
from torch.utils import data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量,控制是否显示绘图
is_show_figure = True


# 记录多次运行时间。
class Timer:
    def __init__(self):
        self.tik = 0
        self.times = []
        self.start()

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


# @save
# only for jupyter
# 在动画中绘制数据
class Animator:

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: (self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# # 用矢量图显示
# def useSvgDisplay():
#     display.set_matplotlib_formats('svg')


# 设置图的尺寸
def set_figsize(figureSize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figureSize


# 根据全局变量is_show_figure的值来决定是否显示matplotlib库中的绘图函数的绘图
def show_figure():
    if is_show_figure:
        plt.show()
        # plt.pause(5)


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
    show_figure()


# @save
# 使用n_jobs个进程来读取数据
def get_data_loader_workers(n_jobs=1):
    return n_jobs


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


# @save
# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        y_pred = net(X)
        y = y.reshape(y_pred.shape)
        l = loss(y_pred, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


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
