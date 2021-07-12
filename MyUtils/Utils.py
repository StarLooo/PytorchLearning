import random
import sys
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt
import torch.nn as nn

# 全局变量,控制是否显示绘图
isShowFigure = True


# 用矢量图显示
def useSvgDisplay():
    display.set_matplotlib_formats('svg')


# 设置图的尺寸
def setFigureSize(figureSize=(3.5, 2.5)):
    useSvgDisplay()
    plt.rcParams['figure.figsize'] = figureSize


# 根据全局变量isShowFigure的值来决定是否显示matplotlib库中的绘图函数的绘图
def showFigure():
    if isShowFigure:
        plt.show()


# 小批次样本选取迭代函数，每次返回batch_size(批量大小)个随机样本的特征和标签。
def dataIter(batchSize, features, labels):
    numExamples = len(features)
    indices = list(range(numExamples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, numExamples, batchSize):
        j = torch.LongTensor(indices[i: min(i + batchSize, numExamples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


# 线性回归的计算模型函数
def LinearRegression(X, w, b):
    return torch.mm(X, w) + b


# 平方损失函数
def squaredLoss(y_predict, y):
    return (y_predict - y.view(y_predict.size())) ** 2 / 2


# SGD(随机梯度下降)函数
def sgd(params, learningRate, batchSize):
    for param in params:
        # 注意这里更改param时用的param.data
        param.data -= learningRate * param.grad / batchSize


# 获得fashion_mnist数据集的标签类别(数值->文本)
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 在一行里画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    useSvgDisplay()
    # 这里的_表示我们忽略(不使用)的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    showFigure()


def load_data_fashion_mnist(batch_size=256):
    # 导入mnist训练集
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                    train=True, download=True, transform=transforms.ToTensor())
    # 导入mnist测试集
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',
                                                   train=False, download=True, transform=transforms.ToTensor())
    # 打印训练集和测试集的样本数
    print(len(mnist_train), len(mnist_test))

    # 打印样本的shape和标签
    feature, label = mnist_train[0]
    print(feature.shape)  # Channel x Height x Width
    print(label)

    # 看一下训练数据集中前10个样本的图像内容和文本标签
    first_ten_X, first_ten_y = [], []
    for i in range(10):
        first_ten_X.append(mnist_train[i][0])
        first_ten_y.append(mnist_train[i][1])
    show_fashion_mnist(first_ten_X, get_fashion_mnist_labels(first_ten_y))

    if sys.platform.startswith('win'):
        num_workers = 4
    else:
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)  # 按行求和
    return X_exp / partition  # 这里应用了广播机制


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 分类准确率计算函数
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().sum().item()


# 计算模型的总体分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += accuracy(net(X), y)
        n += y.shape[0]
    return acc_sum / n


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    @staticmethod
    def forward(x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# 通用训练函数
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, num_sample = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += accuracy(y_hat, y)
            num_sample += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / num_sample, train_acc_sum / num_sample, test_acc))


# 绘制y-x图的函数
def xy_plot(x_values, y_values, name):
    setFigureSize(figureSize=(5, 2.5))
    plt.plot(x_values.detach().numpy(), y_values.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    showFigure()
