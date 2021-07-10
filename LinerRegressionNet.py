import random
import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt

# 全局变量,控制是否显示绘图
isShowFigure = False
# 超参数，学习率
learningRate = 0.03
# 超参数，迭代次数
numEpochs = 5


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


if __name__ == '__main__':
    # 生成数据集
    numInputs = 2  # 特征维度为2
    numExamples = 1000  # 样本数为1000
    true_w = [2, -3.4]  # 真实参数w
    true_b = 4.2  # 真实偏差b
    features = torch.randn(numExamples, numInputs,
                           dtype=torch.float32)  # 随机生成的样本，符合标准正态分布
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 标签
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float32)  # 对标签加上符合(0,0.1^2)正态分布的噪声

    # 查看数据生成结果的特征和标签的前5行
    # print(features[:5, ])
    # print(labels[:5])

    # 查看标签与第二维特征的散点关系图
    setFigureSize()
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    showFigure()

    batchSize = 10
    # 打印第一个批次选取出的batch_size样本特征和标签
    for X, y in dataIter(batchSize, features, labels):
        print(X)
        print(y)
        break

    # 初始化模型参数,将权重w初始化成均值为0、标准差为0.01的正态随机数,偏差b则初始化成0
    w = torch.tensor(np.random.normal(0, 0.01, (numInputs, 1)), requires_grad=True, dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

    net = LinearRegression
    loss = squaredLoss
    lastMeanLoss = 0

    for epoch in range(numEpochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中,会使用训练数据集中所有样本一次(假设样本数能够被批量大小整除)。
        # X和y分别是小批量样本的特征和标签,由dataIter()产生
        for X, y in dataIter(batchSize, features, labels):
            y_predict = net(X, w, b)
            l = loss(y_predict, y).sum()  # l是有关小批量X和y的损失的平均，是标量
            l.backward()  # 小批量的平均损失对模型参数求梯度
            sgd([w, b], learningRate, batchSize)  # 使用小批量随机梯度下降迭代模型参数

            # 不要忘了梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()

        # 本次迭代训练结束，计算总损失向量
        trainLoss = loss(net(features, w, b), labels)
        # 输出本次迭代的结果
        print("epoch times: %d, total mean loss %f" % (epoch + 1, trainLoss.mean().item()))
        lastMeanLoss = trainLoss  # 更新lastMeanLoss

    # 输出最终结果
    print("-----------训练结束,最终结果如下-----------")
    print("last total mean loss %f" % (lastMeanLoss.mean().item()))
    print("----------------------------------------")
    print("true_w:", true_w)
    print("predict_w:", w.tolist())
    print("true_b:", true_b)
    print("predict_b:", b.tolist())
