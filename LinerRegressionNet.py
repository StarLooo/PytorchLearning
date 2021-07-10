import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

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
    # 将训练数据的特征和标签组合
    dataSet = Data.TensorDataset(features, labels)
    # 随机读取小批量
    dataIter = Data.DataLoader(dataSet, batchSize, shuffle=True)

    # 打印第一个批次选取出的batch_size样本特征和标签
    for X, y in dataIter:
        print(X)
        print(y)
        break

    # 用nn.Sequential来更加方便地搭建网络
    net = nn.Sequential(
        nn.Linear(numInputs, 1)
        # 此处还可以传入其他层
    )

    # 初始化模型参数
    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0)

    # 定义损失函数为MSE损失函数
    loss = nn.MSELoss()

    # 定义优化函数
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    # 进行迭代训练
    for epoch in range(0, numEpochs):
        for X, y in dataIter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()

        totMSE = loss(net(features), labels.view(-1, 1))
        print('epoch times: %d, loss: %f' % (epoch + 1, totMSE.item()))

    # 输出最终结果
    dense = net[0]

    print("-----------训练结束,最终结果如下-----------")
    print("last total mean loss %f" % (loss(net(features), labels.view(-1, 1)).item()))
    print("----------------------------------------")
    print("true_w:", true_w)
    print("predict_w:", dense.weight)
    print("true_b:", true_b)
    print("predict_b:", dense.bias)
