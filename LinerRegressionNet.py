import numpy as np
import torch
from MyUtils import utils
from matplotlib import pyplot as plt

# 超参数，学习率
learningRate = 0.03
# 超参数，迭代次数
numEpochs = 5

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
    utils.setFigureSize()
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    utils.showFigure()

    batchSize = 10
    # 打印第一个批次选取出的batch_size样本特征和标签
    for X, y in utils.dataIter(batchSize, features, labels):
        print(X)
        print(y)
        break

    # 初始化模型参数,将权重w初始化成均值为0、标准差为0.01的正态随机数,偏差b则初始化成0
    w = torch.tensor(np.random.normal(0, 0.01, (numInputs, 1)), requires_grad=True, dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float32)

    net = utils.LinearRegression
    loss = utils.squaredLoss
    lastMeanLoss = 0

    for epoch in range(numEpochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中,会使用训练数据集中所有样本一次(假设样本数能够被批量大小整除)。
        # X和y分别是小批量样本的特征和标签,由dataIter()产生
        for X, y in utils.dataIter(batchSize, features, labels):
            y_predict = net(X, w, b)
            l = loss(y_predict, y).sum()  # l是有关小批量X和y的损失的平均，是标量
            l.backward()  # 小批量的平均损失对模型参数求梯度
            utils.sgd([w, b], learningRate, batchSize)  # 使用小批量随机梯度下降迭代模型参数

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
