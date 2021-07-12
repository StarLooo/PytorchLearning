from MyUtils.Utils import *
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

# 超参数:迭代次数num_epochs
num_epochs = 5

# 超参数:学习率0.1
lr = 0.1

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 初始化模型参数
    # 跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是
    # 28×28 = 78428×28 = 784:该向量的每个元素对应图像中每个像素。
    # 由于图像有10个类别，单层神经网络输出层的输出个数为10,因此softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。
    num_inputs = 784
    num_outputs = 10

    net = nn.Sequential(
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
        ])
    )

    # 初始化参数
    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)

    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()

    # 定义优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 开始训练模型
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    # 展示训练成果
    # 选取第一组小批量的测试集数据
    X, y = iter(test_iter).__next__()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # 打印前10个测试样本的实际类别和预测类别
    show_fashion_mnist(X[0:9], titles[0:9])
