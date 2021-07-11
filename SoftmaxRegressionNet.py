import numpy as np
from MyUtils.Utils import *

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
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)


    def softmax_net(X):
        return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


    # 打印初始准确率
    print(evaluate_accuracy(test_iter, softmax_net))

    # 进行训练
    train(softmax_net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    # 展示训练成果
    # 选取第一组小批量的测试集数据
    X, y = test_iter.__next__()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(softmax_net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # 打印前10个测试样本的实际类别和预测类别
    show_fashion_mnist(X[0:9], titles[0:9])
