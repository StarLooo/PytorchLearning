import numpy as np
from MyUtils.Utils import *

# 超参数，隐藏层单元个数
num_hidden = 256

# 超参数:迭代次数num_epochs
num_epochs = 5

# 超参数:学习率100
# 之所以这么大是因为之前我们自己写的cross_entropy()没有对batch_size求平均，
# 但pytorch中的CrossEntropyLoss()会求平均。导致loss变小，影响sgd()中每次计算的下降也会变为原来的1/batchSize，因此需要扩大lr来中和。
lr = 100

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs = 784, 10

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hidden)), dtype=torch.float)
    b1 = torch.zeros(num_hidden, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hidden, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)

    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)


    # 定义多层感知机
    def MLP(X):
        X = X.view((-1, num_inputs))
        H = (torch.matmul(X, W1) + b1).relu()
        return torch.matmul(H, W2) + b2


    # 定义交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 训练模型
    train(MLP, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

    # 展示训练成果
    # 选取第一组小批量的测试集数据
    X, y = iter(test_iter).__next__()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(MLP(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # 打印前10个测试样本的实际类别和预测类别
    show_fashion_mnist(X[0:9], titles[0:9])
