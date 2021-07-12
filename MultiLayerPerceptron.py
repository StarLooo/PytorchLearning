from MyUtils.Utils import *

# 超参数，隐藏层单元个数
num_hidden = 256

# 超参数:迭代次数num_epochs

num_epochs = 5
# 超参数:学习率

lr = 0.5
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs = 784, 10
    # 定义多层感知器
    MLP = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_outputs),
    )

    # 定义交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(MLP.parameters(), lr=lr)

    train(MLP, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    # 展示训练成果
    # 选取第一组小批量的测试集数据
    X, y = iter(test_iter).__next__()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(MLP(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # 打印前10个测试样本的实际类别和预测类别
    show_fashion_mnist(X[0:9], titles[0:9])
