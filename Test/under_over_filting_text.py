from MyUtils.utils import *
import numpy as np
import torch.utils.data

num_epochs, loss = 100, torch.nn.MSELoss()


# 拟合并做图
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)

    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('pred_weight:', net.weight.data,
          '\npred_bias:', net.bias.data)


if __name__ == '__main__':
    # 生成一个人工数据集。在训练数据集和测试数据集中，给定样本特征xx，我们使用如下的三阶多项式函数来生成该样本的标签：
    # y=1.2x−3.4x^2+5.6x^3+5+ϵ
    # 其中噪声项ϵ服从均值为0、标准差为0.01的正态分布。
    # 训练数据集和测试数据集的样本数都设为100。
    n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
    features = torch.randn((n_train + n_test, 1))
    poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
    labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
              + true_w[2] * poly_features[:, 2] + true_b)
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    # 三阶多项式拟合
    print("三阶多项式拟合:")
    fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
                 labels[:n_train], labels[n_train:])

    # 线性拟合(欠拟合)
    print("线性拟合(欠拟合):")
    fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
                 labels[n_train:])

    # 样本不足(过拟合)
    print("样本不足(过拟合):")
    fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
                 labels[n_train:])
