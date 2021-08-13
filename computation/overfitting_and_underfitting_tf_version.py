# -*- coding: utf-8 -*-
import math
import numpy as np
import tensorflow as tf
import os
import Utils.utils_tf_version as utils
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


# @save
# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):  # @save
    metric = utils.Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(tf.reduce_sum(l), tf.size(l))
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式特征中实现了它
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = utils.load_array((train_features, train_labels), batch_size)
    test_iter = utils.load_array((test_features, test_labels), batch_size,
                                 is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    for epoch in range(num_epochs):
        utils.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print("epoch:", epoch + 1,
                  "train loss:", round(evaluate_loss(net, train_iter, loss), 6),
                  "test loss:", round(evaluate_loss(net, test_iter, loss), 6))
    print("training finished!")
    print('weight:', net.get_weights()[0].T)


if __name__ == '__main__':
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配大量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
    # labels的维度: (n_train + n_test,)
    labels = np.dot(poly_features, true_w)
    noise = np.random.normal(scale=0.1, size=labels.shape)
    labels += noise

    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [tf.constant(x, dtype=tf.float32) for x in
                                               [true_w, features, poly_features, labels]]
    # 从多项式特征中选择前4个维度，即 1, x, x^2/2!, x^3/3!
    # 拟合最合适
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    print("***********************************")

    # 从多项式特征中选择前2个维度，即 1, x
    # 欠拟合
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    print("***********************************")

    # 从多项式特征中选取所有维度(20维)
    # 过拟合
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
    print("***********************************")
