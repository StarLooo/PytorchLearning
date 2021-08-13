# -*- coding: utf-8 -*-
import os
import warnings

import tensorflow as tf

import Utils.utils_tf_version as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 256
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.1


# 定义relu激活函数
def relu(X):
    return tf.math.maximum(X, 0)


def multi_layer_perceptron_wheel():
    # 读数据
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    # 定义输入层、输出层和单隐藏层的宽度：
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    # 初始化权重和偏置
    W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
    b1 = tf.Variable(tf.zeros(num_hiddens))
    W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
    b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

    params = [W1, b1, W2, b2]

    # 定义我们的单隐藏层MLP网络
    def mlp_net(X):
        X = tf.reshape(X, (-1, num_inputs))
        H = relu(tf.matmul(X, W1) + b1)
        return tf.matmul(H, W2) + b2

    # 使用稀疏分类交叉熵损失函数
    def loss(y_hat, y):
        return tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)

    # 使用SGD优化
    updater = utils.Updater([W1, W2, b1, b2], lr)

    # 训练
    utils.train_ch3(mlp_net, train_iter, test_iter, loss, num_epochs, updater)

    # 预测
    utils.predict_ch3(mlp_net, test_iter)


def multi_layer_perceptron_easy():
    # 读数据
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    # 定义网络
    mlp_net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10)])

    # 定义损失
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 定义优化器
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)

    # 训练
    utils.train_ch3(mlp_net, train_iter, test_iter, loss, num_epochs, trainer)

    # 预测
    utils.predict_ch3(mlp_net, test_iter)


if __name__ == '__main__':
    multi_layer_perceptron_wheel()
    print("*******************************")
    multi_layer_perceptron_easy()
