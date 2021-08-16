# -*- coding: utf-8 -*-
import os
import warnings

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="tensorflow")


# @save
# 计算二维互相关运算
def corr2d(X, K):
    assert X.shape.ndims == 2 and X.shape.ndims == 2
    h_n_in, w_n_in = X.shape
    h_k_in, w_k_in = K.shape
    h_out, w_out = h_n_in - h_k_in + 1, w_n_in - w_k_in + 1
    Y = tf.Variable(tf.zeros((h_out, w_out)))
    for i in range(h_out):
        for j in range(w_out):
            Y[i, j].assign(tf.reduce_sum(X[i: i + h_k_in, j: j + w_k_in] * K))
    return Y


# 实现二维卷积层
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1,),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias


if __name__ == '__main__':
    # 测试corr2d函数
    X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))  # expected [[19,25],[38,43]]

    # 垂直边缘检测示例
    X = tf.Variable(tf.ones((6, 8)))
    X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
    print(X)
    K = tf.constant([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    Z = corr2d(tf.transpose(X), K)
    print(Z)

    # 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

    # 构造一个二维卷积层，它具有1个输入通道，1个输出出通道和形状为(1，2)的卷积核，不带bias
    # 这个二维卷积层使用四维输入和输出格式(批量大小、通道、高度、宽度)
    # 其中批量大小和通道数都为1
    X = tf.reshape(X, (1, 6, 8, 1))
    Y = tf.reshape(Y, (1, 6, 7, 1))

    Y_hat = conv2d(X)
    for i in range(10):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(conv2d.weights[0])
            Y_hat = conv2d(X)
            l = (abs(Y_hat - Y)) ** 2
            # 迭代卷积核
            update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
            weights = conv2d.get_weights()
            weights[0] = conv2d.weights[0] - update
            conv2d.set_weights(weights)
            if (i + 1) % 2 == 0:
                print(f'batch {i + 1}, loss {tf.reduce_sum(l):.3f}')
    print("learning weight:", conv2d.get_weights())
