# -*- coding: utf-8 -*-
import os
import warnings

import tensorflow as tf
import Utils.utils_tf_version as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="tensorflow")


# 多输入通道的二维互相关运算
def corr2d_multi_in(X, K):
    assert X.shape.ndims == 3 and K.shape.ndims == 3
    # 先遍历X和K的第0个维度(通道维度)，再把它们加在一起
    return tf.reduce_sum([utils.corr2d(x, k) for x, k in zip(X, K)], axis=0)


# 具有多输入通道、多输出通道的二维互相关运算
def corr2d_multi_in_out(X, K):
    # 迭代K的第0个维度(即输出通道)，每次都对输入X执行互相关运算。
    # 最后将所有结果都叠加在一起
    return tf.stack(values=[corr2d_multi_in(X, k) for k in K], axis=0)


# 1*1卷积核
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X, (c_i, h * w))
    K = tf.reshape(K, (c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))


if __name__ == '__main__':
    # 测试corr2d_multi_in
    X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))  # expected [[56,72],[104,120]]

    # 测试corr2d_multi_in_out
    K = tf.stack(values=(K, K + 1, K + 2), axis=0)
    # channels_out * channels_in * height * width
    print(K.shape)
    print(corr2d_multi_in_out(X, K))

    # 测试corr2d_multi_in_out_1x1
    X = tf.random.normal(shape=(3, 3, 3), mean=0, stddev=1)
    K = tf.random.normal(shape=(2, 3, 1, 1), mean=0, stddev=1)
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6
