# -*- coding: utf-8 -*-
import os
import warnings

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="tensorflow")


# 为了方便起见，我们定义了一个计算卷积层的函数
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # (a,b) + (c,d) = (a,b,c,d)
    X = tf.reshape(X, (1,) + X.shape + (1,))
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return tf.reshape(Y, Y.shape[1:3])


if __name__ == '__main__':
    # 填充 padding:
    # 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列，此时恰好可以使得输入输出张量的大小一样
    conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', groups=1)
    X = tf.random.uniform(shape=(8, 8))
    print((1,) + (8, 8) + (1,))
    print(comp_conv2d(conv2d, X).shape)

    # 请注意，上下各填充了2行，左右各填充了1列，因此总共添加了4行和2列，此时恰好可以使得输入输出张量的大小一样
    conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 3), padding='same', groups=1)
    print(comp_conv2d(conv2d, X).shape)

    # 步幅 stride:
    conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', strides=2, groups=1)
    print(comp_conv2d(conv2d, X).shape)

    conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 5), padding='valid', strides=(3, 4), groups=1)
    print(comp_conv2d(conv2d, X).shape)
