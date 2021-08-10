# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display
import math
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量,控制是否显示绘图
is_show_figure = True


# 记录多次运行时间。
class Timer:
    def __init__(self):
        self.tik = 0
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


# # 用矢量图显示
# def useSvgDisplay():
#     display.set_matplotlib_formats('svg')


# 设置图的尺寸
def set_figsize(figureSize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figureSize


# 根据全局变量is_show_figure的值来决定是否显示matplotlib库中的绘图函数的绘图
def show_figure():
    if is_show_figure:
        plt.show()


# @Save
# 生成数据集
def synthetic_data(w, b, num_examples):
    d = len(w)
    X = tf.random.normal(shape=(num_examples, d))
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    noise = tf.random.normal(shape=y.shape, mean=0.0, stddev=0.01)
    y += noise
    y = tf.reshape(y, (-1, 1))
    return X, y


# @Save
# 小批量数据产生器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, batch_indices), tf.gather(labels, batch_indices)


# @save
# 线性回归模型函数
def linear_regression(X, w, b):
    assert X.shape[1] == w.shape[0]
    return tf.matmul(X, w) + b


# @save
# 均方损失函数
def squared_loss(y_hat, y_true):
    return (y_hat - tf.reshape(y_true, y_hat.shape)) ** 2 / 2


# @save
# 小批量随机梯度下降函数
def sgd(params, grads, lr, batch_size):
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


# @save
# 构造一个torch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个TensorFlow数据迭代器。"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
