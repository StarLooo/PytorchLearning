# -*- coding: utf-8 -*-
import time
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt

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


# @save
# 将Fashion-MNIST数据集的数字标签映射为文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# @save
# 绘制一系列图片
def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if tf.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    show_figure()


# @save
# 使用n_jobs个进程来读取数据。
def get_data_loader_workers(n_jobs=1):
    return n_jobs


# @save
# 加载Fashion_MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    # 下载Fashion-MNIST数据集，然后将其加载到内存中。
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()

    # 将所有数字除以255，使所有像素值介于0和1之间，在最后添加一个批处理维度，
    # 并将标签转换为int32。
    def process(X, y):
        return tf.expand_dims(X, axis=3) / 255, tf.cast(y, dtype='int32')

    def resize_fn(X, y):
        return tf.image.resize_with_pad(X, resize, resize) if resize else X, y

    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn)
    )


# @save
# 计算分类准确的类的个数
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))


# @save
# 在n个变量上累加
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# @save
# 计算在指定数据集上模型的分类精度
def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), tf.size(y))
    return metric[0] / metric[1]


# @save
# 训练模型一个迭代周期(定义见第3章)
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # 注意Keras内置的损失接受的是(标签，预测)，这不同于用户在本书中的实现
            # 本书中的实现接受(预测，标签)，例如我们上面实现的交叉熵
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        # 如果使用的是keras中的优化器
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(grads_and_vars=zip(grads, params))
        # 使用的定制的优化器
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras的loss默认返回一个批量的平均损失
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


# @save
# 训练模型(定义见第3章)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    train_metrics = None
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        print("epoch:", epoch + 1, "train loss:", round(train_loss, 6), "train_acc", round(train_acc, 6))
    test_acc = evaluate_accuracy(net, test_iter)
    print("train finished, test_acc", test_acc)


# @save
# 预测标签
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        trues = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(tf.argmax(net(X), axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
        break


# 自定义一个简单的小批量随机梯度下降updater
# @save
class Updater:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        sgd(self.params, grads, self.lr, batch_size)


# @save
# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):  # @save
    metric = Accumulator(2)  # 损失的总和, 样本数量
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(tf.reduce_sum(l), tf.size(l))
    return metric[0] / metric[1]


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
