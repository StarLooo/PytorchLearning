# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import os
import Utils.utils_tf_version as utils
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 10
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.03


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


# 线性回归从零开始实现
def linear_regression_net_wheel():
    # 设置真实值
    true_w = tf.constant([2, -3.4])
    true_b = 4.2

    # 生成数据集
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 查看标签与第二维特征的散点关系图
    print('features shape:', features.shape, '\nlabel shape:', labels.shape)
    utils.set_figsize()
    utils.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    utils.show_figure()

    # 初始化参数
    w = tf.Variable(initial_value=tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
    b = tf.Variable(initial_value=tf.zeros(1), trainable=True)

    # 定义模型
    net = linear_regression
    loss = squared_loss

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = loss(net(X, w, b), y)
            # 计算l关于[`w`, `b`]的梯度
            dw, db = g.gradient(target=l, sources=[w, b])
            # 使用参数的梯度更新参数
            sgd([w, b], [dw, db], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

    # 总结
    print(f'w的估计误差: {true_w - tf.reshape(w, true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')


# @save
# 构造一个torch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个TensorFlow数据迭代器。"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


# 线性回归简洁实现
def linear_regression_net_easy():
    # 设置真实值
    true_w = tf.constant([2, -3.4])
    true_b = 4.2

    # 生成数据集
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 生成数据集迭代器
    data_iter = load_array((features, labels), batch_size)

    # 定义模型并初始化参数
    initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.01)  # `keras` 是TensorFlow的高级API
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(units=1, kernel_initializer=initializer))
    loss = tf.keras.losses.MeanSquaredError()
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as tape:
                l = loss(net(X, training=True), y)
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    # 总结训练结果
    w = net.get_weights()[0]
    print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
    b = net.get_weights()[1]
    print('b的估计误差：', true_b - b)


if __name__ == '__main__':
    linear_regression_net_wheel()
    print("***************************************")
    linear_regression_net_easy()
