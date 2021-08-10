# -*- coding: utf-8 -*-
import os
import warnings
import tensorflow as tf
from matplotlib import pyplot as plt
import Utils.utils_tf_version as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")
# 超参数batch_size，批量大小
batch_size = 10
# 超参数num_epochs，迭代次数
num_epochs = 5
# 超参数lr，学习率
lr = 0.1


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
    utils.show_figure()


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


# def accuracy(y_hat, y_true):
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = tf.argmax(y_hat, axis=1)
#     cmp = tf.cast(y_hat, y_true.dtype) == y_true
#     return float(tf.reduce_sum(tf.cast(cmp, y_true.dtype)))


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
        preds = get_fashion_mnist_labels(tf.argmax(net(X)))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
        break


# 定义softmax操作
def softmax(X):
    X_exp = tf.exp(X)
    partition = tf.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制


# 定义softmax回归
def softmax_regression(X, W, b):
    return softmax(tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b)


# 定义交叉熵损失
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))


# 自定义一个简单的小批量随机梯度下降updater
# @save
class Updater:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        utils.sgd(self.params, grads, self.lr, batch_size)


# softmax回归从零开始实现
def softmax_regression_net_wheel():
    # 获取数据迭代器
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    for X, y in train_iter:
        print("train data shape:", X.shape,
              "train data type:", X.dtype)
        print("test data shape:", y.shape,
              "test data type:", y.dtype)
        break

    # 设置输入和输出层数
    num_inputs = 784
    num_outputs = 10

    # 初始化模型参数
    W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))

    # 定义模型和损失
    def net(X):
        return softmax_regression(X, W, b)

    loss = cross_entropy

    # 定义updater
    updater = Updater([W, b], lr=0.1)

    # 初始分类精度，应该接近0.1
    evaluate_accuracy(net, test_iter)

    # 开始训练
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    # 部分预测展示
    predict_ch3(net, test_iter)


# softmax回归简单实现
def softmax_regression_net_easy():
    # 获取数据迭代器
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    for X, y in train_iter:
        print("train data shape:", X.shape,
              "train data type:", X.dtype)
        print("test data shape:", y.shape,
              "test data type:", y.dtype)
        break

    # 定义模型和损失、优化器
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)

    # 初始分类精度，应该接近0.1
    print(evaluate_accuracy(net, test_iter))

    # 开始训练
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    # 部分预测展示
    predict_ch3(net, test_iter)


if __name__ == '__main__':
    softmax_regression_net_wheel()
    print("***************************************")
    softmax_regression_net_easy()
