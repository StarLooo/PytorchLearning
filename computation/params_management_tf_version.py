# -*- coding: utf-8 -*-
import os
import warnings

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="tensorflow")


def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)


def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add(block1(name=f'block-{i}'))
    return net


class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data = tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor = (tf.abs(data) >= 5)
        factor = tf.cast(factor, tf.float32)
        return data * factor


if __name__ == '__main__':
    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu),
        tf.keras.layers.Dense(1),
    ])
    X = tf.random.uniform((2, 4))
    print(net(X))

    print(net.layers[2].weights)
    print(type(net.layers[2].weights[1]))
    print(net.layers[2].weights[1])
    print(tf.convert_to_tensor(net.layers[2].weights[1]))
    print(net.layers[1].weights)
    print(net.get_weights())
    print(net.get_weights()[1])

    rgnet = tf.keras.Sequential()
    rgnet.add(block2())
    rgnet.add(tf.keras.layers.Dense(1))
    rgnet(X)
    print(rgnet.summary())
    print(rgnet.layers[0].layers[1].layers[1].weights[1])

    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            4, activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            bias_initializer=tf.zeros_initializer()),
        tf.keras.layers.Dense(1)])
    print(net(X))
    print((net.weights[0], '\n', net.weights[1]))

    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            4, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.Constant(1),
            bias_initializer=tf.zeros_initializer()),
        tf.keras.layers.Dense(1),
    ])
    print(net(X))
    print((net.weights[0], '\n', net.weights[1]))

    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            4,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform()),
        tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Constant(1)),
    ])
    print(net(X))
    print(net.layers[1].weights[0])
    print(net.layers[2].weights[0])

    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            4,
            activation=tf.nn.relu,
            kernel_initializer=MyInit()),
        tf.keras.layers.Dense(1),
    ])
    print(net(X))
    print(net.layers[1].weights[0])

    net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
    net.layers[1].weights[0][0, 0].assign(42)
    print(net.layers[1].weights[0])

    # tf.keras的表现有点不同。它会自动删除重复层
    shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        shared,
        shared,
        tf.keras.layers.Dense(1),
    ])
    print(net(X))
    # 检查参数是否不同
    print(len(net.layers) == 3)
