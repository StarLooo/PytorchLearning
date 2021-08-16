# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)


if __name__ == '__main__':
    x = tf.range(4)
    np.save('x-file.npy', x)
    x_load = np.load('x-file.npy', allow_pickle=True)
    print(x_load)

    y = tf.zeros(4)
    np.save('xy-files.npy', [x, y])
    x_load, y_load = np.load('xy-files.npy', allow_pickle=True)
    print((x_load, y_load))

    my_dict = {'x': x, 'y': y}
    np.save('my_dict.npy', my_dict)
    my_dict_load = np.load('my_dict.npy', allow_pickle=True)
    print(my_dict_load)

    net = MLP()
    X = tf.random.uniform((2, 20))
    Y = net(X)
    net.save_weights('mlp.params')
    clone = MLP()
    clone.load_weights('mlp.params')
    Y_clone = clone(X)
    print(Y_clone == Y)
