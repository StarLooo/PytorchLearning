# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", module="torch")

if __name__ == '__main__':
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
    ])
    # 此时尚未初始化任何参数
    print([net.layers[i].get_weights() for i in range(len(net.layers))])

    X = tf.random.uniform((2, 20))
    print(X)
    print(net(X))
    print([param.shape for param in net.get_weights()])
