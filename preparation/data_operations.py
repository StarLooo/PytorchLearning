# -*- coding: utf-8 -*-
import torch
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def testFunc1():
    # torch version
    x = torch.arange(12)
    print(x)
    print(x.shape)
    print(x.numel())
    print(torch.sum(x))
    X = x.reshape(3, -1)
    print(X)
    print(torch.zeros((2, 3, 4)))
    print(torch.ones(3, 5))
    print(torch.randn(2, 2))
    print(torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    print("********************************************************")
    # tf version
    y = tf.range(12)
    print(y)
    print(y.shape)
    print(tf.size(y))
    print(tf.reduce_sum(y))
    Y = tf.reshape(y, (3, -1))
    print(Y)
    print(tf.zeros((2, 3, 4)))
    print(tf.ones((3, 5)))
    print(tf.random.normal(shape=[2, 2]))
    print(tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def testFunc2():
    # torch version
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([1, 2, 3, 4])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)
    print(torch.exp(x))
    print("********************************************************")
    # tf version
    x = tf.constant([1, 2, 4, 8])
    y = tf.constant([1, 2, 3, 4])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)
    print(tf.exp(tf.constant([1.0, 2.0, 3.0, 4.0])))


def testFunc3():
    # torch version
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(torch.cat((X, Y), dim=0))
    print(torch.cat((X, Y), dim=1))
    print(X == Y)
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a + b)
    print("********************************************************")
    # tf version
    X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
    Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(tf.concat([X, Y], axis=0))
    print(tf.concat([X, Y], axis=1))
    print(X == Y)
    a = tf.reshape(tf.range(3), (3, 1))
    b = tf.reshape(tf.range(2), (1, 2))
    print(a + b)


def testFunc4():
    # torch version
    X = torch.arange(12).reshape((3, 4))
    print(X[-1])
    print(X[1:3])
    print(X[:, :2])
    X[1, 2] = 9
    print(X)
    print("********************************************************")
    # tf version
    Y = tf.reshape(tf.range(12), (3, 4))
    print(Y[-1])
    print(Y[1:3])
    print(Y[:, :2])
    Y_var = tf.Variable(Y)
    Y_var[1, 2].assign(9)
    print(Y_var)
    Y_var[0:2, :].assign(tf.ones(Y_var[0:2, :].shape, dtype=tf.int32) * 12)
    print(Y_var)


def testFunc5():
    # torch version
    X = torch.ones(3, 4)
    Y = torch.randn(3, 4)
    before = id(Y)
    Y = Y + X
    print(id(Y) == before)
    Z = torch.zeros_like(Y)
    print('id(Z):', id(Z))
    Z[:] = X + Y
    print('id(Z):', id(Z))
    before = id(X)
    X += Y
    print(id(X) == before)
    A = X.numpy()
    B = torch.tensor(A)
    print(type(A), type(B))
    x = torch.tensor([3.5])
    print(x.item())
    print("********************************************************")
    # tf version
    X = tf.ones((3, 4), dtype=tf.float32)
    Y = tf.random.normal(shape=(3, 4))
    before = id(Y)
    Y = Y + X
    print(id(Y) == before)

    @tf.function
    def computation(X, Y):
        Z = tf.zeros_like(Y)  # 这个未使用的值将被删除
        A = X + Y  # 当不再需要时，分配将被复用
        B = A + Y
        C = B + Y
        return C + Y

    print(computation(X, Y))
    A = X.numpy()
    B = tf.constant(A)
    print(type(A), type(B))
    y = torch.tensor([3.5])
    print(y.item())


if __name__ == '__main__':
    testFunc1()
    testFunc2()
    testFunc3()
    testFunc4()
    testFunc5()
