# -*- coding: utf-8 -*-
import torch
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def testFunc1():
    # torch version
    A = torch.arange(20).reshape(5, 4)
    B = torch.arange(20).reshape(5, 4)
    print(A.T)
    print(A * B)
    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    print(a + X)
    print((a * X).shape)
    print("********************************************************")
    # tf version
    A = tf.reshape(tf.range(20), (5, 4))
    B = tf.reshape(tf.range(20), (5, 4))
    print(tf.transpose(A))
    print(A * B)
    a = 2
    X = tf.reshape(tf.range(24), (2, 3, 4))
    print(a + X)
    print((a * X).shape)


def testFunc2():
    # torch version
    A = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    print(A)
    print(A.sum())
    A_sum_axis0 = A.sum(axis=0)
    print(A_sum_axis0)
    A_sum_axis1 = A.sum(axis=1)
    print(A_sum_axis1)
    print(A.mean())
    print(A.mean() == A.sum() / A.numel())
    A_sum_axis0_keepdims = A.sum(axis=0, keepdims=True)
    print(A_sum_axis0_keepdims)
    print(A_sum_axis0_keepdims.shape)
    A_sum_axis1_keepdims = A.sum(axis=1, keepdims=True)
    print(A_sum_axis1_keepdims)
    print(A_sum_axis1_keepdims.shape)
    print(A.cumsum(axis=0))
    print(A.cumsum(axis=1))
    print("********************************************************")
    # tf version
    A = tf.reshape(tf.range(20, dtype=tf.float32), (4, 5))
    print(A)
    print(tf.reduce_sum(A))
    A_sum_axis0 = tf.reduce_sum(A, axis=0)
    print(A_sum_axis0)
    A_sum_axis1 = tf.reduce_sum(A, axis=1)
    print(A_sum_axis1)
    print(tf.reduce_mean(A))
    print(tf.reduce_mean(A) == tf.reduce_sum(A) / tf.size(A).numpy())
    A_sum_axis0_keepdims = tf.reduce_sum(A, axis=0, keepdims=True)
    print(A_sum_axis0_keepdims)
    print(A_sum_axis0_keepdims.shape)
    A_sum_axis1_keepdims = tf.reduce_sum(A, axis=1, keepdims=True)
    print(A_sum_axis1_keepdims)
    print(A_sum_axis1_keepdims.shape)
    print(tf.cumsum(A, axis=0))
    print(tf.cumsum(A, axis=1))


def testFunc3():
    # torch version
    x = torch.arange(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    print(x)
    print(y)
    x_dot_y = torch.dot(x, y)
    print(x_dot_y)
    print(torch.sum(x * y) == x_dot_y)
    print(x_dot_y.shape)
    A = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    print(A)
    assert A.shape[1] == len(x)
    print(torch.mv(A, x))
    B = torch.ones(4, 3)
    assert A.shape[1] == B.shape[0]
    print(torch.mm(A, B))
    print("********************************************************")
    # tf version
    x = tf.range(4, dtype=tf.float32)
    y = tf.ones(4, dtype=tf.float32)
    print(x)
    print(y)
    x_dot_y = tf.tensordot(x, y, axes=1)
    print(x_dot_y)
    print(tf.reduce_sum(x * y) == x_dot_y)
    print(x_dot_y.shape)
    A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
    print(A)
    assert A.shape[1] == len(x)
    tf.linalg.matvec(A, x)
    B = tf.ones((4, 3), tf.float32)
    assert A.shape[1] == B.shape[0]
    print(tf.matmul(A, B))


def testFunc4():
    # torch version
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))
    print(torch.abs(u).sum())
    X = torch.ones((4, 9))
    print(torch.norm(X))
    # tf version
    v = tf.constant([3.0, -4.0])
    print(tf.norm(u))
    print(tf.reduce_sum(tf.abs(u)))
    X = tf.norm(tf.ones((4, 9)))
    print(tf.norm(X))


if __name__ == '__main__':
    testFunc1()
    testFunc2()
    testFunc3()
    testFunc4()
