# -*- coding: utf-8 -*-
import torch
import tensorflow as tf
import os
import warnings

warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # torch version
    x = torch.arange(4.0)
    print(x)
    x.requires_grad_(True)
    print(x.grad)
    y = 2 * torch.dot(x, x)
    print(y)
    y.backward()
    print(x.grad)
    print(x.grad == 4 * x)
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(x.grad)
    print(x.grad == torch.ones_like(x))
    # 对非标量调用`backward`需要传入一个`gradient`参数，该参数指定微分函数关于`self`的梯度。
    # 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
    x.grad.zero_()
    y = x * x
    print(y)
    # 等价于y.backward(torch.ones(len(x)))
    y.sum().backward()
    print(x.grad)
    x.grad.zero_()
    y = x * x
    u = y.detach()
    z = u * x
    z.sum().backward()
    print(x.grad == u)


    def f(a):
        b = a * 2
        while b.norm() < 1000:
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c


    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()
    print(a.grad == d / a)

    print("********************************************************")

    # tf version
    x = tf.range(4, dtype=tf.float32)
    x = tf.Variable(x)
    print(x)
    # 把所有计算记录在磁带上
    with tf.GradientTape() as t:
        y = 2 * tf.tensordot(x, x, axes=1)
    print(y)
    x_grad = t.gradient(y, x)
    print(x_grad)
    print(x_grad == 4 * x)
    with tf.GradientTape() as t:
        y = tf.reduce_sum(x)
    print(y)
    x_grad = t.gradient(y, x)
    print(x_grad)  # 被新计算的梯度覆盖
    print(x_grad == tf.ones_like(x))
    with tf.GradientTape() as t:
        y = x * x
    print(y)
    print(t.gradient(y, x))  # 等价于 `y = tf.reduce_sum(x * x)`
    # 设置 `persistent=True` 来运行 `t.gradient`多次
    with tf.GradientTape(persistent=True) as t:
        y = x * x
        u = tf.stop_gradient(y)
        z = u * x
    x_grad = t.gradient(z, x)
    print(x_grad == u)
    print(t.gradient(y, x) == 2 * x)


    def g(a):
        b = a * 2
        while tf.norm(b) < 1000:
            b = b * 2
        if tf.reduce_sum(b) > 0:
            c = b
        else:
            c = 100 * b
        return c


    a = tf.Variable(tf.random.normal(shape=()))
    with tf.GradientTape() as t:
        d = g(a)
    d_grad = t.gradient(d, a)
    print(d_grad == d / a)
