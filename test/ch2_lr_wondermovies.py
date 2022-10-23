#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2022/10/23 7:21 下午
# @Author:Andrew duokunzhang@163.com
# @File：ch2_lr_wondermovies.py 预测用户使用wondermovies平台的平均小时数

import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from torch.autograd import Variable


# Training Data
def get_data():
    train_X = np.asarray(
        [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray(
        [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype), requires_grad=False).view(17, 1)
    y = Variable(torch.from_numpy(train_Y).type(dtype), requires_grad=False)
    # plot_variable(X, y)
    return X, y


def plot_variable(x, y, z='', **kwargs):
    l = []
    for a in [x, y]:
        # if type(a) == Variable:
        l.append(a.data.numpy())
    plt.figure()
    plt.plot(l[0], l[1], z, **kwargs)
    plt.show()


def get_weights():
    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)
    return w, b


def simple_network(x):
    y_pred = torch.matmul(x, w) + b
    return y_pred


def loss_fn(y, y_pred):
    loss = (y_pred - y).pow(2).sum()
    for param in [w, b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()  # 计算梯度
    return loss.data.item()


def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


learning_rate = 1e-4

# if __name__ == "__main__":
#     x: object
x, y = get_data()  # x - represents training data,y - represents target variables
w, b = get_weights()  # w,b - Learnable parameters
plot_variable(x, y, 'ro')

for i in range(5000):
    y_pred = simple_network(x)  # function which computes wx + b

    loss = loss_fn(y, y_pred)  # calculates sum of the squared differences of y and y_pred
    if i % 100 == 0:
        print(loss)

    optimize(learning_rate)

print(w.data, b.data.item())
plot_variable(x, y_pred, label='Fitted line')