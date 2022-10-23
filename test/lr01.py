#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2022/10/23 6:49 下午
# @Author:Andrew duokunzhang@163.com
# @File：lr01.py 手动写的LR，以来torch的自动微分功能

import matplotlib.pyplot as plt
import torch

# config

lr = 0.01
max_iterations = 1000
start_iterrations = 0

# ----------------------step 1 data------------------
x = torch.rand(size=(200, 1)) * 10

y = 2 * x + (5 + torch.randn(200, 1))

# ---------------------step 2 model-----------------
w = torch.randn((1), requires_grad=True)
b = torch.ones((1), requires_grad=True)

for start_iterations in range(max_iterations):

    # forward
    y_pred = torch.add(torch.mul(w, x), b)

    # loss
    loss_fun = (0.5 * (y - y_pred) ** 2).mean()

    # backward
    loss_fun.backward()

    # optimizer
    w.data.sub_(lr * w.grad)
    b.data.sub_(lr * b.grad)

    # 梯度清零

    w.grad.zero_()
    b.grad.zero_()

    # 绘图
    if start_iterations % 50 == 0:
        plt.scatter(x.detach().numpy(), y.detach().numpy())
        plt.plot(x.detach().numpy(), y_pred.detach().numpy(), 'r-', lw=6)
        plt.text(2, 22, 'Loss = %.5f' % loss_fun.detach().numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(0, 10)
        plt.ylim(0, 20)

        plt.pause(0.5)
        plt.show()

        print("epoch : %d, train loss : %.3f, (w = %.2f, b = %.2f)" % (start_iterations, loss_fun.detach().numpy(), w.data, b.data))
