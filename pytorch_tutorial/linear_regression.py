# coding: utf-8
"""
@author: zc12345 
@contact: 18292885866@163.com

@file: linear_regression.py
@time: 2019/7/1 16:06
@description:

"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def autograd_example():
    x = torch.tensor(1., requires_grad=True)
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(3., requires_grad=True)

    y = w * x + b

    y.backward()

    print(x.grad)

    x = torch.randn(10, 3)
    y = torch.randn(10, 2)

    linear = nn.Linear(3, 2)
    print("linear weight:", linear.weight, ", bias:", linear.bias)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    for i in range(100):
        pred = linear(x)
        loss = criterion(pred, y)

        print("loss = ", loss.item())
        loss.backward()
        # print('dL/dw: ', linear.weight.grad)
        # print('dL/db: ', linear.bias.grad)

        optimizer.step()


def load_toy_data():
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    return x_train, y_train


def linear_regression_train():
    input_size = 1
    output_size = 1
    num_epoches = 100
    lr = 0.0005

    x_train, y_train = load_toy_data()
    model = nn.Linear(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epoches):
        x = torch.from_numpy(x_train)
        y = torch.from_numpy(y_train)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch {}/{}: loss:{:.4f}'.format(epoch + 1, num_epoches, loss.item()))
    torch.save(model.state_dict(), 'model.ckpt')


def plot():
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    model.load_state_dict(torch.load('model.ckpt'))
    x_train, y_train = load_toy_data()
    pred = model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, pred, label='Fitted line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # autograd_example()
    linear_regression_train()
    plot()