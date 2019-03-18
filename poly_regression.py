import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

w_target = np.array([0.5, 3, 2.4]) # 定义参数
b_target = np.array([0.9]) # 定义参数
x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
# plt.plot(x_sample, y_sample, label='real curve')
# plt.show()
# print([x_sample ** i for i in range(1, 4)])
x_train = np.stack([x_sample ** i for i in range(1, 3)], axis=1)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)
w = Variable(torch.randn(2, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
x_train = Variable(x_train)
y_train = Variable(y_train)
def multi_linear(x):
    return torch.mm(x, w) + b

y_pred = multi_linear(x_train)
def get_loss(y_, y):
    return torch.mean((y_-y)**2)
loss = get_loss(y_pred, y_train)
# print(loss)
# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
# plt.show()
print(loss)
loss.backward()
w.data = w.data - 0.001 * w.grad.data
b.data = b.data - 0.001 * b.grad.data
y_pred = multi_linear(x_train)
# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
# plt.show()
for e in range(0, 1000):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    w.grad.data.zero_()
    b.grad.data.zero_()
    loss.backward()

    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    # print('epoch {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))
plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
plt.show()

