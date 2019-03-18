import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2017)

with open('./data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

x0 = list(filter(lambda x : x[-1] == 0.0, data))
x1 = list(filter(lambda x : x[-1] == 1.0, data))

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.show()

np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2]) # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# plot_x = np.arange(-10, 10.01, 0.01)
# plot_y = sigmoid(plot_x)
#
# plt.plot(plot_x, plot_y, 'r')
# plt.show()

x_data = Variable(x_data)
y_data = Variable(y_data)

w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))
#
# def logistic_regression(x):
#     return F.sigmoid(torch.mm(x, w) + b)
#
# optimizer = torch.optim.SGD([w, b], lr=1.)
# def logistic_regression(x):
#     return torch.sigmoid(torch.mm(x, w) + b)
def logistic_reg(x):
    return torch.mm(x, w) + b

criterion = nn.BCEWithLogitsLoss()

w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]

# def binary_loss(y_pred, y):
#     logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
#     return -logits

# y_pred = logistic_regression(x_data)
# loss = binary_loss(y_pred, y_data)
# print(loss)
# loss.backward()
# w.data = w.data - 0.1 * w.grad.data
# b.data = b.data - 0.1 * b.grad.data
# y_pred = logistic_regression(x_data)
# loss = binary_loss(y_pred, y_data)
# print(loss)
#
# for e in range(0, 10000):
#     y_pred = logistic_regression(x_data)
#     loss = binary_loss(y_pred, y_data)
#     loss.backward()
#     w.data = w.data - 0.01 * w.grad.data
#     b.data = b.data - 0.01 * b.grad.data
#     print(loss)
#
#
optimizer = torch.optim.SGD([w, b], lr=1.)

y_pred = logistic_reg(x_data)
loss = criterion(y_pred, y_data)
print(loss.data)
#
# for e in range(1000):
#     # 前向传播
#     y_pred = logistic_regression(x_data)
#     loss = binary_loss(y_pred, y_data) # 计算 loss
#     # 反向传播
#     optimizer.zero_grad() # 使用优化器将梯度归 0
#     loss.backward()
#     optimizer.step() # 使用优化器来更新参数
#     # 计算正确率
#     mask = y_pred.ge(0.5).float()
#     acc = (mask == y_data).sum().data[0] / y_data.shape[0]
#     if (e + 1) % 200 == 0:
#         print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data[0], acc))

for e in range(0, 1000):
    y_pred = logistic_reg(x_data)
    loss = criterion(y_pred, y_data)
    loss.backward()
    optimizer.step()
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.shape[0]
    if (e+1)%200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data[0], acc))





plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0.numpy() * plot_x - b0.numpy()) / w1.numpy()

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.show()