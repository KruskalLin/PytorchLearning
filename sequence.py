import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

def plot_decision_boundary(model, x, y):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
    plt.show()

np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12, N) + np.random.randn(N) * 0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
#
# plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
# plt.show()
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
# w = nn.Parameter(torch.randn(2, 1))
# b = nn.Parameter(torch.zeros(1))
# optimizer = torch.optim.SGD([w,b], 1e-1)
#
# def logistic_regression(x):
#     return torch.mm(x, w) + b
#
# criterion = nn.BCEWithLogitsLoss()
#
# for e in range(100):
#     out = logistic_regression(Variable(x))
#     loss = criterion(out, Variable(y))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (e + 1) % 20 == 0:
#         print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))
#
# def plot_logistic(x):
#     x = Variable(torch.from_numpy(x).float())
#     out = torch.sigmoid(logistic_regression(x))
#     out = (out > 0.5) * 1
#     return out.data.numpy()
#
# plot_decision_boundary(lambda x: plot_logistic(x), x.numpy(), y.numpy())
#
# w1 = nn.Parameter(torch.randn(2, 4) * 0.01) # 隐藏层神经元个数 2
# b1 = nn.Parameter(torch.zeros(4))
#
# w2 = nn.Parameter(torch.randn(4, 1) * 0.01)
# b2 = nn.Parameter(torch.zeros(1))
#
# def two_network(x):
#     x1 = torch.mm(x, w1) + b1
#     x1 = F.tanh(x1) # 使用 PyTorch 自带的 tanh 激活函数
#     x2 = torch.mm(x1, w2) + b2
#     return x2
#
# optimizer = torch.optim.SGD([w1, w2, b1, b2], 1.)
#
criterion = nn.BCEWithLogitsLoss()
#
# for e in range(10000):
#     out = two_network(Variable(x))
#     loss = criterion(out, Variable(y))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (e + 1) % 1000 == 0:
#         print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))
#
# def plot_network(x):
#     x = Variable(torch.from_numpy(x).float())
#     out = two_network(x)
#     out = (out > 0.5) * 1
#     return out.data.numpy()
#
# plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
# plt.title('2 layer network')

seq_net = nn.Sequential(
    nn.Linear(2, 4), # PyTorch 中的线性层，wx + b
    nn.Tanh(),
    nn.Linear(4, 1)
)

param = seq_net.parameters()
optimizer = torch.optim.SGD(param, 1e-1)

for e in range(10000):
    out = seq_net(Variable(x))
    loss = criterion(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))
def plot_seq(x):
    out = F.sigmoid(seq_net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out
plot_decision_boundary(lambda x: plot_seq(x), x.numpy(), y.numpy())

# torch.save(seq_net, 'save_seq_net.pth')
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
print(seq_net.state_dict())
torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')
seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))
print(seq_net2.state_dict())