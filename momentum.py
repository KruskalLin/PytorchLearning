import numpy as np
import torch
from torchvision.datasets import MNIST # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def data_tf(x):
    x = np.array(x, dtype='float32') / 255 # 将数据变到 0 ~ 1 之间
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

train_set = MNIST('./data', train=True, transform=data_tf, download=True) # 载入数据集，申明定义的数据变换
test_set = MNIST('./data', train=False, transform=data_tf, download=True)

# 定义 loss 函数
criterion = nn.CrossEntropyLoss()

def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data

def momentum(parameters, vs, lr, gamma):
    for param, v in zip(parameters, vs):
        v[:] = gamma * v + lr * param.grad.data
        param.data = param.data - v

train_data = DataLoader(train_set, batch_size=128, shuffle=True)

net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

train_losses = []
idx = 0
# optimizer = torch.optim.SGD(net.parameters(), lr=1., momentum=0.9)
for e in range(0, 5):
    train_loss = 0.0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        loss = criterion(net(im), label)
        net.zero_grad()
        loss.backward()
        sgd_update(net.parameters(), 1e-2)
        train_loss += loss.data[0]
        if idx % 30 == 0:
            train_losses.append(loss.data[0])
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_data)))
x_axis = np.linspace(0, 5, len(train_losses), endpoint=True)
plt.semilogy(x_axis, train_losses, label='sgd')

train_losses = []
idx = 0
vs = []
for param in net.parameters():
    vs.append(torch.zeros_like(param.data))
for e in range(0, 5):
    train_loss = 0.0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        loss = criterion(net(im), label)
        net.zero_grad()
        loss.backward()
        momentum(net.parameters(), vs, 1e-2, 0.9)
        train_loss += loss.data[0]
        if idx % 30 == 0:
            train_losses.append(loss.data[0])
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_data)))
x_axis = np.linspace(0, 5, len(train_losses), endpoint=True)
plt.semilogy(x_axis, train_losses, label='momentum')
plt.show()