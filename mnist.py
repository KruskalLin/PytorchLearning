import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

def data_tf(x):
    x = np.array(x, dtype='float32') / 255 # 归一
    x = (x - 0.5) / 0.5 # 标准化
    x = x.flatten() # 拉平
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)
losses = []
acces = []
eval_losses = []
eval_acces = []
for e in range(0, 5):

    train_loss = 0.0
    train_acc = 0.0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        optimizer.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0].float()
        acc = num_correct / im.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    eval_loss = 0.0
    eval_acc = 0.0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.data[0]
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0].float()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

print(losses)

plt.plot(np.arange(len(losses)), losses)
plt.show()