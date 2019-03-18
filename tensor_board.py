from torch import nn
import torch
import numpy as np
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter()
class ReLUtanh(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)           # relu就是截断负数，让所有负数等于0
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.tanh(grad_input)
        grad_input[input_ < 0] = 0               # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input
def relu(input_):
    return ReLUtanh()(input_)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(784, 400)
        self.layer2 = nn.Linear(400, 200)
        self.layer3 = nn.Linear(200, 100)
        self.layer4 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = relu(x)
        x = self.layer2(x)
        x = relu(x)
        x = self.layer3(x)
        x = relu(x)
        x = self.layer4(x)
        return x


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
mlp = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), 1e-1)
for e in range(0, 5):
    losses = []
    acces = []
    train_loss = 0
    train_acc = 0
    mlp.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        out = mlp(im)
        optimizer.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data[0].float()
        acc = num_correct / im.shape[0]
        train_acc += acc
    writer.add_scalars('Loss', {'train': train_loss / len(train_data)}, e)
    writer.add_scalars('Acc', {'train': train_acc / len(train_data)}, e)
