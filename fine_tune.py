import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder

import os
from PIL import Image
import matplotlib.pyplot as plt

root_path = './hymenoptera_data/train/'
im_list = [os.path.join(root_path, 'ants', i) for i in os.listdir(root_path + 'ants')[:4]]
im_list += [os.path.join(root_path, 'bees', i) for i in os.listdir(root_path + 'bees')[:5]]

nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(Image.open(im_list[nrows*i+j]))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()

train_tf = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用 ImageNet 的均值和方差
])

valid_tf = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 使用 ImageFolder 定义数据集
train_set = ImageFolder('./hymenoptera_data/train/', train_tf)
valid_set = ImageFolder('./hymenoptera_data/val/', valid_tf)
# 使用 DataLoader 定义迭代器
train_data = DataLoader(train_set, 64, True, num_workers=4)
valid_data = DataLoader(valid_set, 128, False, num_workers=4)

net = models.resnet50(pretrained=True)
print(net)

print(net.conv1.weight)

net.fc = nn.Linear(2048, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

from utils import train
train(net, train_data, valid_data, 20, optimizer, criterion)

net = net.eval() # 将网络改为预测模式

im1 = Image.open('./hymenoptera_data/train/ants/0013035.jpg')

im = valid_tf(im1) # 做数据预处理
out = net(Variable(im.unsqueeze(0)).cuda())
pred_label = out.max(1)[1].data[0]
print('predict label: {}'.format(train_set.classes[pred_label]))

net = models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False # 将模型的参数设置为不求梯度
net.fc = nn.Linear(2048, 2)

optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-2, weight_decay=1e-4)
train(net, train_data, valid_data, 20, optimizer, criterion)
net = models.resnet50()
net.fc = nn.Linear(2048, 2)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
train(net, train_data, valid_data, 20, optimizer, criterion)