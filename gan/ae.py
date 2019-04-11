import os

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 标准化
])

train_set = MNIST('../data', transform=im_tfs, download=True)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Linear(28 * 28, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 3)  # 输出的 code 是 3 维，便于可视化
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 28 * 28),
        #     nn.Tanh()
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode
net = autoencoder()
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    print(x.shape)
    x = x.clamp(0, 1)
    print(x.shape)
    x = x.view(x.shape[0], 1, 28, 28)
    print(x.shape)
    return x


# for e in range(100):
#     for im, _ in train_data:
#         im = im.view(im.shape[0], -1)
#         im = Variable(im)
#         # 前向传播
#         _, output = net(im)
#         loss = criterion(output, im) / im.shape[0]  # 平均
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (e + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
#         print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data[0]))
#         pic = to_img(output.cpu().data)
#         if not os.path.exists('./simple_autoencoder'):
#             os.mkdir('./simple_autoencoder')
#         save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 可视化结果
view_data = Variable((train_set.train_data[:200].type(torch.FloatTensor).view(-1, 28*28) / 255. - 0.5) / 0.5)
encode, _ = net(view_data)    # 提取压缩的特征值
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
values = train_set.train_labels[:200].numpy()  # 标签值
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色
    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()


