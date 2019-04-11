import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('./cat.png').convert('L') # 读入一张灰度图的图片
im = np.array(im, dtype='float32') # 将其转换为一个矩阵
# plt.imshow(im.astype('uint8'), cmap='gray')
# plt.show()

im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
conv1 = nn.Conv2d(1, 1, 4, bias=False)
# sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
# sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
# conv1.weight.data = torch.from_numpy(sobel_kernel)
edge1 = conv1(Variable(im))
# print(edge1.shape)
edge1 = edge1.data.squeeze().numpy()
plt.imshow(edge1, cmap='gray')
plt.show()
# small_im2 = F.max_pool2d(Variable(edge1), 2, 2)
# small_im2 = small_im2.data.squeeze().numpy()
# plt.imshow(small_im2, cmap='gray')
# plt.show()