from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
import numpy as np
im = Image.open('./cat.png')
print('before scale, shape: {}'.format(im.size))
new_im = tfs.Resize((100, 200))(im)
print('after scale, shape: {}'.format(new_im.size))
new_im = np.array(new_im, dtype='float32') # 将其转换为一个矩阵
# random_im2 = tfs.RandomCrop((150, 100))(im)
# center_im = tfs.CenterCrop(100)(im)
# h_filp = tfs.RandomHorizontalFlip()(im)
# v_flip = tfs.RandomVerticalFlip()(im)
# rot_im = tfs.RandomRotation(45)(im)
# bright_im = tfs.ColorJitter(brightness=1)(im) # 0 - 2
# bright_im = np.array(bright_im, dtype='float32') # 将其转换为一个矩阵
# contrast_im = tfs.ColorJitter(contrast=1)(im) # 0-2
# color_im = tfs.ColorJitter(hue=0.5)(im) # color -0.5 - 0.5
im_aug = tfs.Compose([
    tfs.Resize(120),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(96),
    tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
])
nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(im_aug(im))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
plt.show()
# 数据增强提高了模型应对于更多的不同数据集的泛化能力

# plt.imshow(bright_im.astype('uint8'))
# plt.show()