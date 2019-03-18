from torchvision.datasets import ImageFolder
from torchvision import transforms as tfs

# folder_set = ImageFolder('./example_data/image/')
#
# im, label = folder_set[0]
# print(label)
data_tf = tfs.ToTensor()

folder_set = ImageFolder('./example_data/image/', transform=data_tf)

im, label = folder_set[0]
print(im)

from torch.utils.data import Dataset


class custom_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform  # 传入数据预处理
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        self.img_list = [i.split()[0] for i in lines]  # 得到所有的图像名字
        self.label_list = [i.split()[1] for i in lines]  # 得到所有的 label

    def __getitem__(self, idx):  # 根据 idx 取出其中一个
        img = self.img_list[idx]
        label = self.label_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):  # 总数据的多少
        return len(self.label_list)

txt_dataset = custom_dataset('./example_data/train.txt') # 读入 txt 文件
# data, label = txt_dataset[0]
# print(data)
# print(label)

from torch.utils.data import DataLoader

train_data1 = DataLoader(folder_set, batch_size=2, shuffle=True) # 将 2 个数据作为一个 batch
for im, label in train_data1: # 访问迭代器
    print(label)

train_data2 = DataLoader(txt_dataset, 8, True) # batch size 设置为 8

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True) # 将数据集按照 label 的长度从大到小排序
    img, label = zip(*batch) # 将数据和 label 配对取出
    # 填充
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += '0' * (max_len - len(label[i]))
        pad_label.append(temp_label)
        lens.append(len(label[i]))

    return img, pad_label, lens # 输出 label 的真实长度

train_data3 = DataLoader(txt_dataset, 8, True, collate_fn=collate_fn) # batch size 设置为 8
im, label, lens = next(iter(train_data3))
print(label)