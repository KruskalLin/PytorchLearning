import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1234)

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
def linear_model(x):
    return x * w + b
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.show()
def get_loss(y_, y):
    return torch.mean((y_-y)**2)
loss = get_loss(y_, y_train)
print(loss)
loss.backward()
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data
for e in range(10):  # 进行 10 次更新
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)
    print(w)
    w.grad.zero_()  # 记得归零梯度
    b.grad.zero_()  # 记得归零梯度
    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: {}, loss: {}'.format(e, loss.data[0]))

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.show()