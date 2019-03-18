import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# numpy_tensor = np.random.randn(10, 20)
# torch_tensor = torch.Tensor(numpy_tensor)
# print((torch_tensor.numpy()))
# print(torch_tensor.size())
# print(torch_tensor.type())
# print(torch_tensor.dim())
# print(torch_tensor.numel())
#
# torch_tensor = torch.randn(3,2)
# torch_tensor = torch_tensor.type(torch.DoubleTensor)
# # print(torch_tensor.numpy().dtype)
# # max_value, max_idx = torch.max(torch_tensor, dim=1)
# # print(max_value)
# # print(max_idx)
# # print(torch.sum(torch_tensor, dim=1))
# print(torch_tensor)
# print(torch_tensor.unsqueeze(0))
# print(torch_tensor.unsqueeze(2))
# print(torch_tensor.unsqueeze(2).squeeze(2))

# x = torch.randn(3, 4, 5)
# print(x)
# x = x.permute(2, 0, 1)
# print(x)

# x = torch.ones(3, 3)
# print(x)
# x.unsqueeze_(0)
# print(x)

x = np.arange(-3, 3.01, 0.1)
y = x ** 2
# plt.plot(x, y)
# plt.plot(2, 4, 'ro')
# plt.show()
x = Variable(torch.FloatTensor([2]), requires_grad=True)
y = x ** 2
y.backward()
print(x.grad.data)