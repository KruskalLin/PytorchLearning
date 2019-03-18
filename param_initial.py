import numpy as np
import torch
from torch import nn
from torch.nn import init
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
# w1 = net1[0].weight
# b1 = net1[0].bias
# print(w1)
#
#  for layer in net1:
#     if isinstance(layer, nn.Linear): # 判断是否是线性层
#         param_shape = layer.weight.shape
param_shape = net1[0].weight.shape
print(torch.from_numpy(
    np.random.uniform(-np.sqrt(6) / (param_shape[0] + param_shape[-1]), np.sqrt(6) / (param_shape[0] + param_shape[-1]),
                      size=param_shape))) # 哪里写错了

init.xavier_uniform(net1[0].weight)
print(net1[0].weight)