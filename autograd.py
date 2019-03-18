from torch.autograd import Variable
import torch
x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)
out = torch.mean(y - torch.matmul(x, w))
out.backward()
# print(w.grad)
# m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True) # 构建一个 1 x 2 的矩阵
# n = Variable(torch.zeros(1, 2)) # 构建一个相同大小的 0 矩阵
# print(m)
# print(n)
# n[0, 0] = m[0, 0] ** 2
# n[0, 1] = m[0, 1] ** 3
# print(n)
# n.backward(torch.ones_like(n)) # 将 (w0, w1) 取成 (1, 1)
# print(m.grad)

# x = Variable(torch.FloatTensor([3]), requires_grad=True)
# y = x * 2 + x ** 2 + 3
# print(y)
# y.backward(retain_graph=True)
# print(x.grad)
# y.backward(retain_graph=True)
# print(x.grad)
# y.backward()
# print(x.grad)

x = Variable(torch.Tensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros(2))
k[0] = x[0] ** 2 + 3 * x[1]
k[1] = x[0] * 2 + x[1] ** 2
j = torch.zeros(2, 2)
k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
j[0] = x.grad.data

x.grad.data.zero_() # 归零之前求得的梯度

k.backward(torch.FloatTensor([0, 1]))
j[1] = x.grad.data
print(j)

first_counter = torch.Tensor([0])
second_counter = torch.Tensor([10])
while (first_counter < second_counter)[0]:
    first_counter += 2
    second_counter += 1

print(first_counter)
print(second_counter)