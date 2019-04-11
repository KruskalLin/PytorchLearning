import torch
from torch.autograd import Variable
from torch import nn

# rnn_single = nn.RNNCell(input_size=100, hidden_size=200)
# x = Variable(torch.randn(6, 5, 100)) # 这是 rnn 的输入格式
# h_t = Variable(torch.zeros(5, 200))
# out = []
# for i in range(6): # 通过循环 6 次作用在整个序列上
#     h_t = rnn_single(x[i], h_t)
#     out.append(h_t)
# print(h_t)
# h_0 = Variable(torch.randn(1, 5, 200))
# out, h_t = rnn_seq(x, h_0)
#
# rnn_seq = nn.RNN(100, 200)
# out, h_t = rnn_seq(x)
# gru_seq = nn.GRU(10, 20)

h_init = Variable(torch.zeros(2, 3, 100))
c_init = Variable(torch.zeros(2, 3, 100))
lstm_seq = nn.LSTM(50, 100, num_layers=2) # 输入维度 100，输出 200，两层
lstm_input = Variable(torch.randn(10, 3, 50))
out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))
