#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, args, data):
        super(EncoderRNN, self).__init__()
        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.hid_dim = args.RNN_hid_dim

        self.rnn_layer = nn.GRU(self.var_dim, self.hid_dim, batch_first=True)

    # X: batch_size x wnd_dim x var_dim
    # hidden: 1 x batch_size x hid_dim (init hidden state of Encoder)
    def forward(self, X, hidden=None):
        X_enc, hidden = self.rnn_layer(X, hidden)
        return X_enc, hidden

    def init_hidden(self, batch_size, use_cuda=True):
        h_0 = Variable(torch.zeros(1, batch_size, self.hid_dim))
        if use_cuda:
            return h_0.cuda()
        else:
            return h_0


class DecoderRNN(nn.Module):
    def __init__(self, args, data):
        super(DecoderRNN, self).__init__()
        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.hid_dim = args.RNN_hid_dim

        self.rnn_layer = nn.GRU(self.var_dim, self.hid_dim, batch_first=True)
        self.fc_layer = nn.Linear(self.hid_dim, self.var_dim)
        self.activation = nn.Sigmoid()

    # X: batch_size x wnd_dim x var_dim
    # hidden: 1 x batch_size x hid_dim (last hidden state of Encoder)
    def forward(self, X, hidden=None):
        X_shft = self.shft_right_one(X)
        X_dec, hidden = self.rnn_layer(X_shft, hidden)
        output = self.fc_layer(X_dec)
        output = self.activation(output)
        return output, hidden


    # X: batch_size x wnd_dim x var_dim
    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft

'''
class Conv1dDecoder(nn.Module):
    def __init__(self, args, data):
        super(Conv1dDecoder, self).__init__()
        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.kernel_size = args.kernel_size
        assert(self.kernel_size % 2 == 1) # CNN kernel size need to be odd numbers
        self.pad_size = self.kernel_size // 2
        self.conv1d_layer = nn.Conv1d(args.RNN_hid_dim, self.var_dim, self.kernel_size,
                                      stride=1, padding=self.pad_size)

    # X_enc: batch_size x wnd_dim x var_dim
    # noise: batch_size x wnd_dim x var_dim
    def forward(self, X_enc):
        # X_enc = X_enc + noise
        # print('Conv1dDecoder: X_enc', X_enc.size())
        X_enc = X_enc.permute(0, 2, 1).contiguous()
        # print('Conv1dDecoder: X_enc', X_enc.size())
        X_dec = self.conv1d_layer(X_enc)
        # print('Conv1dDecoder: X_dec', X_dec.size())
        X_dec = X_dec.permute(0, 2, 1).contiguous()
        # print('Conv1dDecoder: X_dec', X_dec.size())
        # exit(0)
        return X_dec
'''
