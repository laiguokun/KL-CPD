#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import cPickle as pickle
import math
import numpy as np
import os
import random
import sklearn.metrics
import time
import torch
import torch.nn as nn

from models import base
from models import LSTNet
from data_loader import DataLoader
from optim import Optim


# Simple GRU baseline
class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.RNN_hid_dim, self.var_dim),
        )

    # X_p:      batch_size x seq_len x var_dim
    # X_p_enc:  batch_size x seq_len x RNN_hid_dim
    # h_t:      1 x batch_size x RNN_hid_dim
    # y_t:      batch_size x var_dim
    def forward(self, X_p):
        X_p_enc, h_t = self.rnn_layer(X_p)
        h_t = h_t.squeeze(0)
        y_t = self.fc_layer(h_t)

        return y_t


# Y, L should be numpy array
def valid_epoch(loader, data, model, batch_size, Y_true, L_true):
    model.eval()

    Y_pred = []
    for inputs in loader.get_batches(data, batch_size, shuffle=False):
        X_p = inputs[0]
        Y_pred_batch = model(X_p)
        Y_pred.append(Y_pred_batch)
        #Y_pred.append(Y_pred_batch.data.cpu().numpy())
    Y_pred = torch.cat(Y_pred, 0)
    Y_pred = Y_pred.data.cpu().numpy()

    sqr_err = np.sum((Y_true - Y_pred)**2, axis=1)
    abs_err = np.sum(abs(Y_true - Y_pred), axis=1)
    mse, mae = np.mean(sqr_err), np.mean(abs_err)
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, sqr_err)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    eval_dict = {'sqr_err': sqr_err,
                 'abs_err': abs_err,
                 'Y_pred': Y_pred,
                 'L_pred': sqr_err,
                 'Y_true': Y_true,
                 'L_true': L_true,
                 'mse': mse, 'mae': mae, 'auc': auc}
    return eval_dict



# ========= Setup input argument =========#
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data_path', type=str, required=True, help='path to data in matlab format')
parser.add_argument('--trn_ratio', type=float, default=0.7,help='how much data used for training')
parser.add_argument('--val_ratio', type=float, default=0.8,help='how much data used for validation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--random_seed', type=int, default=1126,help='random seed')
parser.add_argument('--wnd_dim', type=int, required=True, default=10, help='window size (past and future)')
parser.add_argument('--sub_dim', type=int, default=1, help='dimension of subspace embedding')
# RNN hyperparemters
parser.add_argument('--model', type=str, default='RNN', help='RNN|LSTNet')
parser.add_argument('--CNN_hid_dim', type=int, default=10, help='number of CNN hidden units')
parser.add_argument('--RNN_hid_dim', type=int, default=10, help='number of RNN hidden units')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_dim', type=int, default=10, help='The window size of the highway component')
parser.add_argument('--RNN_skp_len', type=int, default=10, help='skip-length of RNN-skip layer in LSTNet')
parser.add_argument('--RNN_skp_dim', type=int, default=5, help='hidden units nubmer of RNN-skip layer')
parser.add_argument('--output_func', type=str, default=None, help='None|sigmoid|tanh for activation in last layer output')
parser.add_argument('--dropout', type=float, default=0., help='dropout applied to layers (0 = no dropout)')
# optimization
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--max_iter', type=int, default=100, help='max iteration for training')
parser.add_argument('--loss', type=str, default='L2', help='L1|L2|Huber for loss function')
parser.add_argument('--optim', type=str, default='adam', help='sgd|rmsprop|adam for optimization method')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 regularization)')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum for sgd')
parser.add_argument('--grad_clip', type=float, default=10.0, help='gradient clipping for RNN (both netG and netD)')
parser.add_argument('--eval_freq', type=int, default=25, help='evaluation frequency per generator update')

args = parser.parse_args()
print(args)
#assert(os.path.isdir(args.save_path))
assert(args.sub_dim == 1)


# ========= Setup GPU device and fix random seed=========#
if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu)
    print('Using GPU device', torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")
np.random.seed(seed=args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)


# ========= Load Dataset and initialize model=========#
Data = DataLoader(args, trn_ratio=args.trn_ratio, val_ratio=args.val_ratio)
if args.model == 'RNN':
    model = Model(args, Data)
elif args.model == 'LSTNet':
    model = LSTNet.Model(args, Data)
else:
    raise NotImplementedError('unknown model type %s! [RNN|LSTNet]' % (args.model))

if args.cuda:
    model.cuda()
params_count = sum([p.nelement() for p in model.parameters()])
print(model)
print('number of parameters: %d' % (params_count))


# ========= Setup loss function and optimizer  =========#
if args.loss == 'L1':
    criterion = nn.L1Loss(size_average=True)
elif args.loss == 'L2':
    criterion = nn.MSELoss(size_average=True)
elif args.loss == 'Huber':
    criterion = nn.SmoothL1Loss(size_average=True)
else:
    raise NotImplementedError('Loss function %s is not support! Consider L1|L2|Huber' % (args.loss_func))
if args.cuda:
    cirterion = criterion.cuda()
optimizer = Optim(model.parameters(),
                  args.optim,
                  lr=args.lr,
                  grad_clip=args.grad_clip,
                  weight_decay=args.weight_decay,
                  momentum=args.momentum)


# ========= Main loop for training  =========#
Y_val = Data.val_set['Y'].numpy()
L_val = Data.val_set['L'].numpy()
Y_tst = Data.tst_set['Y'].numpy()
L_tst = Data.tst_set['L'].numpy()

n_batchs = int(math.ceil(len(Data.trn_set['Y']) / float(args.batch_size)))
print('n_batchs', n_batchs, 'batch_size', args.batch_size)

update = 0
total_time = 0.0
best_epoch = -1
best_val_mae = 1e+6
best_val_auc = -1
best_tst_auc = -1
try:
    print('begin training')
    for epoch in range(1, args.max_iter + 1):
        trn_loader = Data.get_batches(Data.trn_set, batch_size=args.batch_size, shuffle=True)
        for bidx in range(n_batchs):
            model.train()
            start_time = time.time()
            inputs = next(trn_loader)
            X_p, X_f, Y_true = inputs[0], inputs[1], inputs[2]

            model.zero_grad()
            Y_pred = model(X_p)
            loss = criterion(Y_pred, Y_true)

            loss.backward()
            optimizer.step()
            update += 1

            # eval on val and tst set
            if update % args.eval_freq == 0:
                val_dict = valid_epoch(Data, Data.val_set, model, args.batch_size, Y_val, L_val)
                total_time = time.time() - start_time
                print('iter %4d tm %4.2fm trn_loss %.4e val_mse %.4e val_mae %.4e val_auc %.6f'
                      % (epoch, total_time / 60.0, loss.data[0], val_dict['mse'], val_dict['mae'], val_dict['auc']), end='')

                tst_dict = valid_epoch(Data, Data.tst_set, model, args.batch_size, Y_tst, L_tst)
                print (" tst_mse %.4e tst_mae %.4e tst_auc %.6f" % (tst_dict['mse'], tst_dict['mae'], tst_dict['auc']), end='')

                assert(np.isnan(val_dict['auc']) != True)
                # if val_dict['mae'] < best_val_mae:
                if val_dict['auc'] > best_val_auc:
                    best_val_mae = val_dict['mae']
                    best_val_auc = val_dict['auc']
                    best_tst_auc = tst_dict['auc']
                    best_epoch = epoch
                    #save_pred_name = '%s/%s.pred.pkl' % (args.save_path, args.save_name)
                    #with open(save_pred_name, 'wb') as f:
                    #    pickle.dump(tst_dict, f)
                    #torch.save(model.state_dict(), '%s/%s.model.pkl' % (args.save_path, args.save_name))
                print(" [best_val_auc %.6f best_tst_auc %.6f best_epoch %3d]" % (best_val_auc, best_tst_auc, best_epoch))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
