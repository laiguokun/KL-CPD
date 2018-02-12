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
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import mmd_util
from data_loader import DataLoader
from optim import Optim


# Simple GRU baseline
class NetD(nn.Module):
    def __init__(self, args, data):
        super(NetD, self).__init__()

        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    # X_p:      batch_size x seq_len x var_dim
    # X_p_enc:  batch_size x seq_len x RNN_hid_dim
    # h_t:      1 x batch_size x RNN_hid_dim
    # X_p_dec:  batch_size x seq_len x var_dim
    def forward(self, X):
        #batch_size = X_p.size(0)
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


# Y, L should be numpy array
def valid_epoch(loader, data, netD, batch_size, Y_true, L_true):
    netD.eval()
    Y_pred = []
    for inputs in loader.get_batches(data, batch_size, shuffle=False):
        X_p, X_f = inputs[0], inputs[1]
        batch_size = X_p.size(0)

        X_p_enc, _ = netD(X_p)
        X_f_enc, _ = netD(X_f)
        Y_pred_batch = mmd_util.batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)
        Y_pred.append(Y_pred_batch.data.cpu().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)

    L_pred = Y_pred
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, L_pred)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    eval_dict = {'Y_pred': Y_pred,
                 'L_pred': L_pred,
                 'Y_true': Y_true,
                 'L_true': L_true,
                 'mse': -1, 'mae': -1, 'auc': auc}
    return eval_dict



# ========= Setup input argument =========#
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data_path', type=str, required=True, help='path to data in matlab format')
parser.add_argument('--trn_ratio', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--val_ratio', type=float, default=0.8,help='how much data used for validation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--random_seed', type=int, default=1126,help='random seed')

parser.add_argument('--wnd_dim', type=int, required=True, default=10, help='window size (past and future)')
parser.add_argument('--sub_dim', type=int, default=1, help='dimension of subspace embedding')

# RNN hyperparemters
parser.add_argument('--model', type=str, default='MMD_negsample', help='MMD_negsample')
parser.add_argument('--RNN_hid_dim', type=int, default=10, help='number of RNN hidden units')
parser.add_argument('--lambda_ae', type=float, default=1., help='coefficient for the reconstruction loss')
parser.add_argument('--lambda_real', type=float, default=1., help='coefficient for the adversarial loss')

# optimization
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--max_iter', type=int, default=100, help='max iteration for pretraining RNN')
parser.add_argument('--optim', type=str, default='adam', help='sgd|rmsprop|adam for optimization method')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 regularization)')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum for sgd')
parser.add_argument('--grad_clip', type=float, default=10.0, help='gradient clipping for RNN (both netG and netD)')
parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency per generator update')

# save models
parser.add_argument('--save_path', type=str,  default='./exp_simulate/jumpingmean/save_RNN',help='path to save the final model')
parser.add_argument('--save_name', type=str,  default='tmp',help='model/prediction names')

args = parser.parse_args()
print(args)
assert(os.path.isdir(args.save_path))
assert(args.sub_dim == 1)

# if model == dataspace, no need to pretrain RNN
# directly evaluate MMD on raw data
if args.model == 'MMD_dataspace':
    args.max_iter = 10

#XXX For Yahoo dataset, trn_ratio=0.50, val_ratio=0.75
if 'yahoo' in args.data_path:
    args.trn_ratio = 0.50
    args.val_ratio = 0.75


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
# [INFO] cudnn.benckmark=True enable cudnn auto-tuner to find the best algorithm to use for your hardware
# [INFO] benchmark mode is good whenever input sizes of network do not vary much!!!
# [INFO] https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# [INFO] https://discuss.pytorch.org/t/pytorch-performance/3079/2
cudnn.benchmark == True

# [INFO} For reproducibility and debugging, set cudnn.enabled=False
# [INFO] Some operations are non-deterministic when cudnn.enabled=True
# [INFO] https://discuss.pytorch.org/t/non-determinisic-results/459
# [INFO] https://discuss.pytorch.org/t/non-reproducible-result-with-gpu/1831
cudnn.enabled = True

# ========= Load Dataset and initialize model=========#
Data = DataLoader(args, trn_ratio=args.trn_ratio, val_ratio=args.val_ratio)
netD = NetD(args, Data)

if args.cuda:
    netD.cuda()
params_count = sum([p.nelement() for p in netD.parameters()])
print(netD)
print('number of parameters: %d' % (params_count))


# ========= Setup loss function and optimizer  =========#
optimizerD = Optim(netD.parameters(),
                   args.optim,
                   lr=args.lr,
                   grad_clip=args.grad_clip,
                   weight_decay=args.weight_decay,
                   momentum=args.momentum)


# sigma for mixture of RBF kernel in MMD
sigma_list = [1.0]
#sigma_list = mmd_util.median_heuristic(Data.Y_subspace, beta=1.)
#sigma_list = mmd_util.median_heuristic(Data.Y_subspace, beta=.5)
sigma_var = torch.FloatTensor(sigma_list).cuda()
print('sigma_list:', sigma_var)


# ========= Main loop for adversarial training kernel with negative samples X_f + noise =========#
Y_val = Data.val_set['Y'].numpy()
L_val = Data.val_set['L'].numpy()
Y_tst = Data.tst_set['Y'].numpy()
L_tst = Data.tst_set['L'].numpy()

n_batchs = int(math.ceil(len(Data.trn_set['Y']) / float(args.batch_size)))
print('n_batchs', n_batchs, 'batch_size', args.batch_size)

lambda_ae = args.lambda_ae
lambda_real = args.lambda_real

update = 0
total_time = 0.
best_epoch = -1
best_val_mae = 1e+6
best_val_auc = -1
best_tst_auc = -1
start_time = time.time()
for epoch in range(1, args.max_iter + 1):
    trn_loader = Data.get_batches(Data.trn_set, batch_size=args.batch_size, shuffle=True)
    for bidx in range(n_batchs):
        netD.train()
        inputs = next(trn_loader)
        X_p, X_f, Y_true = inputs[0], inputs[1], inputs[2]

        netD.zero_grad()
        X_p_enc, X_p_dec = netD(X_p)     # batch_size x seq_len x RNN_hid_dim
        X_f_enc, X_f_dec = netD(X_f)     # batch_size x seq_len x RNN_hid_dim
        batch_size, seq_len, nz = X_p_enc.size()

        # MMD on real data
        MMD2_real = mmd_util.batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)
        MMD2_real = MMD2_real.mean()

        # MMD on perturbed fake data
        std = torch.std(X_f_enc) / (1. + 0.1 * (epoch - 1))
        noise = torch.cuda.FloatTensor(batch_size, seq_len, nz).normal_(0, std.data[0])
        noise = Variable(noise)
        X_f_enc_fake = X_f_enc + noise
        MMD2_fake = mmd_util.batch_mmd2_loss(X_f_enc, X_f_enc_fake, sigma_var)
        MMD2_fake = MMD2_fake.mean()

        # reconstruction loss
        L2_loss = torch.mean((X_p_dec - X_p)**2) + torch.mean((X_f_dec - X_f)**2)

        #loss = MMD2_real - lambda_neg * MMD2_fake + lambda_ae * L2_loss
        loss = -MMD2_fake + lambda_ae * L2_loss + lambda_real * MMD2_real
        loss.backward()
        optimizerD.step()
        update += 1

        if update % args.eval_freq == 0:
            # ========= Main block for evaluate MMD(X_p_enc, X_f_enc) on RNN codespace  =========#
            val_dict = valid_epoch(Data, Data.val_set, netD, args.batch_size, Y_val, L_val)
            tst_dict = valid_epoch(Data, Data.tst_set, netD, args.batch_size, Y_tst, L_tst)
            total_time += time.time() - start_time
            print('iter %4d tm %4.2fm MMD2_real %.6f MMD2_fake %.6f val_mse %.4f val_mae %.4f val_auc %.6f'
                    % (epoch, total_time / 60.0, MMD2_real.data[0], MMD2_fake.data[0],
                       val_dict['mse'], val_dict['mae'], val_dict['auc']), end='')

            print (" tst_mse %.4f tst_mae %.4f tst_auc %.6f" % (tst_dict['mse'], tst_dict['mae'], tst_dict['auc']), end='')

            assert(np.isnan(val_dict['auc']) != True)
            if val_dict['auc'] > best_val_auc:
                best_val_mae = val_dict['mae']
                best_val_auc = val_dict['auc']
                best_tst_auc = tst_dict['auc']
                best_epoch = epoch
                save_pred_name = '%s/%s.pred.pkl' % (args.save_path, args.save_name)
                with open(save_pred_name, 'wb') as f:
                    pickle.dump(tst_dict, f)
                torch.save(netD.state_dict(), '%s/%s.netD.pkl' % (args.save_path, args.save_name))
            print(" [best_val_auc %.6f best_tst_auc %.6f best_epoch %3d]" % (best_val_auc, best_tst_auc, best_epoch))

