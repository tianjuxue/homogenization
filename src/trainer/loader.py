import glob
import math
import numpy as np
import argparse
import sys
from collections import namedtuple
import matplotlib.pyplot as plt
from copy import deepcopy
import pdb
import torch
from torch import nn
from torch.nn import functional as torchF
from torch.autograd import Variable
import logging
import scipy.optimize as opt
import os
from .. import arguments
from sklearn.linear_model import Ridge, Lasso


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class Linear(nn.Module):

    def __init__(self, args):
        super(Linear, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


def get_features_C(Xin):
    Xin[:, 0] = Xin[:, 0] + 1
    Xin[:, 3] = Xin[:, 3] + 1
    X_new = Xin.copy()
    X_new[:, 0] = Xin[:, 0] * Xin[:, 0] + Xin[:, 2] * Xin[:, 2]
    X_new[:, 1] = Xin[:, 0] * Xin[:, 1] + Xin[:, 2] * Xin[:, 3]
    X_new[:, 2] = Xin[:, 1] * Xin[:, 1] + Xin[:, 3] * Xin[:, 3]
    X_new = X_new[:, [0, 1, 2, 4, 5]]
    return X_new


def compute_MSE(model, Xin, Xout):
    inBatch = torch.tensor(Xin).float()
    outBatch = torch.tensor(Xout).view(-1, 1).float()
    fhat = model(inBatch)
    f = outBatch
    SS_res = torch.sum((f - fhat)**2)
    MSE = SS_res / Xout.shape[0]
    return MSE.data.numpy()


class Trainer(object):

    def __init__(self, args, Xin, Xout):
        super(Trainer, self).__init__()
        self.args = args
        self.Xin = Xin
        self.Xout = Xout
        self._shuffle_data()

    def _shuffle_data(self):
        self.K = 5
        test_ratio = 0.1
        division = np.linspace(0, 1 - test_ratio, self.K + 1)
        n_samples = len(self.Xin)
        division = division * n_samples
        division = np.rint(division).astype(int)
        inds = np.random.permutation(n_samples).tolist()
        self.training_inds_list = []
        for i in range(self.K):
            self.training_inds_list.append(inds[division[i]:division[i + 1]])
        self.training_inds = inds[:division[self.K]]
        self.test_inds = inds[division[self.K]:]

    def LR(self):
        Xin_training = self.Xin[self.training_inds]
        Xout_training = self.Xout[self.training_inds]
        Xin_test = self.Xin[self.test_inds]
        Xout_test = self.Xout[self.test_inds]
        Xin_training = np.concatenate(
            (Xin_training[:, :-1], np.ones((len(Xin_training), 1))),  axis=1)
        Xin_test = np.concatenate(
            (Xin_test[:, :-1], np.ones((len(Xin_test), 1))),  axis=1)

        tmp = np.linalg.inv(np.matmul(Xin_training.transpose(), Xin_training))
        w = np.matmul(np.matmul(tmp, Xin_training.transpose()),
                      np.expand_dims(Xout_training, axis=1))
        y_predicted_training = np.matmul(Xin_training, w).flatten()
        y_predicted_test = np.matmul(Xin_test, w).flatten()
        train_MSE = np.sum(
            (y_predicted_training - Xout_training)**2) / len(Xout_training)
        test_MSE = np.sum((y_predicted_test - Xout_test)**2) / len(Xout_test)
        print("LR training MSE", train_MSE)
        print("LR test MSE", test_MSE)

    def train(self, k, lr, bsize, hidden_dim):
        self.args.lr = lr
        self.args.batch_ize = bsize
        self.args.hidden_dim = hidden_dim
        # self.args.wd = 1e-3

        cv_training_inds = []
        for i in range(self.K):
            if i is not k:
                cv_training_inds += self.training_inds_list[i]
        cv_validation_inds = self.training_inds_list[k]

        Xin_t = self.Xin[cv_training_inds]
        Xout_t = self.Xout[cv_training_inds]
        Xin_v = self.Xin[cv_validation_inds]
        Xout_v = self.Xout[cv_validation_inds]
        Xin_test = self.Xin[self.test_inds]
        Xout_test = self.Xout[self.test_inds]

        model = Net(args)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd)

        epoch_number = 1000
        log_interval = 100
        nTrain = len(Xin_t)
        step_number = nTrain // args.batch_ize

        for ep_num in range(epoch_number):
            for step in range(1, step_number):
                optimizer.zero_grad()
                inds = np.random.randint(0, nTrain, args.batch_ize)
                inBatch = torch.tensor(Xin_t[inds], requires_grad=True).float()
                outBatch = torch.tensor(Xout_t[inds]).view(-1, 1).float()
                fhat = model(inBatch)
                f = outBatch
                f_loss = torch.sum((f - fhat)**2)
                f_loss.backward()
                optimizer.step()

            if (ep_num + 1) % log_interval == 0:
                print('\nepoch:', ep_num + 1)
                print("MSE for training", compute_MSE(model, Xin_t, Xout_t),
                      "MSE for validation", compute_MSE(model, Xin_v, Xout_v))
                torch.save(model, MODEL_PATH + '/model_step_' + str(ep_num))

        print('\n\n')
        print("MSE for training", compute_MSE(model, Xin_t, Xout_t))
        print("MSE for testing", compute_MSE(model, Xin_test, Xout_test))

        return compute_MSE(model, Xin_v, Xout_v)


def load_data_all(args, prune=False):
    DATA_PATH_shear = 'saved_data_shear'
    DATA_PATH_normal = 'saved_data_normal'
    Xin_shear, Xout_shear = load_data_single(DATA_PATH_shear)
    Xin_normal, Xout_normal = load_data_single(DATA_PATH_normal)
    Xin = np.concatenate((Xin_shear, Xin_normal))
    Xout = np.concatenate((Xout_shear, Xout_normal))

    if prune:
        index = np.where(np.sum(np.absolute(Xin[:, :4]), axis=1) > 1e-10)
        Xin = Xin[index]
        Xout = Xout[index]

    Xin = get_features_C(Xin)
    args.input_dim = Xin.shape[1]
    return Xin, Xout


def load_data_single(file_path):
    files = glob.glob(os.path.join(file_path, '*.npy'))
    X_vec = []
    y_vec = []

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print("processed ", i, " files")
        data = np.load(f).item()['data'][1]
        X = np.concatenate((data[0], data[1]))
        y = data[2]
        X_vec.append(X)
        y_vec.append(y)

    X_input = np.asarray(X_vec)
    y_input = np.asarray(y_vec)
    return X_input, y_input


def scheduled_run():
    lr_list = [1e-1, 1e-2, 1e-3]
    bsize_list = [64, 128, 256]
    hidden_dim_list = [64, 128, 256]
    error_list = []
    for lr in lr_list:
        for bsize in bsize_list:
            for hidden_dim in hidden_dim_list:
                error = run_trainer(args, Xin, Xout, lr, bsize, hidden_dim)
                error_list.append(error)
    print(error_list)

if __name__ == '__main__':
    args = arguments.args
    MODEL_PATH = args.checkpoints_path
    # prune flag removes the repeated elements in Xin
    Xin, Xout = load_data_all(args, prune=False)
    trainer = Trainer(args, Xin, Xout)
    trainer.LR()
    exit()
    error = trainer.train(1, 1e-2, 32, 256)


# def run_trainer(args, Xin, Xout, lr, bsize, hidden_dim):
#     args.lr = lr
#     args.batch_ize = bsize
#     args.hidden_dim = hidden_dim

#     nSamps = len(Xin)
#     nTrain = int((1 - args.val_fraction) * nSamps)
#     inds = np.random.permutation(nSamps)
#     indsTrain = inds[:nTrain]
#     indsTest = inds[nTrain:]

#     Xt_in = Xin[indsTest]
#     Xt_out = Xout[indsTest]

#     Xin = Xin[indsTrain]
#     Xout = Xout[indsTrain]

#     nTest = len(Xt_in)
#     fAv = fAvTest = f_dev = f_dev_test = 0.

#     model = Linear(args)
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=args.lr, weight_decay=args.wd)

#     if not os.path.exists(MODEL_PATH):
#         os.mkdir(MODEL_PATH)

#     epoch_number = 1000
#     log_interval = 100
#     step_number = nTrain // args.batch_ize

#     epoch = []
#     mse_train = []
#     mse_test = []
#     mpe_train = []
#     mpe_test = []

#     for ep_num in range(epoch_number):
#         for step in range(1, step_number):
#             optimizer.zero_grad()

#             # Testing
#             inds_test = np.random.randint(0, nTest, args.batch_ize)
#             inBatch_test = torch.tensor(
#                 Xt_in[inds_test], requires_grad=True).float()
#             outBatch_test = torch.tensor(Xt_out[inds_test]).view(-1, 1).float()
#             fhat_test = model(inBatch_test)
#             f_test = outBatch_test
#             f_loss_test = torch.sum((f_test - fhat_test)**2)
#             fAvTest += float(f_loss_test.data)
#             # mean percentage error (MPE)
#             f_dev_test += float(torch.sum(abs(f_test -
# fhat_test) / (f_test + 1e-5)).data)

#             # Training
#             inds = np.random.randint(0, nTrain, args.batch_ize)
#             inBatch = torch.tensor(Xin[inds], requires_grad=True).float()
#             outBatch = torch.tensor(Xout[inds]).view(-1, 1).float()
#             fhat = model(inBatch)
#             f = outBatch
#             f_loss = torch.sum((f - fhat)**2)
#             fAv += float(f_loss.data)
#             f_dev += float(torch.sum(abs(f - fhat) / (f + 1e-5)).data)

#             total_loss = f_loss
#             total_loss.backward()
#             optimizer.step()

#         if (ep_num + 1) % log_interval == 0:
#             print('\nepoch:', ep_num + 1)
#             epoch.append(ep_num)
#             train_mse = compute_MSE(model, Xin, Xout)
#             test_mse = compute_MSE(model, Xt_in, Xt_out)
#             print("MSE for training", train_mse, "MSE for testing", test_mse)
#             mse_train.append(train_mse)
#             mse_test.append(test_mse)
#             # torch.save(model, MODEL_PATH + '/model_step_'+str(ep_num))
#             fAv = fAvTest = f_dev = f_dev_test = 0.

#     print('\n\n')
#     print("MSE for training", compute_MSE(model, Xin, Xout))
#     print("MSE for testing", compute_MSE(model, Xt_in, Xt_out))

#     return compute_MSE(model, Xt_in, Xt_out)
