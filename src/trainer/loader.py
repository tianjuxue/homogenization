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
from .screener import load_data_all, load_data_single
from .. import arguments
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures


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


class Trainer(object):

    def __init__(self, args, Xin, Xout):
        super(Trainer, self).__init__()
        self.args = args
        self.Xin = Xin
        self.Xout = Xout
        self.shuffle_data()

    def shuffle_data(self, K=5, test_ratio=0.1):
        division = np.linspace(0, 1 - test_ratio, K + 1)
        n_samples = len(self.Xin)
        division = division * n_samples
        division = np.rint(division).astype(int)
        inds = np.random.permutation(n_samples).tolist()
        self.training_inds_list = []
        for i in range(K):
            self.training_inds_list.append(inds[division[i]:division[i + 1]])
        self.training_inds = inds[:division[K]]
        self.test_inds = inds[division[K]:]
        self.K = K

    def polynomial_regression(self, degree):
        Xin_training = self.Xin[self.training_inds]
        Xout_training = self.Xout[self.training_inds]
        Xin_test = self.Xin[self.test_inds]
        Xout_test = self.Xout[self.test_inds]
        category_training = Xin_training[:, -1:]
        category_test = Xin_test[:, -1:]

        Xin_training = Xin_training[:, :-1]
        Xin_test = Xin_test[:, :-1]
        poly = PolynomialFeatures(degree)
        Xin_training = poly.fit_transform(Xin_training)
        Xin_test = poly.fit_transform(Xin_test)
        Xin_training = np.concatenate((Xin_training, category_training), axis=1)
        Xin_test = np.concatenate((Xin_test, category_test), axis=1)

        print(np.linalg.matrix_rank(Xin_training))
        print(np.linalg.matrix_rank(np.matmul(Xin_training.transpose(), Xin_training)))
        print(Xin_training.shape[1])

        # Analytical form
        tmp = np.linalg.inv(np.matmul(Xin_training.transpose(), Xin_training))
        w = np.matmul(np.matmul(tmp, Xin_training.transpose()),
                      np.expand_dims(Xout_training, axis=1))
        y_predicted_training = np.matmul(Xin_training, w).flatten()
        y_predicted_test = np.matmul(Xin_test, w).flatten()

        train_MSE = np.sum(
            (y_predicted_training - Xout_training)**2) / len(Xout_training)
        test_MSE = np.sum((y_predicted_test - Xout_test)**2) / len(Xout_test)
        print("Polynomial regression degree {} training MSE {}, test MSE {}\n".format(
            degree, train_MSE, test_MSE))

        return train_MSE, test_MSE


    def train(self, k, hidden_dim, lr, bsize):
        self.args.hidden_dim = hidden_dim
        self.args.lr = lr
        self.args.batch_size = bsize
        # self.args.wd = 1e-7

        print("Hyper parameters: hidden_dim={}, lr={}, batch_size={}".format(
            hidden_dim, lr, bsize))

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
        step_number = nTrain // args.batch_size

        for ep_num in range(epoch_number):
            for step in range(1, step_number):
                optimizer.zero_grad()
                inds = np.random.randint(0, nTrain, args.batch_size)
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
                torch.save(model, MODEL_PATH + '/model_step_' + str(ep_num + 1))

        # print('\n\n')
        # print("MSE for training", compute_MSE(model, Xin_t, Xout_t))
        # print("MSE for testing", compute_MSE(model, Xin_test, Xout_test))

        return compute_MSE(model, Xin_v, Xout_v)


def compute_MSE(model, Xin, Xout):
    inBatch = torch.tensor(Xin).float()
    outBatch = torch.tensor(Xout).view(-1, 1).float()
    fhat = model(inBatch)
    f = outBatch
    SS_res = torch.sum((f - fhat)**2)
    MSE = SS_res / Xout.shape[0]
    return MSE.data.numpy()


def hyper_tuning(args):
    hidden_dim_list = [64, 128, 256]
    lr_list = [1e-1, 1e-2, 1e-3]
    bsize_list = [32, 64, 128]
    errors = np.empty([len(lr_list), len(bsize_list), len(hidden_dim_list)])
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
    trainer = Trainer(args, Xin, Xout)

    for i, hidden_dim in enumerate(hidden_dim_list):
        for j, lr in enumerate(lr_list):
            for k, bsize in enumerate(bsize_list):
                error = trainer.train(
                    k=1, lr=lr, bsize=bsize, hidden_dim=hidden_dim)
                errors[i, j, k] = error
    print(errors)
    np.save('plots/new_data/numpy/hyper/error.npy', np.asarray(errors))


def show_errors():
    data = np.load('plots/new_data/numpy/hyper/error.npy')
    np.set_printoptions(precision=6)
    print(data)


def polynomial_regression(args):
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
    trainer = Trainer(args, Xin, Xout)
    degrees = np.arange(1, 10, 1)
    train_MSE_tosave = []
    test_MSE_tosave = []
    for d in degrees:
        train_MSE, test_MSE =  trainer.polynomial_regression(d)
        train_MSE_tosave.append(train_MSE)
        test_MSE_tosave.append(test_MSE)
    np.save('plots/new_data/numpy/polynomial/train_MSE.npy', np.asarray(train_MSE_tosave))
    np.save('plots/new_data/numpy/polynomial/test_MSE.npy', np.asarray(test_MSE_tosave))


def training(args):
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
    trainer = Trainer(args, Xin, Xout)
    trainer.shuffle_data(K=10, test_ratio=0.)
    error = trainer.train(k=1, hidden_dim=256, lr=1e-2, bsize=32)


if __name__ == '__main__':
    args = arguments.args
    MODEL_PATH = args.checkpoints_path
    # hyper_tuning(args)
    # training(args)
    polynomial_regression(args)