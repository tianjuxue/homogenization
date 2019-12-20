import glob
import math
import numpy as np
import argparse
import sys
from collections import namedtuple
import matplotlib.pyplot as plt
from copy import deepcopy
import pdb
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as torchF
from torch.autograd import Variable
import logging
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
from .. import arguments
from sklearn.linear_model import Ridge, Lasso


class Trainer(object):
    def load_data(self, file_path):
        files = glob.glob(os.path.join(file_path, '*.npy'))
        # files = glob.glob(os.path.join(file_path, '*anneal10.npy'))
        X_vec = []
        y_vec = []
        y_extra_vec = []

        for i, f in enumerate(files):
            if i%1000 == 0:
                print("processed ", i, " files")
            data = np.load(f).item()['data'][1]

            X = np.concatenate((data[0], data[1]))
            y = data[2]
            # y_extra = data[3]

            X_vec.append(X)
            y_vec.append(y)
            # y_extra_vec.append(y_extra)

        X_input = np.asarray(X_vec)
        y_input = np.asarray(y_vec)
        # y_extra_input = np.asarray(y_extra_vec)

        return X_input, y_input


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim , 1)  

    def forward(self, x):
        # x = torchF.relu(self.fc1(x))
        x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
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
    X_new[:, 0] = Xin[:, 0]*Xin[:, 0] + Xin[:, 2]*Xin[:, 2]
    X_new[:, 1] = Xin[:, 0]*Xin[:, 1] + Xin[:, 2]*Xin[:, 3]
    X_new[:, 2] = Xin[:, 1]*Xin[:, 1] + Xin[:, 3]*Xin[:, 3]
    X_new = X_new[:, [0, 1, 2, 4, 5]]
    return X_new

#TODO(Tianju): Modify global variables to be local
def plot_train_test_curve():
    plt.figure()
    plt.plot(epoch, mse_train, '*-', color='b', label='training')
    plt.plot(epoch, mse_test, '*-', color='r', label='test')
    plt.xlabel('step')
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.show()


def compute_R2(model, Xin, Xout):
    inBatch  = torch.tensor(Xin, requires_grad=True).float()
    outBatch = torch.tensor(Xout).view(-1, 1).float()
    fhat = model(inBatch)
    fmean = torch.mean(outBatch)
    f = outBatch
    SS_res = torch.sum((f - fhat)**2)
    SS_reg = torch.sum((f - fmean)**2)
    R2 = 1 - SS_res/SS_reg
    return R2.data.numpy()


def compute_MSE(model, Xin, Xout):
    inBatch  = torch.tensor(Xin, requires_grad=True).float()
    outBatch = torch.tensor(Xout).view(-1, 1).float()
    fhat = model(inBatch)
    f = outBatch
    SS_res = torch.sum((f - fhat)**2)
    MSE = SS_res/Xout.shape[0]
    return MSE.data.numpy()


def run_trainer(args, Xin, Xout, lr, bsize, hidden_dim):
    args.lr = lr
    args.batch_ize = bsize
    args.hidden_dim = hidden_dim

    nSamps = len(Xin)
    nTrain = int((1 - args.val_fraction) * nSamps)
    inds = np.random.permutation(nSamps)    
    indsTrain = inds[:nTrain]
    indsTest  = inds[nTrain:]

    Xt_in = Xin[indsTest]
    Xt_out = Xout[indsTest]

    Xin = Xin[indsTrain]
    Xout = Xout[indsTrain]

    nTest = len(Xt_in)
    fAv = fAvTest = f_dev = f_dev_test = 0. 

    model = Linear(args)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    
    epoch_number = 1000
    log_interval = 100
    step_number = nTrain//args.batch_ize

    epoch = []
    mse_train = []
    mse_test = []
    mpe_train = []
    mpe_test = []

    for ep_num in range(epoch_number):
        for step in range(1, step_number):
            optimizer.zero_grad()

            # Testing
            inds_test = np.random.randint(0, nTest, args.batch_ize)
            inBatch_test  = torch.tensor(Xt_in[inds_test], requires_grad=True).float()
            outBatch_test = torch.tensor(Xt_out[inds_test]).view(-1, 1).float()
            fhat_test = model(inBatch_test)
            f_test = outBatch_test
            f_loss_test = torch.sum((f_test - fhat_test)**2)
            fAvTest += float(f_loss_test.data)
            f_dev_test += float(torch.sum(abs(f_test - fhat_test)/(f_test + 1e-5)).data) # mean percentage error (MPE)

            # Training
            inds = np.random.randint(0, nTrain, args.batch_ize)
            inBatch  = torch.tensor(Xin[inds], requires_grad=True).float()
            outBatch = torch.tensor(Xout[inds]).view(-1, 1).float()
            fhat = model(inBatch)
            f = outBatch
            f_loss = torch.sum((f - fhat)**2)
            fAv += float(f_loss.data)
            f_dev += float(torch.sum(abs(f - fhat)/(f + 1e-5)).data)

            total_loss = f_loss
            total_loss.backward()
            optimizer.step()

        if (ep_num + 1)%log_interval == 0:
            print('\nepoch:', ep_num + 1)           
            epoch.append(ep_num)
            train_mse = compute_MSE(model, Xin, Xout)
            test_mse = compute_MSE(model, Xt_in, Xt_out)            
            print("MSE for training", train_mse, "MSE for testing", test_mse)
            mse_train.append(train_mse)
            mse_test.append(test_mse) 
            # torch.save(model, MODEL_PATH + '/model_step_'+str(ep_num))
            fAv = fAvTest = f_dev = f_dev_test = 0.

    print('\n\n')
    print("R2 for training", compute_R2(model, Xin, Xout))
    print("R2 for testing", compute_R2(model, Xt_in, Xt_out))   

    print("MSE for training", compute_MSE(model, Xin, Xout))
    print("MSE for testing", compute_MSE(model, Xt_in, Xt_out)) 

    return compute_MSE(model, Xt_in, Xt_out)


if __name__ == '__main__':
    args = arguments.args
    DATA_PATH = args.data_path_integrated_regular
    MODEL_PATH = args.checkpoints_path_shear

    trainer = Trainer() 

    DATA_PATH_shear1 = 'saved_data_integrated_regular_periodic_shear1'
    DATA_PATH_shear2 = 'saved_data_integrated_regular_periodic_shear2'  
    DATA_PATH_normal = 'saved_data_integrated_regular_periodic_normal'

    Xin_shear1, Xout_shear1 = trainer.load_data(DATA_PATH_shear1)
    Xin_shear2, Xout_shear2 = trainer.load_data(DATA_PATH_shear2)    
    Xin_normal, Xout_normal = trainer.load_data(DATA_PATH_normal)
 
    Xin = np.concatenate((Xin_shear2, Xin_normal))
    Xout = np.concatenate((Xout_shear2, Xout_normal))

    print(Xin.shape)
    print(Xout.shape)

    # Xout = get_labels(Xin, Xout)
    Xin = get_features_C(Xin)
    args.input_dim = Xin.shape[1]

    print("Total number of features is", Xin.shape[1])

    error = run_trainer(args, Xin, Xout, 1e-2, 32, 256)
    exit()

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