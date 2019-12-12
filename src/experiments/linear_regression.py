import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import numpy as np
import argparse
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

class Trainer(object):
    def load_data(self, file_path):
        files = glob.glob(os.path.join(file_path, '*.npy'))

        X_vec = []
        y_vec = []

        for i, f in enumerate(files):
            if i%1000 == 0:
                print("processed ", i, " files")
            data = np.load(f).item()['data'][1]

            X = np.concatenate((data[0], data[1]))
            y = data[2]

            X_vec.append(X)
            y_vec.append(y)

        X_input = np.asarray(X_vec)
        y_input = np.asarray(y_vec)

        return X_input, y_input

def feature_map(I1_b, I2_b, J, n=1, m=1):
    d = 2
    N_samples = I1_b.shape[0]
    N_features = (n + 1)**2 + m
    N_features = 2
    Features = np.zeros((N_samples, N_features))
    count = 0

    count = 0
    for i in range(1, n + 1):
        Features[:, count] = (I1_b - d)**i 
        count = count + 1

    # for i in range(n + 1):
    #     for j in range(n + 1):
    #         Features[:, count] = (I1_b - d)**i * (I2_b - 1)**j
    #         count = count + 1

    for i in range(1, m + 1):
        Features[:, count] = (J - 1)**(2*i)
        count = count + 1

    # assert(count == 3)

    return Features

def get_features_general(Xin):
    d = 2
    N_samples = Xin.shape[0]
    F = Xin.copy()
    F = F[:, 0:4]
    F[:, 0] = F[:, 0] + 1
    F[:, 3] = F[:, 3] + 1
    F = F.reshape(N_samples, 2, 2)
    F_T = np.transpose(F, (0, 2, 1))
    C = np.matmul(F_T, F)
    C_square = np.matmul(C, C)
    J = np.linalg.det(F)
    I1 = np.trace(C, axis1=1, axis2=2)
    I2 = 0.5*(I1*I1 - np.trace(C_square, axis1=1, axis2=2))

    I1_b = J**(-2/d) * I1
    I2_b = J**(-4/d) * I2

    print("loader reports J", J)
    print("loader reports I1_b", I1_b)

    Features = feature_map(I1_b, I2_b, J)

    return Features



if __name__ == '__main__':
    args = arguments.args
    DATA_PATH = args.data_path_dummy_modified
    MODEL_PATH = args.checkpoints_path_dummy

    args.lr = 1e-2

    trainer = Trainer() 
    x_train, y_train = trainer.load_data(DATA_PATH)
    # Xin = get_features_C(Xin)
    # Xin = np.array([[1.09733702-1,  -0.04642577, 0.03862332, 0.90796399-1, 0, 0]])

    x_train = get_features_general(x_train)


    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_train), x_train)), np.transpose(x_train)), y_train)

    print(theta)

    exit()

    print("Total number of features is", x_train.shape[1])

    # ground_truth_output = Xin[:,0]*16.77 + Xin[:, 1]*833

    # print(ground_truth_output)
    # print(Xout)

    # exit()

    # mm_t = Net(Xin.shape[1])


    # Hyper-parameters
    input_size = 2
    output_size = 1
    num_epochs = 6000
    learning_rate = 10

    # # Toy dataset
    # x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
    #                     [9.779], [6.182], [7.59], [2.167], [7.042], 
    #                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
    #                     [3.366], [2.596], [2.53], [1.221], [2.827], 
    #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    # Linear regression model
    model = nn.Linear(input_size, output_size, bias=False)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # Train the model
    for epoch in range(num_epochs):
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(x_train).float()
        targets = torch.from_numpy(y_train).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            print(model.weight)








