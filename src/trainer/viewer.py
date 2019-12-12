import numpy as np
import torch
from .loader import Net
from .. import arguments
from .loader import Trainer
import matplotlib.pyplot as plt
import glob
import os

if __name__ == '__main__':
    args = arguments.args
    model_path = args.checkpoints_path_shear + '/model_step_' + str(1000*29)
    # model_path = 'saved_checkpoints_tmp/shear'
    network =  torch.load(model_path)

    DATA_PATH = args.data_path_integrated_regular
    trainer = Trainer() 
    Xin, Xout = trainer.load_data(DATA_PATH)


    # for i, xin in enumerate(Xin):
    #     if np.absolute(xin[1]) < 1e-1 and np.absolute(xin[2]) < 1e-1 and \
    #         not (np.sum(np.absolute(xin)) == 0) and \
    #         xin[3] < 0:
    #         print(xin)
    #         print(Xout[i])


    # exit()

    # np.savetxt("Xin.csv", Xin, delimiter=",")
    # np.savetxt("Xout.csv", Xout, delimiter=",")


    # files = glob.glob(os.path.join(DATA_PATH, '121000_anneal*.npy'))
    # X_vec = []
    # y_vec = []
    # for i, f in enumerate(files):
    #     data = np.load(f).item()['data'][1]
    #     X = np.concatenate((data[0], data[1]))
    #     y = data[2]
    #     X_vec.append(X)
    #     y_vec.append(y)
    # X_input = np.asarray(X_vec)
    # y_input = np.asarray(y_vec)

    # print(X_input)
    # print(y_input)

    # exit()

    x_values = np.linspace(-0.2, 0.2, 31)
    y_value = []
    for x_value in x_values:
        F11 = 1
        F12 = x_value
        F21 = 0
        F22 = 1
        # test_input = torch.tensor([1, x_value, x_value**2 + 0.875**2, -0.0, 0.0])
        test_input = torch.tensor([F11**2 + F21**2, F12*F11 + F21*F22, F12**2 + F22**2, -0.2, 0.2])        
        test_output = network(test_input)
        print(test_output)
        y_value.append(test_output)

    y_value = np.asarray(y_value)
    plt.figure()
    plt.plot(x_values, y_value, '*-')
    plt.show()

