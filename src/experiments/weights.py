import fenics as fa
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ..trainer.loader import Net
from .. import arguments
from ..collector.generator import Generator, GeneratorDummy
from ..trainer.loader import get_features_general, Trainer
from sklearn.linear_model import Ridge, Lasso

def my_sign(x):
    return -1 if x < 0 else 1

def my_relu(x):
    return x/2*(my_sign(x) + 1)

def layer(x, weights, bias):
    weights = weights.data.numpy()
    bias = bias.data.numpy()
    input_size = weights.shape[1]
    output_size = weights.shape[0]
    output = []
    for i in range(output_size):
        tmp = 0
        for j in range(input_size):
            tmp = tmp + weights[i][j]*x[j]
        tmp = tmp + bias[i]
        tmp = my_relu(tmp)
        output.append(tmp)
    return output

def manual_nn(x):
    x = layer(x, network.fc1.weight, network.fc1.bias)
    x = layer(x, network.fc2.weight, network.fc2.bias)
    x = layer(x, network.fc3.weight, network.fc3.bias)
    return x

def get_energy_val(manual_input):
    torch_input = torch.tensor(manual_input)
    numpy_input = np.array(manual_input[0:4])
    # torch_output = get_LR_output(manual_input)
    torch_output = network(torch_input)
    torch_output = torch_output.data.numpy()
    generator_output = generator._compute_energy(numpy_input)
    return torch_output, generator_output

def get_LR_output(manual_input):
    Xin = np.expand_dims(np.array(manual_input), axis=0)
    Xin = get_features_general(Xin)
    Xout = clf.predict(Xin)
    return np.asscalar(Xout)


if __name__ == '__main__':
    args = arguments.args
    generator = GeneratorDummy(args)
    model_path = args.checkpoints_path_shear + '/model_step_' + str(100*99)
    network =  torch.load(model_path)

    # trainer = Trainer() 
    # Xin, Xout = trainer.load_data(args.data_path_shear)
    # Xin = get_features_general(Xin)
    # print("Total number of features is", Xin.shape[1])

    # batchSize = 128
    # nSamps = len(Xin)
    # nTrain = int((1 - args.val_fraction) * nSamps)
    # inds = np.random.permutation(nSamps)    
    # indsTrain = inds[:nTrain]
    # indsTest  = inds[nTrain:]

    # Xt_in = Xin[indsTest]
    # Xt_out = Xout[indsTest]

    # Xin = Xin[indsTrain]
    # Xout = Xout[indsTrain]

    # clf = Ridge(alpha=1e-5)   
    # clf.fit(Xt_in, Xt_out)
    # Xt_out_hat = clf.predict(Xt_in)
    # print(clf.score(Xt_in, Xt_out))
    # print(clf.coef_)
   
    net_output = []
    true_output = [] 
    
    x_val = np.linspace(-1, 1, 21)

    for x in x_val:
        manual_input = [0, x, x, 0, -0.2, 0.2]
        torch_output, generator_output = get_energy_val(manual_input)
        net_output.append(torch_output)
        true_output.append(generator_output)

    net_output = np.asarray(net_output)
    true_output = np.asarray(true_output)

    plt.figure()
    plt.plot(x_val, net_output, '*-', label='net')
    # plt.plot(x_val, true_output, '*-', label='true')
    plt.legend(loc='upper right')
    plt.show()

    # def_grad_all, void_shape_all, predicted_energy_all = generator.generate_data()


