import fenics as fa
import math
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from .. import arguments
from ..collector.driver_sobol import solver_fluctuation


def RVE(args, parameters_list):
    W = []
    for i, parameters in enumerate(parameters_list):
        print("\ni is {}".format(i))
        args.F_list_fixed = [[parameters[0], parameters[1]], [
            parameters[2], parameters[3]]]
        args.def_grad = np.array(parameters[0:4])
        args.void_shape = np.array(parameters[4:6])
        w, _ = solver_fluctuation(args)
        W.append(w)
    return np.asarray(W)


def from_H_to_C(parameters):
    parameters[0] = parameters[0] + 1
    parameters[3] = parameters[3] + 1
    parameters_new = np.zeros(4)
    parameters_new[0] = parameters[0] * \
        parameters[0] + parameters[2] * parameters[2]
    parameters_new[1] = parameters[0] * \
        parameters[1] + parameters[2] * parameters[3]
    parameters_new[2] = parameters[1] * \
        parameters[1] + parameters[3] * parameters[3]
    parameters_new[3] = parameters[4]
    return parameters_new


def NN(parameters_list):
    regr = pickle.load(open('saved_weights/model.sav', 'rb'))
    shift = np.load('saved_weights/mean.npy')
    W = []
    for i, parameters in enumerate(parameters_list):
        parameters_new = from_H_to_C(parameters)
        parameters_new = parameters_new - shift
        w = regr.predict(parameters_new.reshape(1, -1))
        W.append(w)
    return np.asarray(W)


def run_single(args, pore_shape, H22, H12):
    if pore_shape == 0:
        parameters_list_1 = [np.array([0, 0, 0, h22, 0, 0]) for h22 in H22]
        parameters_list_2 = [np.array([0, h12, 0, 0, 0, 0]) for h12 in H12]
    else:
        parameters_list_1 = [
            np.array([0, 0, 0, h22, -0.2, 0.2]) for h22 in H22]
        parameters_list_2 = [
            np.array([0, h12, 0, 0, -0.2, 0.2]) for h12 in H12]

    # W_RVE_H12 = RVE(args, parameters_list_2)
    # np.save('plots/new_data/numpy/custom_testing/W_RVE_H12_pore' +
    #         str(pore_shape) + '.npy', W_RVE_H12)
    # W_RVE_H22 = RVE(args, parameters_list_1)
    # np.save('plots/new_data/numpy/custom_testing/W_RVE_H22_pore' +
    #         str(pore_shape) + '.npy', W_RVE_H22)

    W_NN_H22 = NN(parameters_list_1)
    np.save('plots/new_data/numpy/custom_testing/W_NN_H22_pore' +
            str(pore_shape) + '.npy', W_NN_H22)
    W_NN_H12 = NN(parameters_list_2)
    np.save('plots/new_data/numpy/custom_testing/W_NN_H12_pore' +
            str(pore_shape) + '.npy', W_NN_H12)


def run(args):
    H22 = np.linspace(-0.2, 0.2, 41)
    np.save('plots/new_data/numpy/custom_testing/H22.npy', H22)
    H12 = np.linspace(-0.2, 0.2, 41)
    np.save('plots/new_data/numpy/custom_testing/H12.npy', H12)
    run_single(args, 0, H22, H12)
    run_single(args, 2, H22, H12)
    plot_results()


def plot_results():
    H22 = np.load('plots/new_data/numpy/custom_testing/H22.npy')
    H12 = np.load('plots/new_data/numpy/custom_testing/H12.npy')
    W_NN_H22_pore0 = np.load(
        'plots/new_data/numpy/custom_testing/W_NN_H22_pore0.npy')
    W_NN_H12_pore0 = np.load(
        'plots/new_data/numpy/custom_testing/W_NN_H12_pore0.npy')
    W_RVE_H22_pore0 = np.load(
        'plots/new_data/numpy/custom_testing/W_RVE_H22_pore0.npy')
    W_RVE_H12_pore0 = np.load(
        'plots/new_data/numpy/custom_testing/W_RVE_H12_pore0.npy')
    W_NN_H22_pore2 = np.load(
        'plots/new_data/numpy/custom_testing/W_NN_H22_pore2.npy')
    W_NN_H12_pore2 = np.load(
        'plots/new_data/numpy/custom_testing/W_NN_H12_pore2.npy')
    W_RVE_H22_pore2 = np.load(
        'plots/new_data/numpy/custom_testing/W_RVE_H22_pore2.npy')
    W_RVE_H12_pore2 = np.load(
        'plots/new_data/numpy/custom_testing/W_RVE_H12_pore2.npy')

    case_H22_NN = np.concatenate((W_NN_H22_pore0, W_NN_H22_pore2))
    case_H22_RVE = np.concatenate((W_RVE_H22_pore0, W_RVE_H22_pore2))
    MSE_test_case_H22 = mean_squared_error(case_H22_RVE, case_H22_NN)
    print("MSE_test_case_H22", MSE_test_case_H22)
    case_H12_NN = np.concatenate((W_NN_H12_pore0, W_NN_H12_pore2))
    case_H12_RVE = np.concatenate((W_RVE_H12_pore0, W_RVE_H12_pore2))
    MSE_test_case_H12 = mean_squared_error(case_H12_RVE, case_H12_NN)
    print("MSE_test_case_H12", MSE_test_case_H12)
    exit()

    plt.figure(0)
    plt.plot(H22, W_RVE_H22_pore0 -
             W_RVE_H22_pore0[20], linestyle='-', color='blue')
    plt.plot(H22, W_NN_H22_pore0 -
             W_NN_H22_pore0[20], linestyle='--', color='blue')
    plt.plot(H22, W_RVE_H22_pore2 -
             W_RVE_H22_pore2[20], linestyle='-', color='red')
    plt.plot(H22, W_NN_H22_pore2 -
             W_NN_H22_pore2[20], linestyle='--', color='red')
    plt.tick_params(labelsize=14)
    plt.figure(1)
    plt.plot(H12, W_RVE_H12_pore0 -
             W_RVE_H12_pore0[20], linestyle='-', color='blue')
    plt.plot(H12, W_NN_H12_pore0 -
             W_NN_H12_pore0[20], linestyle='--', color='blue')
    plt.plot(H12, W_RVE_H12_pore2 -
             W_RVE_H12_pore2[20], linestyle='-', color='red')
    plt.plot(H12, W_NN_H12_pore2 -
             W_NN_H12_pore2[20], linestyle='--', color='red')
    plt.tick_params(labelsize=14)
    plt.show()


if __name__ == '__main__':
    args = arguments.args
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 1000
    args.n_cells = 2
    args.fluctuation = True
    args.enable_fast_solve = True
    args.metamaterial_mesh_size = 15
    run(args)
