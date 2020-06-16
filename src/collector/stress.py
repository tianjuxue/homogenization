import fenics as fa
import math
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
from scipy.interpolate import UnivariateSpline
from .. import arguments
from .driver_sobol import solver_fluctuation
from ..validation.custom_testing import NN, from_H_to_C
from ..trainer.loader_sklearn import PolyReg


def simulate_RVE():
    W = np.zeros((n1, n2))
    P = np.zeros((n1, n2))
    for i, h11 in enumerate(H11):
        for j, h22 in enumerate(H22):
            print("\ni is {} and j is {}".format(i, j))
            parameters = np.array([h11, 0, 0, h22, 0, 0])
            args.F_list_fixed = [[parameters[0], parameters[1]], [parameters[2], parameters[3]]]
            args.def_grad = np.array(parameters[0:4])
            args.void_shape = np.array(parameters[4:6])
            w, s = solver_fluctuation(args)
            W[i, j] = w
            P[i, j] = s
    np.save('sheng_mao/data/W{}x{}.npy'.format(n1, n2), W)
    np.save('sheng_mao/data/P{}x{}.npy'.format(n1, n2), P)


def stress_path(W, P=None):
    n2 = W.shape[1]
    H11_min = []
    W_min = []
    P_star = []
    for i in range(n2):
        if P is not None:
            P_star.append(P[W[:, i].argmin(), i])
        H11_min.append(H11[W[:, i].argmin()])
        W_min.append(W[:, i].min())
    W_min = np.asarray(W_min)
    P_star = np.asarray(P_star)

    spl = UnivariateSpline(H22[::-1], W_min[::-1], k=4, s=0)
    yp = spl.derivative()
    P_der = yp(H22)
    return P_der, P_star 


def run(args):

    # RVE related
    print("Start RVE...")
    W_RVE = np.load('sheng_mao/data/W{}x{}.npy'.format(n1, n2))
    P_RVE_integral = np.load('sheng_mao/data/P{}x{}.npy'.format(n1, n2))
    P_RVE, _ = stress_path(W_RVE, P_RVE_integral)
    print("End RVE.")

    # NN related
    print("Start NN...")
    W_NN = []
    for h11 in H11:
        parameters_list = [np.array([h11, 0, 0, h22, 0, 0]) for h22 in H22]
        W_NN.append(NN(parameters_list))
    W_NN = np.asarray(W_NN)
    P_NN, _ = stress_path(W_NN)
    print("End NN.")

    # Polynomial related
    # TODO: Bad implementation
    print("Start poly...")
    shift = np.load('saved_weights/mean.npy')
    poly_regr = pickle.load(open('saved_weights/poly.sav', 'rb'))
    W_Poly = []
    for h11 in H11:
        W_poly_sub = []
        for h22 in H22:
            parameters = np.array([h11, 0, 0, h22, 0, 0])
            parameters = from_H_to_C(parameters) - shift
            X_poly = PolyReg.poly_features_flexible(10, parameters.reshape(1, -1))
            y_pred_test = poly_regr.predict(X_poly)
            W_poly_sub.append(y_pred_test)
        W_Poly.append(W_poly_sub)
    W_Poly = np.asarray(W_Poly)
    P_Poly, _ = stress_path(W_Poly)
    print("End poly.")

    plt.figure()
    plt.plot(H22, P_RVE - P_RVE[0], linestyle='-', marker='o', color='blue')     
    plt.plot(H22, P_NN - P_NN[0], linestyle='-', marker='o', color='red')
    plt.plot(H22, P_Poly - P_Poly[0], linestyle='-', marker='o', color='orange')
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
    n1 = 41
    n2 = 21
    H11 = np.linspace(-0.1, 0.1, n1)
    H22 = np.linspace(0, -0.125, n2)
    run(args)