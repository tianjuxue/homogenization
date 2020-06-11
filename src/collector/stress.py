import fenics as fa
import math
import numpy as np
import matplotlib.pyplot as plt
import os.path
from ..pde.static import Metamaterial
from .. import arguments
from .driver_sobol import solver_fluctuation


def run():
    n1 = 21
    n2 = 11
    W = np.zeros((n1, n2))
    P = np.zeros((n1, n2))
    H11 = np.linspace(-0.1, 0.1, n1)
    H22 = np.linspace(0, -0.125, n2)
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
    np.save('sheng_mao/data/W21x11.npy', W)
    np.save('sheng_mao/data/P21x11.npy', P)


if __name__ == '__main__':
    args = arguments.args
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 1000
    args.n_cells = 2
    args.fluctuation = True
    args.enable_fast_solve = True
    args.metamaterial_mesh_size = 15
    run()