import numpy as np
import os
from ..collector.generator import Generator
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def run_and_save():
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = True
    generator.args.n_cells = 4
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.gradient = False
    generator.anneal_factors = np.linspace(0, 1, 11)
    generator.def_grad = np.array([0, 0, 0, -0.1])
    generator.void_shape = np.array([-0., 0.])
    generator.args.padding = True
    generator._anealing_solver_disp(return_all=True)



if __name__ == '__main__':
    args = arguments.args
    run_and_save()