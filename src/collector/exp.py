"""Collects data for gripper case"""

import numpy as np
import os
from .generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt

if __name__ == '__main__':


    args = arguments.args
    PATH = args.data_path_integrated_regular
    generator = Generator(args)

    generator.args.relaxation_parameter = 0.05
    generator.args.max_newton_iter = 5000
    generator.enable_fast_solve = False
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = True
    generator.void_shape = np.array([-0., 0.]) 
    generator.anneal_factors = np.linspace(0, 1, 21)


    x_vals = np.linspace(-0.7, 0.7, 3)

    pr = []
    energy = []

    for x_val in x_vals:
        generator.args.F_list_fixed = [[0, x_val], [0., -0.125]]
        generator.def_grad = np.array([0, x_val, 0, -0.125])        
        generator._anealing_solver_fluctuation(False)
        energy_density = generator.energy_density
        energy.append(energy_density[-1])
        pr.append(x_val/0.125)
        print(generator.energy_density)
        print(generator.probe_all)        


    energy = np.asarray(energy)

    fig = plt.figure()
    plt.tick_params(labelsize=16)
    plt.xlabel("shear strain")
    plt.ylabel("strain energy density")
    plt.plot(x_vals, energy, '-*', label='energy')          
    # plt.legend(loc='lower right')
    plt.show()
    fig.savefig("foo.pdf", bbox_inches='tight')

    exit()


    try:
        generator._anealing_solver_fluctuation(False)
        print(generator.energy_density)
        print(generator.probe_all)
    except Exception as e:
        print(e)
        print(generator.energy_density)
        print(generator.probe_all)
    
    exit()

    generator.args.fluctuation = False
    generator.args.F_list_fixed = np.array([[0.1, 0], [0, -0.125]])


    x_vals = np.linspace(0, -0.05, 10)
    energy = []

    for x_val in x_vals:

        def_grad = np.array([x_val, 0, 0, -0.125])
        void_shape = np.array([0.1, -0.05]) 

        generator.def_grad = def_grad
        generator.void_shape = void_shape

        generator.anneal_factors = np.linspace(0, 1, 3)

        energy_density = generator._anealing_solver_fluctuation(False)
        energy.append(energy_density[-1])

        print(energy_density[-1])

    plt.figure()
    energy = np.asarray(energy)
    plt.plot(x_vals, energy, '-*', label='energy')          
    plt.legend(loc='upper right')
    plt.show()


    # print(energy_density, force)
