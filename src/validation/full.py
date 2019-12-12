"""Collects data for gripper case"""

import numpy as np
import os
from ..collector.generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt

if __name__ == '__main__':

    if True:
        args = arguments.args
        PATH = args.data_path_integrated_regular
        generator = Generator(args)

        generator.args.relaxation_parameter = 0.1
        generator.args.max_newton_iter = 2000
        generator.enable_fast_solve = False
        generator.args.n_cells = 8
        generator.args.metamaterial_mesh_size = 15
        generator.args.fluctuation = False
        generator.args.F_list_fixed = [[-0., -0.], [0., -0.125]]
        generator.args.gradient = False
        generator.anneal_factors = np.linspace(0, 1, 11)

        generator.def_grad = np.array([0, 0, 0, 0.2])
        generator.void_shape = np.array([-0.2, 0.2]) 
      
        energy_density, force = generator._anealing_solver_disp(True)
        print(energy_density)
        force = np.asarray([f[1][1] for f in force])
        print(force)

        # np.save('force.npy', force)

        exit()


    path_prefix = 'plots/data/tension/'

    full_force_p1 = np.load(path_prefix + 'full_force_p1.npy')
    full_force_p2 = np.load(path_prefix + 'full_force_p2.npy')
    full_force_gradient = np.load(path_prefix + 'full_force_gradient.npy')

    homo_force_p1 = np.load(path_prefix + 'homo_force_p1.npy')
    homo_force_p2 = np.load(path_prefix + 'homo_force_p2.npy')
    homo_force_gradient = np.load(path_prefix + 'homo_force_gradient.npy')

    x_vals = np.linspace(0, 0.2, 11)
    
    fig = plt.figure()
    plt.tick_params(labelsize=14)

    plt.xlabel('strain')
    plt.ylabel('stress')
    plt.plot(x_vals, (full_force_p1 - full_force_p1[0]) / 4, '-', color='blue', label='DNS ' + r'$\xi_a$')    
    plt.plot(x_vals, (homo_force_p1 - homo_force_p1[0]) / 4, ':', color='blue', label='NN ' + r'$\xi_a$')  
    plt.plot(x_vals, (full_force_p2 - full_force_p2[0]) / 4, '-', color='red', label='DNS ' + r'$\xi_b$')    
    plt.plot(x_vals, (homo_force_p2 - homo_force_p2[0]) / 4, ':', color='red', label='NN ' + r'$\xi_b$')  
    plt.plot(x_vals, (full_force_gradient - full_force_gradient[0]) / 4, '-', color='orange', label='DNS gradient')    
    plt.plot(x_vals, (homo_force_gradient - homo_force_gradient[0]) / 4, ':', color='orange', label='NN gradient')  

    plt.legend(loc='upper left')
    plt.show()
    fig.savefig("gradient_validation.pdf", bbox_inches='tight')
