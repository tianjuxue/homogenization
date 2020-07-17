import numpy as np
import os
from ..collector.generator import Generator
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
import time


def run_and_save(disp, pore_flag, name):
    print("\ndisp={}, pore_flag={}, name={}".format(disp, pore_flag, name))
    start = time.time()
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.6
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = False
    generator.args.n_cells = 16
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.gradient = False if pore_flag < 3 else True
    # Initial submission
    generator.anneal_factors = np.linspace(0, 1, 21)
    generator.def_grad = np.array([0, 0, 0, disp])
    generator.pore_flag = pore_flag

    if pore_flag == 0:
        generator.void_shape = np.array([-0., 0.])
        if disp < 0:
            generator.enable_fast_solve = True
            # For force and time to be correct
            # generator.anneal_factors = np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11)))
            # For graphics to be correct
            generator.anneal_factors = np.concatenate((np.linspace(0, 0.76, 7), np.linspace(0.76, 1., 51)))
    elif pore_flag == 1:
        generator.void_shape = np.array([-0.1, 0.1])
    elif pore_flag == 2:
        # For graphics to be correct
        generator.enable_fast_solve = False
        generator.args.relaxation_parameter = 0.2
        generator.anneal_factors = np.linspace(0, 1, 21)

        # For force and time to be correct
        # generator.enable_fast_solve = False
        # generator.args.relaxation_parameter = 0.6
        # generator.anneal_factors = np.linspace(0, 1, 21)

        generator.void_shape = np.array([-0.2, 0.2])
    else:
        # Shouldn't matter
        generator.void_shape = np.array([-0., 0.])
        generator.anneal_factors = np.linspace(0, 1, 11)

    energy_density, force, _ = generator._anealing_solver_disp(return_all=True)

    force = np.asarray([f[1][1] for f in force]) / (args.n_cells * args.L0)
    energy = np.asarray(energy_density)

    end = time.time()
    time_elapsed = end - start

    deform_info = 'com' if disp < 0 else 'ten'
    # np.save('plots/new_data/numpy/energy/' + name + '_energy_' + deform_info +
    #         '_pore' + str(pore_flag) + '.npy', energy)
    # np.save('plots/new_data/numpy/force/' + name + '_force_' + deform_info +
    #         '_pore' + str(pore_flag) + '.npy', force)
    # np.save('plots/new_data/numpy/time/' + name + '_time_' + deform_info +
    #         '_pore' + str(pore_flag) + '.npy', time_elapsed)

    # fa.File('plots/new_data/sol/post_processing/input/' + name + '_mesh_' +
    #      deform_info + '_pore' + str(pore_flag) + '.xml') << generator.pde.mesh
    # fa.File('plots/new_data/sol/post_processing/input/' + name + '_sol_' +
    #      deform_info + '_pore' + str(pore_flag) + '.xml') << generator.pde.u

    print('energy_list', energy)
    print('force_list', force)
    print('time_elapsed', time_elapsed)
    print('\n')


def run():
    # run_and_save(disp=-0.1, pore_flag=2, name='DNS')
    # run_and_save(disp=0.1, pore_flag=0, name='DNS')
    # run_and_save(disp=0.1, pore_flag=2, name='DNS')
    run_and_save(disp=-0.1, pore_flag=0, name='DNS')


if __name__ == '__main__':
    args = arguments.args
    # fa.set_log_level(20)
    run()
