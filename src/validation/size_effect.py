import numpy as np
import os
from ..collector.generator import Generator
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time


def run_and_save(size):
    print('\nsize is', size)
    start = time.time()
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = False
    generator.args.n_cells = size
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.F_list_fixed = [[-0., -0.], [0., -0.1]]
    generator.args.gradient = False
    generator.anneal_factors = np.linspace(0, 1, 11)
    generator.def_grad = np.array([0, 0, 0, -0.1])
    generator.void_shape = np.array([-0., 0.])
    energy_density, force = generator._anealing_solver_disp(return_force=True)
    end = time.time()
    t_total = end - start
    t_mesh = generator.pde.time_elapsed
  
    force = np.asarray([f[1][1] for f in force]) / (args.n_cells * args.L0)
    np.save('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore0_size' + str(size)  + '.npy', force)

    print('time elapsed: total', t_total)
    print('time elapsed on mesh', t_mesh)
    return t_total, t_mesh


def simulate():
    sizes = [2, 4, 8, 16]
    time_total = []
    time_mesh = []
    for size in sizes:
        t_total, t_mesh = run_and_save(size)
        time_total.append(t_total)
        time_mesh.append(t_mesh)

    print("t_total", t_total)
    print("time_mesh", time_mesh)

    np.save('plots/new_data/numpy/size_effect/sizes.npy', np.asarray(sizes))
    np.save('plots/new_data/numpy/size_effect/time_total.npy', np.asarray(time_total))
    np.save('plots/new_data/numpy/size_effect/time_mesh.npy', np.asarray(time_mesh))


def plot_results_time():
    sizes = np.load('plots/new_data/numpy/size_effect/sizes.npy')
    time_total = np.load('plots/new_data/numpy/size_effect/time_total.npy')
    time_mesh = np.load('plots/new_data/numpy/size_effect/time_mesh.npy')
    fig, ax = plt.subplots()
    ax.plot(sizes, time_total, linestyle='--', marker='o', color='red')
    ax.plot(sizes, time_mesh, linestyle='--', marker='o', color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # labels = ['A', 'B', 'C', 'D']
    # ax.set_xticks(sizes)
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which=u'both',length=0)
    ax.tick_params(labelsize=16)    
    ax.axes.xaxis.set_visible(False)


def plot_results_force():
    NN_force_com_pore0 = np.load('plots/new_data/numpy/force/NN_force_com_pore0.npy') 
    strain_com = np.linspace(0, -0.1, len(NN_force_com_pore0))

    fig = plt.figure(0)
    plt.tick_params(labelsize=14)
    plt.plot(strain_com, (NN_force_com_pore0 - NN_force_com_pore0[0]), '--', color='blue', label='NN ' + r'$\xi_a$')

    sizes = np.load('plots/new_data/numpy/size_effect/sizes.npy')
    colors = ['orange', 'red', 'green', 'purple']
    for i, sz in enumerate(sizes):
        DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore0_size' + str(sz)  + '.npy')
        plt.plot(strain_com, (DNS_force_com_pore0 - DNS_force_com_pore0[0]), '-', color=colors[i], label='NN ' + r'$\xi_a$')


def run():
    run_simulation = False
    if run_simulation:
        simulate()
    # plot_results_time()
    plot_results_force()
    plt.show()

if __name__ == '__main__':
    args = arguments.args
    run()
