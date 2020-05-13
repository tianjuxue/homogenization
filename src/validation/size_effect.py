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
    # generator.args.relaxation_parameter = 0.2    
    generator.args.relaxation_parameter = 0.2
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = False
    generator.args.n_cells = size
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.padding = False
    # generator.anneal_factors = np.linspace(0.76, 1, 21)
    generator.anneal_factors = np.linspace(0., 1, 21)
    generator.def_grad = np.array([0, 0, 0, -0.1])
    generator.void_shape = np.array([-0.2, 0.2])
    pore_type = 0 if np.sum(np.absolute(generator.void_shape)) < 1e-3  else 2 

    # mesh_name = 'plots/new_data/mesh/' + 'DNS' + '_size' + str(size) + '_pore' + str(2) + '.xml'
    # if os.path.isfile(mesh_name):
    #     mesh = fa.Mesh(mesh_name)
    #     V = fa.VectorFunctionSpace(mesh, 'P', 1)
    #     disp_sol = fa.Function(V, 'plots/new_data/sol/intermediate/size' + str(size) + '_disp_0.07600.xml')
    #     print("load solution")
    # else:
    #     disp_sol = None

    energy_density, force, sols = generator._anealing_solver_disp(u_guess=None, return_all=True)
    end = time.time()
    t_total = end - start
    t_mesh = generator.pde.time_elapsed
  
    force = np.asarray([f[1][1] for f in force]) / (args.n_cells * args.L0)
    np.save('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore' + str(pore_type) + '_size' + str(size)  + '.npy', force)

    print('time elapsed: total', t_total)
    print('time elapsed on mesh', t_mesh)
    return t_total, t_mesh


def simulate():
    sizes = [32]
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
    ax.tick_params(axis='x', which=u'both',length=0)
    ax.tick_params(labelsize=16)    
    ax.axes.xaxis.set_visible(False)


def plot_results_force():
    fig = plt.figure(0)
    plt.tick_params(labelsize=14)

    NN_force_com_pore0 = np.load('plots/new_data/numpy/force/NN_force_com_pore2.npy') 
    plt.plot(np.linspace(0, -0.1, len(NN_force_com_pore0)), (NN_force_com_pore0 - NN_force_com_pore0[0]), '--', color='blue')

    sizes = np.load('plots/new_data/numpy/size_effect/sizes.npy')
    sizes = [32]
    colors = ['orange', 'blue', 'red', 'purple']
    for i, sz in enumerate(sizes):
        DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore2_size' + str(sz)  + '.npy')
        # plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color=colors[i])
        plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='-', marker='o', color=colors[i])

def plot_custom_force():
    DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/DNS_force_com_pore0_size4.npy') 
    plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='blue')

    DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/DNS_force_com_pore0_size4_dr.npy') 
    plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='red')


def run():
    # simulate()
    # plot_results_time()
    plot_results_force()
    plt.show()


if __name__ == '__main__':
    args = arguments.args
    fa.set_log_level(20)
    run()
