import numpy as np
import os
from ..collector.generator import Generator
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time


def run_and_save(size, pore_flag):
    print('\nsize is {} for pore{}'.format(size, pore_flag))
    start = time.time()
    generator = Generator(args)   
    generator.args.relaxation_parameter = 0.6
    generator.args.max_newton_iter = 2000
    generator.args.n_cells = size
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.padding = False
    # generator.anneal_factors = np.linspace(0.76, 1, 21)
    generator.def_grad = np.array([0, 0, 0, -0.1])
    generator.pore_flag = pore_flag
 
    if pore_flag == 0:
        # generator.enable_fast_solve = True
        generator.enable_fast_solve = False
        generator.args.relaxation_parameter = 0.2        
        generator.void_shape = np.array([-0., 0.])
        generator.anneal_factors = np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11)))
        # generator.anneal_factors = np.concatenate((np.linspace(0, 0.76, 1), np.linspace(0.76, 1., 51)))
        
        # To generate appendix figures
        # generator.anneal_factors = np.concatenate((np.linspace(0, 0.7, 8), np.linspace(0.7, 1., 41)))
    else:
        generator.enable_fast_solve = False
        generator.args.relaxation_parameter = 0.2
        generator.void_shape = np.array([-0.2, 0.2])
        generator.anneal_factors = np.linspace(0., 1, 21)

    energy_density, force, sols = generator._anealing_solver_disp(u_guess=None, return_all=True)
    end = time.time()
    t_total = end - start
    t_mesh = generator.pde.time_elapsed
  
    force = np.asarray([f[1][1] for f in force]) / (args.n_cells * args.L0)
    np.save('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore' + str(pore_flag) + '_size' + str(size)  + '.npy', force)

    print('time elapsed: total', t_total)
    print('time elapsed on mesh', t_mesh)
    return t_total, t_mesh


def simulate(pore_flag):
    sizes = [16]
    time_total = []
    time_mesh = []
    for size in sizes:
        t_total, t_mesh = run_and_save(size, pore_flag)
        time_total.append(t_total)
        time_mesh.append(t_mesh)

    print("t_total", t_total)
    print("time_mesh", time_mesh)

    # np.save('plots/new_data/numpy/size_effect/sizes_pore{}.npy'.format(pore_flag), np.asarray(sizes))
    np.save('plots/new_data/numpy/size_effect/time_pore{}_size{}.npy'.format(pore_flag, sizes[0]), np.asarray(time_total))
    # np.save('plots/new_data/numpy/size_effect/time_mesh.npy', np.asarray(time_mesh))


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

    NN_force_com_pore0 = np.load('plots/new_data/numpy/force/NN_force_com_pore0.npy') 
    plt.plot(np.linspace(0, -0.1, len(NN_force_com_pore0)), (NN_force_com_pore0 - NN_force_com_pore0[0]), '--', color='blue')

    sizes = np.load('plots/new_data/numpy/size_effect/sizes.npy')
    sizes = [16]
    colors = ['blue', 'orange', 'red', 'purple']
    for i, sz in enumerate(sizes):
        DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/' + 'DNS_force_com_pore0_size' + str(sz)  + '.npy')
        plt.plot(-0.1*np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11))),
            (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='-', marker='o', color=colors[i])

def plot_custom_force():
    DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/DNS_force_com_pore0_size4.npy') 
    plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='blue')

    DNS_force_com_pore0 = np.load('plots/new_data/numpy/size_effect/DNS_force_com_pore0_size4_dr.npy') 
    plt.plot(np.linspace(0, -0.1, len(DNS_force_com_pore0)), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), linestyle='--', marker='o', color='red')


def report_time():
    time_pore0_sizes16 = np.load('plots/new_data/numpy/size_effect/time_pore0_size16.npy')
    time_pore2_sizes16 = np.load('plots/new_data/numpy/size_effect/time_pore2_size16.npy')
    time_pore0_sizes20 = np.load('plots/new_data/numpy/size_effect/time_pore0_size20.npy')
    time_pore2_sizes20 = np.load('plots/new_data/numpy/size_effect/time_pore2_size20.npy')
    time_pore0_sizes32 = np.load('plots/new_data/numpy/size_effect/time_pore0_size32.npy')
    time_pore2_sizes32 = np.load('plots/new_data/numpy/size_effect/time_pore2_size32.npy')

    print(time_pore0_sizes16)
    print(time_pore2_sizes16)
    print(time_pore0_sizes20)
    print(time_pore2_sizes20)
    print(time_pore0_sizes32)
    print(time_pore2_sizes32)

    print(time_pore0_sizes16/16)
    print(time_pore2_sizes16/20)
    print(time_pore0_sizes20/16)
    print(time_pore2_sizes20/20)
    print(time_pore0_sizes32/16)
    print(time_pore2_sizes32/20)


def run():
    # try:
    #     simulate(0)
    # except Exception as e:
    #     print("hehe, fail")
    # simulate(0)    
    # simulate(2)  
 
    report_time()
    # plot_results_time()
    # plot_results_force()
    # plt.show()


if __name__ == '__main__':
    args = arguments.args
    fa.set_log_level(30)
    run()
