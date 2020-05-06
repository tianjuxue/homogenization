import numpy as np
import os
from ..collector.generator import Generator
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
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
    generator._anealing_solver_disp()
    end = time.time()
    t_total = end - start
    t_mesh = generator.pde.time_elapsed
    print('time elapsed: total', t_total)
    print('time elapsed on mesh', t_mesh)
    return t_total, t_mesh


def simulate():
    sizes = [8]
    time_total = []
    time_mesh = []
    for size in sizes:
        t_total, t_mesh = run_and_save(size)
        time_total.append(t_total)
        time_mesh.append(t_mesh)

    print("t_total", t_total)
    print("time_mesh", time_mesh)

    # np.save('plots/new_data/numpy/size_effect/sizes.npy', np.asarray(sizes))
    # np.save('plots/new_data/numpy/size_effect/time_total.npy', np.asarray(time_total))
    # np.save('plots/new_data/numpy/size_effect/time_mesh.npy', np.asarray(time_mesh))

def plot_results():
    sizes = np.load('plots/new_data/numpy/size_effect/sizes.npy')
    time_total = np.load('plots/new_data/numpy/size_effect/time_total.npy')
    time_mesh = np.load('plots/new_data/numpy/size_effect/time_mesh.npy')
    plt.figure(0)
    plt.plot(sizes, time_total, linestyle='--', marker='o', color='red')
    plt.plot(sizes, time_mesh, linestyle='--', marker='o', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(labelsize=14)
    plt.show()


def run():
    run_simulation = True
    if run_simulation:
        simulate()
    plot_results()


if __name__ == '__main__':
    args = arguments.args
    run()
