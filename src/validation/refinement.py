import numpy as np
import os
import fenics as fa
import sobol_seq
import sys
import time
import matplotlib.pyplot as plt
from ..collector.generator import Generator
from .. import arguments


def simulate(generator, res):
    generator.args.metamaterial_mesh_size = res
    parameters = [0, 0.1, 0, -0.1, 0, 0]
    generator.args.F_list_fixed = [
        [parameters[0], parameters[1]], [parameters[2], parameters[3]]]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])

    start = time.time()
    generator.generate_data_single(def_grad, void_shape)
    end = time.time()

    energy = generator.energy_density
    mesh_max = generator.pde.mesh.hmax()
    mesh_min = generator.pde.mesh.hmin()
    n_ele = generator.pde.mesh.num_cells()
    # magic number 0.5 is the area
    h = np.sqrt(0.5 / n_ele)
    print(h)
    return h, generator.pde, energy[-1], end - start


def ref(generator):
    ratio = 1.5
    resolution = [15 / ratio, 15, 15 * ratio, 15 * ratio**2, 60]
    pdes = []
    energy = []
    mesh_size = []
    error = []
    compute_time = []
    for res in resolution:
        h, pde, e, t = simulate(generator, res)
        pdes.append(pde)
        energy.append(e)
        mesh_size.append(h)
        compute_time.append(t)

    for i in range(len(resolution) - 1):
        err = get_error(pdes[-1], pdes[i])
        error.append(err)
    compute_time = compute_time[:-1]
    mesh_size = mesh_size[:-1]

    print(error)
    print(mesh_size)
    print(compute_time)

    np.save('plots/new_data/numpy/refinement/mesh_size.npy',
            np.asarray(mesh_size))
    np.save('plots/new_data/numpy/refinement/error.npy', np.asarray(error))
    np.save('plots/new_data/numpy/refinement/compute_time.npy',
            np.asarray(compute_time))


def plot_results():
    mesh_size = np.load('plots/new_data/numpy/refinement/mesh_size.npy')
    error = np.load('plots/new_data/numpy/refinement/error.npy')
    compute_time = np.load('plots/new_data/numpy/refinement/compute_time.npy')

    print("mesh_size", mesh_size)
    print("error", error)
    print("compute_time", compute_time)

    ms = np.linspace(mesh_size[0], mesh_size[-1], 100)
    first_order = ms
    first_order = first_order / first_order[0] * error[0]
    second_order = ms**2
    second_order = second_order / second_order[0] * error[0]

    # plt.figure(0)
    # plt.loglog(mesh_size, error, linestyle='--', marker='o', color='red')
    # plt.loglog(ms, first_order, linestyle='-', color='black')
    # plt.loglog(ms, second_order, linestyle='--', color='black')
    # plt.tick_params(labelsize=14)
    # plt.figure(1)
    # plt.plot(mesh_size, compute_time, linestyle='--', marker='o', color='red')
    # plt.tick_params(labelsize=14)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.loglog(mesh_size, error, linestyle='--', marker='o', color='red')
    ax1.loglog(ms, first_order, linestyle='-', color='black')
    ax1.loglog(ms, second_order, linestyle='--', color='black')
    ax1.tick_params(labelsize=14)

    ax2.plot(mesh_size, compute_time, linestyle='--', marker='o', color='blue')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.tick_params(labelsize=14)


def get_error(pde1, pde2):
    V1 = pde1.V
    V2 = pde2.V
    u1 = pde1.u
    u2 = fa.interpolate(pde2.u, V1)
    error = fa.Function(V1)
    error.vector().set_local(u1.vector()[:] - u2.vector()[:])
    return fa.norm(error)


def run(args):
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 1000
    generator.args.n_cells = 2
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 11)
    generator.enable_fast_solve = True
    run_simulation = False
    if run_simulation:
        ref(generator)
    plot_results()
    simulate(generator, 60)


if __name__ == '__main__':
    args = arguments.args
    np.set_printoptions(precision=6)
    run(args)
    plt.show()
