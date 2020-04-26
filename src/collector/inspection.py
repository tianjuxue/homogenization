import numpy as np
import os
from .generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt


def compute_shear(generator):
    generator.void_shape = np.array([-0.2, 0.2])
    x_vals = np.linspace(-0.7, 0.7, 21)
    energy = []
    for x_val in x_vals:
        generator.args.F_list_fixed = [[0, x_val], [0., -0.125]]
        generator.def_grad = np.array([0, x_val, 0, -0.125])
        generator._anealing_solver_fluctuation(False)
        energy_density = generator.energy_density
        energy.append(energy_density[-1])
        print(generator.energy_density)
        print(generator.probe_all)

    energy = np.asarray(energy)
    np.savez('plots/new_data/numpy/inspection/shear.npz',
             H12=x_vals,
             shear_energy=energy
             )


def compute_normal(generator):
    generator.void_shape = np.array([-0., 0.])
    x_vals = np.linspace(-0.08, 0.08, 11)
    energy = []
    for x_val in x_vals:
        generator.args.F_list_fixed = [[x_val, 0], [0., -0.125]]
        generator.def_grad = np.array([x_val, 0,  0, -0.125])
        generator._anealing_solver_fluctuation(False)
        energy_density = generator.energy_density
        energy.append(energy_density[-1])
        print(generator.energy_density)
        print(generator.probe_all)

    energy = np.asarray(energy)
    np.savez('plots/new_data/numpy/inspection/normal.npz',
             H11=x_vals,
             normal_energy=energy
             )


def plot_shear():
    data = np.load('plots/new_data/numpy/inspection/shear.npz')
    H12 = data['H12']
    energy = data['shear_energy']
    fig = plt.figure()
    plt.xticks(np.arange(min(H12), max(H12) + 1,
                         step=(max(H12) - min(H12)) / 4))
    plt.tick_params(labelsize=14)
    plt.plot(H12, 10*energy, '-*', label='energy')
    plt.show()
    fig.savefig("plots/new_data/images/shear.pdf", bbox_inches='tight')


def plot_normal():
    data = np.load('plots/new_data/numpy/inspection/normal.npz')
    H11 = data['H11']
    energy = data['normal_energy']
    fig = plt.figure()
    plt.xticks(np.arange(min(H11), max(H11) + 1,
                         step=(max(H11) - min(H11)) / 4))
    plt.tick_params(labelsize=14)
    plt.plot(H11, 10*np.flip(energy), '-*', label='energy')
    plt.show()
    fig.savefig("plots/new_data/images/normal.pdf", bbox_inches='tight')


if __name__ == '__main__':
    args = arguments.args
    generator = Generator(args)

    generator.args.relaxation_parameter = 0.05
    generator.args.max_newton_iter = 5000
    generator.enable_fast_solve = False
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 11)

    # compute_normal(generator)
    # compute_shear(generator)

    plot_normal()
    plot_shear()


