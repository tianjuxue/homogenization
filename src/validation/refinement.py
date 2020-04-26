import numpy as np
import os
import fenics as fa
import sobol_seq
import sys
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
    generator.generate_data_single(def_grad, void_shape)
    energy = generator.energy_density
    mesh_max = generator.pde.mesh.hmax()
    mesh_min = generator.pde.mesh.hmin()
    n_ele =  generator.pde.mesh.num_cells()
    # magic number 0.5 is the area
    h = np.sqrt(0.5/n_ele)
    return h, generator.pde, energy[-1]


def ref(generator):
    resolution = [7.5, 15, 30, 60]
    pdes = []
    energy = []
    mesh_size = []
    error = []
    for res in resolution:
        h, pde, e = simulate(generator, res) 
        pdes.append(pde)
        energy.append(e)
        mesh_size.append(h)

    for i in range(len(resolution) - 1):
        err = get_error(pdes[-1], pdes[i])
        error.append(err)

    print(error)
    print(mesh_size)

    plt.figure()
    plt.loglog(mesh_size[:-1], error, linestyle='--', marker='o', color='red')
    plt.tick_params(labelsize=14)
    plt.show()

    # plt.figure()
    # plt.plot(mesh_size, energy, linestyle='--', marker='o', color='red')
    # plt.tick_params(labelsize=14)
    # plt.show()

def get_error(pde1, pde2):
    V1 = pde1.V
    V2 = pde2.V
    u1 = pde1.u
    u2 = fa.interpolate(pde2.u, V1)
    error = fa.Function(V1)
    error.vector().set_local(u1.vector()[:] - u2.vector()[:])
    return fa.norm(error)


if __name__ == '__main__':
    args = arguments.args
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 1000
    generator.args.n_cells = 2
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 11)
    generator.enable_fast_solve = True

    ref(generator)
