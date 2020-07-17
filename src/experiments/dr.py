'''这个文件测试了dr solver对于RVE和DNS的应用
'''
import numpy as np
import os
import fenics as fa
import sobol_seq
import sys
import time
import matplotlib.pyplot as plt
from ..pde.static import Metamaterial
from ..collector.generator import Generator
from .. import arguments


def simulate_periodic(args):
    args.n_cells = 2
    args.fluctuation = True
    anneal_factors = np.linspace(0, 1, 11)
    parameters = [0., 0., -0., -0.125, -0.0, 0.0]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])
    args.F_list_fixed = [[parameters[0], parameters[1]],
                         [parameters[2], parameters[3]]]
    energy_density = solver_fluctuation(
        args, void_shape,  anneal_factors, def_grad)
    print(energy_density)


def simulate_full(args):
    args.n_cells = 4
    args.fluctuation = False
    anneal_factors = np.linspace(0, 1, 21)
    parameters = [0., 0., -0., -0.1, -0., 0.]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])
    energy_density, force = solver_disp(
        args, void_shape, anneal_factors, def_grad, return_force=True)
    force = np.asarray([f[1][1] for f in force]) / (args.n_cells * args.L0)
    # np.save('plots/new_data/numpy/size_effect/DNS_force_com_pore0_size' + str(args.n_cells) + '_dr.npy', force)


def solver_disp(args, void_shape, anneal_factors, def_grad, return_force=False):
    args.c1, args.c2 = void_shape
    pde = Metamaterial(args)
    guess = fa.Function(pde.V).vector()
    energy_density = []
    force = []
    pde.args.F_list = None
    file = fa.File("tmp/dr/u.pvd")
    for i, factor in enumerate(anneal_factors):
        print("   Now at step", i)
        e11, e12, e21, e22 = factor * def_grad
        boundary_fn = fa.Expression(('e11*x[0] + e12*x[1]', 'e21*x[0] + e22*x[1]'),
                                    e11=e11, e12=e12, e21=e21, e22=e22, degree=2)

        boundary_fn_dic = {'bottom': fa.Constant(
            (0, 0)), 'top': boundary_fn}

        u = pde.solve_problem(boundary_fn=None,
                              boundary_point_fn=None,
                              boundary_fn_dic=boundary_fn_dic,
                              initial_guess=guess,
                              enable_fast_solve=True,
                              enable_dynamic_solve=True)

        guess = u.vector()
        energy = pde.energy(u)
        energy_density.append(
            energy / pow(args.n_cells * args.L0, 2))

        u.rename('u', 'u')
        file << (u, i)

        if return_force:
            force.append(pde.force(u))

    print("Total energy is", energy)

    if return_force:
        return energy_density, force

    return energy_density


def solver_fluctuation(args, void_shape, anneal_factors, def_grad):
    args.c1, args.c2 = void_shape
    pde = Metamaterial(args)
    guess = fa.Function(pde.V).vector()

    boundary_fn = fa.Constant((0, 0))

    energy_density = []
    for i, factor in enumerate(anneal_factors):
        print("   Now at step", i)
        pde.args.F_list = factor * np.asarray(args.F_list_fixed)
        u = pde.solve_problem(boundary_fn=None,
                              boundary_point_fn=boundary_fn,
                              boundary_fn_dic=None,
                              initial_guess=guess,
                              enable_fast_solve=True,
                              enable_dynamic_solve=True)
        guess = u.vector()
        energy = pde.energy(u)
        energy_density.append(energy / pow(args.n_cells * args.L0, 2))

    e11, e12, e21, e22 = def_grad
    affine_fn = fa.Expression(('e11*x[0] + e12*x[1]', 'e21*x[0] + e22*x[1]'),
                              e11=e11, e12=e12, e21=e21, e22=e22, degree=2)
    result = fa.project(affine_fn + u, pde.V_non_periodic)

    file = fa.File("u.pvd")
    result.rename('u', 'u')
    file << result

    return energy_density


if __name__ == '__main__':
    args = arguments.args
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 1000
    args.enable_fast_solve = True
    args.gradient = False
    args.metamaterial_mesh_size = 15
    simulate_periodic(args)
