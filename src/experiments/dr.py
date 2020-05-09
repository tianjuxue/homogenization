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
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 1000
    generator.args.n_cells = 2
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 2)
    generator.enable_fast_solve = True
    run_simulation = False
    generator.args.metamaterial_mesh_size = 15
    parameters = [0., 0., -0., -0.125, -0.2, 0.2]
    generator.args.F_list_fixed = [
        [parameters[0], parameters[1]], [parameters[2], parameters[3]]]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])
    # _, _, energy_density =  generator.generate_data_single(def_grad, void_shape)
    # print(energy_density)
    energy_density = solver_fluctuation(args, void_shape,  generator.anneal_factors, def_grad)
    print(energy_density)


def simulate_full(args):
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = True
    generator.enable_dynamic_solve = True
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = False
    generator.args.F_list_fixed = [[-0., -0.], [0., -0.1]]
    generator.args.gradient = False
    generator.anneal_factors = np.linspace(0, 1, 3)
    generator.def_grad = np.array([0, 0, 0, -0.1])
    generator.void_shape = np.array([-0., 0.])
    generator._anealing_solver_disp()


def solver_fluctuation(args, void_shape, anneal_factors, def_grad):
    args.c1, args.c2 = void_shape
    pde = Metamaterial(args)
    guess = fa.Function(pde.V).vector()

    boundary_fn = fa.Constant((0, 0))

    energy_density = []
    for i, factor in enumerate(anneal_factors):
        print("   Now at step", i) 
        pde.args.F_list =  factor * np.asarray(args.F_list_fixed)
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
    simulate_periodic(args)
