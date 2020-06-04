import fenics as fa
import math
import numpy as np
import matplotlib.pyplot as plt
import os.path
from ..pde.static import Metamaterial
from .. import arguments


def solver_fluctuation(args):
    args.c1, args.c2 = args.void_shape

    pore_type = 0 if np.sum(np.absolute(args.void_shape)) < 1e-3  else 2 
    mesh_name = 'plots/new_data/mesh/RVE_size' + str(args.n_cells) + '_pore' + str(pore_type) + '.xml'
    if os.path.isfile(mesh_name):
        mesh = fa.Mesh(mesh_name)
    else:
        mesh = None    

    pde = Metamaterial(args, mesh)
    boundary_fn = fa.Constant((0, 0))
    energy_density = []

    try:
        pde.args.F_list = np.asarray(args.F_list_fixed)
        u = pde.solve_problem(boundary_fn=None,
                              boundary_point_fn=boundary_fn,
                              boundary_fn_dic=None,
                              initial_guess=None,
                              enable_fast_solve=True,
                              enable_dynamic_solve=True)
        energy = pde.energy(u)
        energy_density.append(energy / pow(args.n_cells * args.L0, 2))
    except Exception as e:
        print("\nDirect dr solve fails, try annealled dr solve...")
        guess = fa.Function(pde.V).vector()
        anneal_factors = np.linspace(0, 1, 11)
        for i, factor in enumerate(anneal_factors):
            print("   Now at step", i)
            pde.args.F_list = factor * np.asarray(args.F_list_fixed)
            try:
                u = pde.solve_problem(boundary_fn=None,
                                      boundary_point_fn=boundary_fn,
                                      boundary_fn_dic=None,
                                      initial_guess=guess,
                                      enable_fast_solve=True,
                                      enable_dynamic_solve=True)
            except:
                return None
            guess = u.vector()
            energy = pde.energy(u)
            energy_density.append(energy / pow(args.n_cells * args.L0, 2))

    if not os.path.isfile(mesh_name):
        fa.File(mesh_name) << pde.mesh
            
    e11, e12, e21, e22 = args.def_grad
    affine_fn = fa.Expression(('e11*x[0] + e12*x[1]', 'e21*x[0] + e22*x[1]'),
                              e11=e11, e12=e12, e21=e21, e22=e22, degree=2)
    result = fa.project(affine_fn + u, pde.V_non_periodic)
    stress = pde.force(u)
    return energy_density[-1], stress[1, 1]

def run():
    n1 = 21
    n2 = 11
    W = np.zeros((n1, n2))
    P = np.zeros((n1, n2))
    H11 = np.linspace(-0.1, 0.1, n1)
    H22 = np.linspace(0, -0.125, n2)
    for i, h11 in enumerate(H11):
        for j, h22 in enumerate(H22):
            print("\ni is {} and j is {}".format(i, j))
            parameters = np.array([h11, 0, 0, h22, 0, 0])
            args.F_list_fixed = [[parameters[0], parameters[1]], [parameters[2], parameters[3]]]
            args.def_grad = np.array(parameters[0:4])
            args.void_shape = np.array(parameters[4:6])
            w, s = solver_fluctuation(args)
            W[i, j] = w
            P[i, j] = s
    np.save('sheng_mao/data/W21x11.npy', W)
    np.save('sheng_mao/data/P21x11.npy', P)


if __name__ == '__main__':
    args = arguments.args
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 1000
    args.n_cells = 2
    args.fluctuation = True
    args.enable_fast_solve = True
    args.metamaterial_mesh_size = 15
    run()