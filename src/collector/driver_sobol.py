import numpy as np
import os
import fenics as fa
import sobol_seq
import sys
from .. import arguments
from .driver_ray import save_fn
from ..pde.static import Metamaterial


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


def run_trial_sobol(args, path, index, parameters):
    args.F_list_fixed = [
        [parameters[0], parameters[1]], [parameters[2], parameters[3]]]
    args.def_grad = np.array(parameters[0:4])
    args.void_shape = np.array(parameters[4:6])
    energy, _ = solver_fluctuation(args)
    if energy is not None:
        save_fn(args,
                'number_' + str(index),
                path,
                args.def_grad,
                args.void_shape,
                energy)

        print("Successfully saved these data")
        print("\ndef_grad_all", args.def_grad)
        print("\nvoid_shape_all", args.void_shape)
        print("\npredicted_energy_all", energy)
    else:
        print("Fail to save any data for this configuration.")


def get_parameters_pore0(vec):
    parameters = [0.4 * (vec[0] - 0.5),
                  0.4 * (vec[1] - 0.5),
                  0.4 * (vec[2] - 0.5),
                  0.4 * (vec[3] - 0.5),
                  -0.,
                  0.]
    return parameters


def get_parameters_pore1(vec):
    parameters = [0.6 * (vec[0] - 0.5),
                  1. * (vec[1] - 0.5),
                  1. * (vec[2] - 0.5),
                  0.6 * (vec[3] - 0.5),
                  -0.1,
                  0.1]
    return parameters


def get_parameters_pore2(vec):
    parameters = [0.4 * (vec[0] - 0.5),
                  1.6 * (vec[1] - 0.5),
                  1.6 * (vec[2] - 0.5),
                  0.4 * (vec[3] - 0.5),
                  -0.2,
                  0.2]
    return parameters


def collect_sobol(args, path, get_parameters):
    if not os.path.exists(path):
        os.mkdir(path)

    total_number_samples = 5000
    vec = sobol_seq.i4_sobol_generate(4, total_number_samples)
    start = 4723
    print("total_number_samples is", total_number_samples)
    for i in range(start, total_number_samples):
        print("\n#####################################################################")
        parameters = get_parameters(vec[i])
        print("Simulation for round", i)
        print("parameters is", parameters)
        run_trial_sobol(args=args, path=path, index=i, parameters=parameters)


if __name__ == '__main__':
    args = arguments.args
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 1000
    args.n_cells = 2
    args.fluctuation = True
    args.enable_fast_solve = True
    args.metamaterial_mesh_size = 15
    collect_sobol(args, path='saved_data_pore0_sobol_dr', get_parameters=get_parameters_pore0)
    # collect_sobol(args, path='saved_data_pore2_sobol_dr', get_parameters=get_parameters_pore2)
