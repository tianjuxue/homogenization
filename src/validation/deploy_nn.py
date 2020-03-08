"""Archived quick dirty tests"""
from dolfin import *
import ufl
from ufl.operators import *
import math
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ..trainer.loader import Net
from .. import arguments


class Left(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)


class Right(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], args.n_macro * args.L0)


class Bottom(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)


class Top(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], args.n_macro * args.L0)


class KinkExpression(UserExpression):

    def __init__(self, param, disp):
        # Construction method of base class has to be called first
        super(KinkExpression, self).__init__()
        self.param = param
        self.disp = disp

    def eval(self, values, x):
        L = args.n_macro * args.L0
        if x[1] < L / 2:
            values[0] = 2 * self.param * x[1]
        else:
            values[0] = -2 * self.param * (x[1] - L)

        values[1] = x[1] * self.disp

    def value_shape(self):
        return (2, )


class XiExpression(UserExpression):

    def eval(self, values, x):
        L = args.n_macro * args.L0
        values[0] = x[1] / L * -0.2
        values[1] = x[1] / L * 0.2

    def value_shape(self):
        return (2,)


def sigmoid(x):
    return 1 / (1 + ufl.operators.exp(-x))


def layer(x, weights, bias, id):
    weights = weights.data.numpy()
    bias = bias.data.numpy()
    input_size = weights.shape[1]
    output_size = weights.shape[0]
    output = []
    for i in range(output_size):
        tmp = 0
        for j in range(input_size):
            tmp = tmp + weights[i][j].item() * x[j]
        tmp = tmp + bias[i].item()
        if id is 1:
            tmp = sigmoid(tmp)
        elif id is 2:
            pass
        else:
            pass
        output.append(tmp)
    return output


def manual_nn(network, x):
    x = layer(x, network.fc1.weight, network.fc1.bias, 1)
    x = layer(x, network.fc2.weight, network.fc2.bias, 2)
    assert (len(x) == 1)
    return x[0]


def get_energy(u, pore_flag, network, V):
    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)
    F = I + grad(u)
    F = variable(F)
    C = F.T * F
    C00 = C[0, 0]
    C01 = C[0, 1]
    C11 = C[1, 1]
    if pore_flag == 0:
        C_list = [C00, C01, C11, -0., 0.]
    elif pore_flag == 1:
        C_list = [C00, C01, C11, -0.2, 0.2]
    else:
        xi_exp = XiExpression()
        xi = interpolate(xi_exp, V)
        C_list = [C00, C01, C11, xi[0], xi[1]]

    energy = manual_nn(network, C_list)
    stress = diff(energy, F)
    return energy, stress


def homogenization(args, disp, pore_flag):
    # pore_flag = 0 means pore A
    # pore_flag = 1 means pore B
    # pore_flag = 2 means mixed

    print("Start to solve with disp={:.6f} and pore_flag={}".format(
        disp, pore_flag))
    model_path = args.checkpoints_path + '/model_step_999'
    # model_path = 'saved_checkpoints_tmp/new_universal'
    network = torch.load(model_path)

    parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {
        "optimize": True,
        "eliminate_zeros": True,
        "precompute_basis_const": True,
        "precompute_ip_const": True
    }

    # Create mesh and define function space
    mesh = UnitSquareMesh(10, 10)
    mesh = RectangleMesh(Point(0, 0), Point(
        args.n_macro * args.L0, args.n_macro * args.L0), 10, 10)

    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)

    left = Left()
    left.mark(sub_domains, 1)
    right = Right()
    right.mark(sub_domains, 2)
    bottom = Bottom()
    bottom.mark(sub_domains, 3)
    top = Top()
    top.mark(sub_domains, 4)

    normal = FacetNormal(mesh)
    ds = Measure("ds")(subdomain_data=sub_domains)

    # Define Dirichlet boundary (x = 0 or x = 1)
    c = Expression(("0.0", "0.0"), degree=1)
    r = Expression(("0.0", "disp"), disp=disp *
                   args.n_macro * args.L0, degree=1)

    bcb = DirichletBC(V, c, bottom)
    bct = DirichletBC(V, r, top)
    bcs = [bcb, bct]

    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    T = Constant((0.0, 0.0))

    energy, stress = get_energy(u, pore_flag, network, V)
    energy_ref = assemble(energy * dx)
    # print("Total energy (reference) is", energy_ref)
    force_ref = assemble(dot(stress, normal)[1] * ds(4))
    # print("Total force (reference) is", force_ref)

    initial_value = 0
    if disp < 0 and pore_flag == 1:
        initial_value = 0.2
    initial_exp = KinkExpression(initial_value, disp)
    u = interpolate(initial_exp, V)
    energy, stress = get_energy(u, pore_flag, network, V)

    Energy = energy * dx
    F = derivative(Energy, u, v)
    J = derivative(F, u, du)

    newton_args = {
        'relaxation_parameter': args.relaxation_parameter,
        'linear_solver': args.linear_solver,
        'maximum_iterations': args.max_newton_iter,
        "relative_tolerance": 1e-4,
        "absolute_tolerance": 1e-4
    }
    solver_args = {
        'nonlinear_solver': args.nonlinear_solver,
        'newton_solver': newton_args
    }

    parameters["form_compiler"]["cpp_optimize"] = True

    ffc_options = {"optimize": True,
                   "eliminate_zeros": True,
                   "precompute_basis_const": True,
                   "precompute_ip_const": True}

    solve(F == 0, u, bcs, J=J,
          solver_parameters=solver_args,
          form_compiler_parameters=ffc_options)

    energy_def = assemble(energy * dx)

    # print("Total energy (deformed) is", energy_def)
    # print("Average energy density is",
    #       assemble(energy * dx) / (args.n_macro * args.L0)**2)
    force_def = assemble(dot(stress, normal)[1] * ds(4))
    # print("Total force (deformed) is", force_def)

    energy_rel = energy_def - energy_ref
    force_rel = force_def - force_ref
    # print("Total energy (relative) is", energy_rel)
    # print("Total force (relative) is", force_rel)

    # Save solution in VTK format
    file = File("u.pvd")
    u.rename('u', 'u')
    file << u

    return energy_rel, force_rel, u


def run_and_save(factors, disp, pore_flag, name):
    start = time.time()
    energy_list = []
    force_list = []
    for factor in factors:
        energy, force, u = homogenization(args, disp * factor, pore_flag)
        energy_list.append(energy)
        force_list.append(force)

    end = time.time()
    time_elapsed = end - start

    deform_info = 'com' if disp < 0 else 'ten'
    np.save('plots/new_data/numpy/energy/' + name + '_energy_' + deform_info +
            '_pore' + str(pore_flag) + '.npy', np.asarray(energy_list))
    np.save('plots/new_data/numpy/force/' + name + '_force_' + deform_info +
            '_pore' + str(pore_flag) + '.npy', np.asarray(force_list))
    np.save('plots/new_data/numpy/time/' + name + '_time_' + deform_info +
            '_pore' + str(pore_flag) + '.npy', np.asarray(time_elapsed))

    print('energy_list', energy_list)
    print('force_list', force_list)
    print('time_elapsed', time_elapsed)
    print('\n')


def run():
    factors = np.linspace(0, 1, 9)
    run_and_save(factors, disp=-0.125, pore_flag=0, name='NN')
    run_and_save(factors, disp=-0.125, pore_flag=1, name='NN')
    run_and_save(factors, disp=0.125, pore_flag=0, name='NN')
    run_and_save(factors, disp=0.125, pore_flag=1, name='NN')

if __name__ == '__main__':
    args = arguments.args
    args.n_macro = 8
    args.relaxation_parameter = 0.1
    args.max_newton_iter = 2000

    energy, force, u = homogenization(args, disp=-0.1, pore_flag=1)
