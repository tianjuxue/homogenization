"""Archived quick dirty tests"""
from dolfin import *
import ufl
from ufl.operators import *

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ..trainer.loader import Net
from .. import arguments

# model_path = 'saved_checkpoints_tmp/normal' is the way to go
# for universal case run
# python -m src.experiments.ufl_function --relaxation_parameter 0.005 --max_newton_iter 3000


class LowerLeft(SubDomain):
    def inside(self, x, on_boundary):                    
        return near(x[0], 0) and near(x[1], 0)

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

class SinExpression(UserExpression):
    def __init__(self, param):
        # Construction method of base class has to be called first
        # see https://github.com/FEniCS/dolfin/blob/df4eea209a85c461e3dcb7c21d4b4c015c46ecdf/python/dolfin/function/expression.py
        super(SinExpression, self).__init__() 
        self.param = param

    def eval(self, values, x):
        L = args.n_macro*args.L0
        values[0] = self.param*L*sin(pi/L*x[1])
        # values[0] = sin(pi/L*x[1])
        values[1] = x[1]*disp

    def value_shape(self):
        return (2,)

class KinkExpression(UserExpression):
    def __init__(self, param):
        # Construction method of base class has to be called first
        super(KinkExpression, self).__init__() 
        self.param = param

    def eval(self, values, x):
        L = args.n_macro*args.L0

        if x[1] < L/2:
            values[0] = 2*self.param*x[1]
        else:
            values[0] = -2*self.param*(x[1] - L)

        values[1] = x[1]*disp

    def value_shape(self):
        return (2,)

class NegSinExpression(UserExpression):
    def __init__(self, param):
        # Construction method of base class has to be called first
        super(NegSinExpression, self).__init__() 
        self.param = param

    def eval(self, values, x):
        L = args.n_macro*args.L0
        if x[0] < L/2:
            ratio = (L/2 - x[0])/(L/2)
            values[0] = ratio*self.param*L*sin(pi/L*x[1])
        else:
            ratio = -(L/2 - x[0])/(L/2)
            values[0] = -ratio*self.param*L*sin(pi/L*x[1])
        values[1] = x[1]*disp

    def value_shape(self):
        return (2,)


class NegPoissonExpression(UserExpression):
    def __init__(self, param):
        # Construction method of base class has to be called first
        super(NegPoissonExpression, self).__init__() 
        self.param = param

    def eval(self, values, x):
        L = args.n_macro*args.L0
        values[0] = x[0]*self.param
        values[1] = x[1]*disp

    def value_shape(self):
        return (2,)


def relu(x):
    return x/2*(ufl.operators.sign(x) + 1)

def sigmoid(x):
    return 1/(1+ufl.operators.exp(-x))

def layer(x, weights, bias, id):
    weights = weights.data.numpy()
    bias = bias.data.numpy()
    input_size = weights.shape[1]
    output_size = weights.shape[0]
    output = []
    for i in range(output_size):
        tmp = 0
        for j in range(input_size):
            tmp = tmp + weights[i][j].item()*x[j]
        tmp = tmp + bias[i].item()
        if id is 1:
            tmp = sigmoid(tmp)
        elif id is 2:
            pass 
        else:
            pass      
        output.append(tmp)
    return output

def manual_nn(x):
    x = layer(x, network.fc1.weight, network.fc1.bias, 1)
    x = layer(x, network.fc2.weight, network.fc2.bias, 2)
    # x = layer(x, network.fc3.weight, network.fc3.bias, 3)
    assert(len(x)==1)
    return x[0]


def feature_map(C, C_square, I1, J, n=2, m=1):
    coef = [0. ,         0.13211346 , 4.11426334, -2.6787249 ,  1.02888553, -0.20085012,\
            0.01485494, -2.46025208,  1.05326055, -0.1149104]
    coef = [0.,          1.99715429 , 0.38999163 ,-0.45676305]
    d = 2
    Features = 0
    counter = 0

    for i in range(n + 1):
        Features = Features + coef[counter]*(I1 - d)**i 
        counter = counter + 1

    for i in range(1, m + 1):
        Features = Features + coef[counter]*(J - 1)**(2*i)
        counter = counter + 1

    assert(len(coef) == counter)

    return Features


def inspect_energy():
    x_value = np.linspace(-0.3, 0.3, 30)
    y_value = []
    for x_val in x_value:

        # initial_exp = Expression(("traverse*sin(pi/L*x[1])", "x[1]*disp"), traverse=x_val*args.n_macro*args.L0, L=args.n_macro*args.L0, disp=disp, degree=3)
        # initial_exp = Expression(("0", "x[1]*x_val"), x_val=x_val, disp=disp, degree=3)
        # initial_exp = NegSinExpression(x_val)
        initial_exp = SinExpression(x_val)        
        u = interpolate(initial_exp, V)

        shear_mod = args.young_modulus / (2 * (1 + args.poisson_ratio))
        bulk_mod = args.young_modulus / (3 * (1 - 2*args.poisson_ratio))

        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        J  = det(F)
        C = F.T*F
        C_square = C*C
        I1 = tr(C)
        I2 = 0.5*(tr(C)**2 - tr(C*C))
        I3 = det(C)

        I1_b = J**(-2 / d)*I1
        I2_b = J**(-4 / d)*I2

        C00 = C[0, 0]
        C01 = C[0, 1]
        C11 = C[1, 1]
        C_list = [C00, C01, C11, -0.2, 0.2]
        energy = manual_nn(C_list) + 0.*(J - 1)**2
        # energy = 1 + (I1_b - d) + 0*(I2_b - d) + 0.01*(J - 1)**2
        # energy = feature_map(C, C_square, I1, I2, J)

        total_energy = assemble(energy*dx)
        print(total_energy)
        y_value.append(total_energy)

    file = File("test.pvd");
    file << u;

    y_value = np.asarray(y_value)
    plt.figure()
    plt.plot(x_value, y_value, '*-')
    plt.show()


if __name__ == '__main__':

    args = arguments.args
    args.n_macro = 8
    set_log_level(20)

    # model_path = args.checkpoints_path_shear + '/model_step_' + str(1000*49)
    model_path = 'saved_checkpoints_tmp/normal'
    network =  torch.load(model_path)

    # print(network.fc1.weight)
    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {"optimize": True, \
                   "eliminate_zeros": True, \
                   "precompute_basis_const": True, \
                   "precompute_ip_const": True}

    # Create mesh and define function space
    mesh = UnitSquareMesh(10, 10)
    mesh = RectangleMesh(Point(0, 0), Point(args.n_macro*args.L0, args.n_macro*args.L0), 10, 10) 

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

    disp = -0.125
    # Define Dirichlet boundary (x = 0 or x = 1)
    c = Expression(("0.0", "0.0"), degree=1)
    r = Expression(("0.0", "disp"), disp=disp*args.n_macro*args.L0, degree=1)

    c_sub = Constant(0.2)
    r_sub = Expression(("disp"), disp=disp*args.n_macro*args.L0, degree=1)

    bcb = DirichletBC(V, c, bottom)
    bct = DirichletBC(V, r, top)
    bcs = [bcb, bct]
    # bcb_sub = DirichletBC(V.sub(1), c_sub, bottom)
    # bct_sub = DirichletBC(V.sub(1), r_sub, top)
    # bc_pt = DirichletBC(V, 
    #                     Constant((0, 0)), 
    #                     LowerLeft(), 
    #                     method='pointwise')
    # bcs = [bcb_sub, bct_sub, bc_pt]

    # Define function2
    du = TrialFunction(V)            # Incremental displacement
    v  = TestFunction(V)             # Test function
    u  = Function(V)                 # Displacement from previous iteration
    T  = Constant((0.0,  0.0))  # Traction force on the boundary

    if False:
        inspect_energy()
        exit()

    # initial_exp = Expression(("traverse*sin(pi/L*x[1])", "x[1]*disp"), traverse=0.3*args.n_macro*args.L0, L=args.n_macro*args.L0, disp=disp, degree=3)
    # u.vector().set_local(initial_guess)
    # initial_exp = NegSinExpression(0.0)
    initial_exp = KinkExpression(0.2)
    u = interpolate(initial_exp, V)

    shear_mod = args.young_modulus / (2 * (1 + args.poisson_ratio))
    bulk_mod = args.young_modulus / (3 * (1 - 2*args.poisson_ratio))

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)
    F = I + grad(u)

    F = variable(F)

    J  = det(F)
    C = F.T*F
    C_square = C*C    
    I1 = tr(C)
    Jinv = J**(-2 / 3) 

    C00 = C[0, 0]
    C01 = C[0, 1]   
    C11 = C[1, 1]
    C_list = [C00, C01, C11, -0.2, 0.2]

    # energy = ((shear_mod / 2) * (Jinv * (I1 + 1) - 3) + (bulk_mod / 2) * (J - 1)**2) 
    energy = manual_nn(C_list) + 0*(J - 1)**2
    stress = diff(energy, F)

    # energy = ((shear_mod / 2) * (Jinv * I1 - d) + (bulk_mod / 2) * (J - 1)**2) + 10*((C[0, 0] - 1)**2 + (C[1, 1] - 1)**2)
    # energy = ((shear_mod / 2) * (Jinv * I1 - d) + (bulk_mod / 2) * (J - 1)**2)
    # energy = (C[0, 0] - 1)**2 + (C[1, 1] - 1)**2 + F[1, 0]**2 + F[0, 1]**2

    print(assemble(energy*dx))
    print(assemble(energy*dx)/(args.n_macro*args.L0)**2)

    print("Total energy reference is", assemble(energy*dx))
    Energy = energy*dx
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Energy, u, v)
    # Compute Jacobian of F
    J = derivative(F, u, du)
    # Solve variational problem

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

    ffc_options = {"optimize": True, \
                   "eliminate_zeros": True, \
                   "precompute_basis_const": True, \
                   "precompute_ip_const": True}

    solve(F == 0, u, bcs, J=J,
            solver_parameters=solver_args,
            form_compiler_parameters=ffc_options)

    print("Total energy is", assemble(energy*dx))
    print("Average energy density is", assemble(energy*dx)/(args.n_macro*args.L0)**2)

    print("bottom")   
    print(assemble(dot(stress, normal)[1]*ds(3)))
    # print("up")
    # print(assemble(dot(stress, normal)[1]*ds(4)))

    # solve(F == 0, u, bcs, J=J,
    #       form_compiler_parameters=ffc_options)

    # problem = NonlinearVariationalProblem(F, u, bcs, J=J, form_compiler_parameters=ffc_options)
    # solver = NonlinearVariationalSolver(problem)
    # solver.parameters['nonlinear_solver'] = 'snes'
    # solver.parameters["snes_solver"]["maximum_iterations"] = 50
    # solver.parameters["snes_solver"]["report"] = False
    # solver.solve()

    # Save solution in VTK format
    file = File("displacement_Energy.pvd")
    u.rename('u', 'u')
    file << u