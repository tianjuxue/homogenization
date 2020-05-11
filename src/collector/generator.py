import fenics as fa
import math
import numpy as np
import matplotlib.pyplot as plt
import os.path
from ..pde.static import Metamaterial


class SinExpression(fa.UserExpression):

    def __init__(self, args, param):
        # Construction method of base class has to be called first
        super(SinExpression, self).__init__()
        # self.param = param
        self.args = args
        self.amplitude = param * self.args.L0 / 2

    def eval(self, values, x):
        L = self.args.L0 / 2
        A = self.amplitude

        values[0] = A * fa.sin(fa.pi / (2 * L) * x[1])
        values[1] = -A * fa.sin(fa.pi / (2 * L) * x[0])

    def value_shape(self):
        return (2,)


class SinPoreExpression(fa.UserExpression):

    def __init__(self, args, param):
        # Construction method of base class has to be called first
        super(SinPoreExpression, self).__init__()
        self.args = args
        self.amplitude = param * self.args.L0 / 2

    def eval(self, values, x):

        L = self.args.L0 / 2
        A = self.amplitude
        region_id = self.get_region_id(x)
        center = self.get_center(region_id)
        sign = self.get_sign(region_id)

        delta_x = x - center
        delta_x_tmp = delta_x.copy()

        delta_x[0] = delta_x_tmp[0] * sign * A / L * \
            fa.sin(fa.pi / (2 * L) * (L - delta_x_tmp[1]))
        delta_x[1] = delta_x_tmp[1] * \
            (-sign * A / L * fa.sin(fa.pi / (2 * L) * (L - delta_x_tmp[0])))

        values[0] = delta_x[0]
        values[1] = delta_x[1]

    def get_sign(self, region_id):
        return region_id % 2 * 2 - 1

    def value_shape(self):
        return (2,)

    def get_center(self, region_id):
        center = np.array([0., 0.])

        if region_id == 0:
            center[0] = self.args.L0 / 2
            center[1] = self.args.L0 / 2
        elif region_id == 1:
            center[0] = self.args.L0 / 2 + self.args.L0
            center[1] = self.args.L0 / 2
        elif region_id == 2:
            center[0] = self.args.L0 / 2 + self.args.L0
            center[1] = self.args.L0 / 2 + self.args.L0
        elif region_id == 3:
            center[0] = self.args.L0 / 2
            center[1] = self.args.L0 / 2 + self.args.L0
        else:
            Exception(
                "region_id should be 0, 1, 2 or 3 but got {}".format(region_id))

        return center

    def get_region_id(self, x):
        if (x[0] < self.args.L0 and x[1] < self.args.L0):
            return 0
        elif (x[0] > self.args.L0 and x[1] < self.args.L0):
            return 1
        elif (x[0] > self.args.L0 and x[1] > self.args.L0):
            return 2
        else:
            return 3


class Generator(object):

    def __init__(self, args):
        self.args = args
        self.n_cells_array = np.array([2, 3, 4])
        self.anneal_steps = 11
        self.anneal_factors = np.linspace(0, 1, self.anneal_steps)
        self.enable_fast_solve = True
        self.enable_dynamic_solve = False
        self.args.n_cells = 1
        self.args.fluctuation = False
        self.args.F_list = None
        self.args.gradient = False
        self._random_data()

    def generate_data_single(self, def_grad, void_shape):
        self.def_grad = def_grad
        self.void_shape = void_shape
        self.def_grad_all = []
        self.void_shape_all = []
        self.predicted_energy_all = []

        for i, factor in enumerate(self.anneal_factors):
            self.def_grad_all.append(factor * self.def_grad)
            self.void_shape_all.append(self.void_shape)

        self.predicted_energy_all = self._anealing_solver_fluctuation()
        return self.def_grad_all, self.void_shape_all, self.predicted_energy_all

    # def _get_factor(self, pde):
    #     x_vals = np.linspace(0, 1, 11)
    #     for x_val in x_vals:
    #         sin_pore_exp = SinPoreExpression(self.args, x_val * 0.4)
    #         u = fa.interpolate(sin_pore_exp, pde.V)
    #         energy = pde.energy(u)
    #         file = fa.File("u.pvd")
    #         file << u
    #         print(energy)

    def _anealing_solver_fluctuation(self, return_force=False):
        self.args.c1, self.args.c2 = self.void_shape

        pore_type = 0 if np.sum(np.absolute(self.void_shape)) < 1e-3  else 2 
        mesh_name = 'plots/new_data/mesh/RVE_' + str(self.args.n_cells) + '_pore' + str(pore_type) + '.xml'
        if os.path.isfile(mesh_name):
            mesh = fa.Mesh(mesh_name)
        else:
            mesh = None

        if pore_type is 0:
            guide = True
        else:
            guide = False

        pde = Metamaterial(self.args, mesh)
        if not os.path.isfile(mesh_name):
            fa.File(mesh_name) << pde.mesh

        guess = fa.Function(pde.V).vector()

        max_amp = self.args.L0 / 2
        probe_point = fa.Point((0., max_amp))
        probe_guide = 0.2 * max_amp
        probe_value = probe_guide

        self.energy_density = []
        self.force = []
        self.probe_all = []

        for i, factor in enumerate(self.anneal_factors):
            print("   Now at step", i)
            boundary_fn = fa.Constant((0, 0))

            pde.args.F_list = factor * np.asarray(self.args.F_list_fixed)

            if  (pde.args.F_list[0][0] < 0 or pde.args.F_list[1][1] < 0) and guide:
                # print("   Start guidance")
                # sin_pore_exp = SinPoreExpression(self.args, 0.2)
                sin_exp = SinExpression(self.args, probe_value / max_amp)
                if i < 2:
                    guess = fa.interpolate(sin_exp, pde.V).vector()

                u = pde.solve_problem(boundary_fn=sin_exp,
                                      boundary_point_fn=None,
                                      boundary_fn_dic=None,
                                      initial_guess=guess,
                                      enable_fast_solve=self.enable_fast_solve)

                guess = u.vector()
                # print("   probe_point reports value", u(probe_point)[0])

            u = pde.solve_problem(boundary_fn=None,
                                  boundary_point_fn=boundary_fn,
                                  boundary_fn_dic=None,
                                  initial_guess=guess,
                                  enable_fast_solve=self.enable_fast_solve)

            if (pde.args.F_list[0][0] >= 0 and pde.args.F_list[1][1] >= 0) or not guide:
                guess = u.vector()

            # print("   probe_point reports value", u(probe_point)[0])
            probe_follow = u(probe_point)[0]
            probe_value = max(np.absolute(probe_guide),
                              np.absolute(probe_follow))

            self.probe_all.append(probe_follow)

            energy = pde.energy(u)
            # print("   Energy in this round", energy)
            self.energy_density.append(
                energy / pow(self.args.n_cells * self.args.L0, 2))

            if return_force:
                self.force.append(pde.force(u))

        e11, e12, e21, e22 = self.def_grad
        affine_fn = fa.Expression(('e11*x[0] + e12*x[1]', 'e21*x[0] + e22*x[1]'),
                                  e11=e11, e12=e12, e21=e21, e22=e22, degree=2)
        result = fa.project(affine_fn + u, pde.V_non_periodic)

        file = fa.File("tmp/u.pvd")
        result.rename('u', 'u')
        file << result
        self.pde = pde
        self.pde.u = u
        self.pde.result = result
        
        if return_force:
            return self.energy_density, self.force

        return self.energy_density

    def _anealing_solver_disp(self, return_all=False):
        self.args.c1, self.args.c2 = self.void_shape

        pore_type = 0 if np.sum(np.absolute(self.void_shape)) < 1e-3  else 2 
        if self.args.padding:
            mesh_name = 'plots/new_data/mesh/DNS_padding_size' + str(self.args.n_cells) + '_pore' + str(pore_type) + '.xml'
        else:
            mesh_name = 'plots/new_data/mesh/DNS_size' + str(self.args.n_cells) + '_pore' + str(pore_type) + '.xml'
            
        if os.path.isfile(mesh_name):
            mesh = fa.Mesh(mesh_name)
        else:
            mesh = None

        pde = Metamaterial(self.args, mesh)
        if not os.path.isfile(mesh_name):
            fa.File(mesh_name) << pde.mesh

        guess = fa.Function(pde.V).vector()

        self.energy_density = []
        self.force = []
        self.sols = []

        file = fa.File("tmp/u.pvd")
        for i, factor in enumerate(self.anneal_factors):
            print("   Now at step", i)
            e11, e12, e21, e22 = factor * self.def_grad
            boundary_fn = fa.Expression(('e11*x[0] + e12*x[1]', 'e21*x[0] + e22*x[1]'),
                                        e11=e11, e12=e12, e21=e21, e22=e22, degree=2)

            boundary_fn_dic = {'bottom': fa.Constant(
                (0, 0)), 'top': boundary_fn}
            u = pde.solve_problem(boundary_fn=None,
                                  boundary_point_fn=None,
                                  boundary_fn_dic=boundary_fn_dic,
                                  initial_guess=guess,
                                  enable_fast_solve=self.enable_fast_solve,
                                  enable_dynamic_solve=self.enable_dynamic_solve)

            guess = u.vector()
            energy = pde.energy(u)
            self.energy_density.append(
                energy / pow(self.args.n_cells * self.args.L0, 2))

            if return_all:
                self.force.append(pde.force(u))
                self.sols.append(u)
                
            u.rename('u', 'u')
            file << (u, i)

        print("Total energy is", energy)
        self.pde = pde
        self.pde.u = u

        if return_all:
            return self.energy_density, self.force, self.sols

        return self.energy_density

    def _random_data(self):

        self.def_grad = np.random.uniform(-0.1, 0.1, 4)
        # shear: self.def_grad = np.random.normal(0, 0.5, 4)
        self.void_shape = np.random.uniform(-0.3, 0, 2)
        self.void_shape[1] = self.void_shape[1] + 0.1

        self.def_grad = np.array([0, 0.3, 0, -0.3])  # (4,) numpy array
        # self.void_shape = np.array([-0.2, 0.2]) # (2,) numpy array
        self.void_shape = np.array([0, 0])  # (2,) numpy array
