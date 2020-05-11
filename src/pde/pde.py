"""Base class for PDEs"""
import fenics as fa
import time
from .dr_solve_homo import DynamicRelaxSolve
from .dr_solve_inhomo import dynamic_relaxation_solve


class PDE(object):
    """Base class for PDEs.
    """

    def __init__(self, args, mesh=None):
        self.args = args
        # self.args.fluctuation = False
        # self.args.F_list = None
        start = time.time()
        if mesh is None:
            self._build_mesh()
        else:
            self.mesh = mesh
        end = time.time()
        self.time_elapsed = end - start

        # TODO(Tianju): Change these member variables
        self._build_function_space()
        self._create_boundary_measure()

    def _create_boundary_measure(self):
        exterior_domain = fa.MeshFunction("size_t",
                                          self.mesh,
                                          self.mesh.topology().dim() - 1)
        exterior_domain.set_all(0)
        self.exterior.mark(exterior_domain, 1)
        self.boundary_ds = fa.Measure("ds")(subdomain_data=exterior_domain)(1)

    def _build_mesh(self):
        raise NotImplementedError()

    def _build_function_space(self):
        raise NotImplementedError()

    def _energy_density(self, u):
        raise NotImplementedError()

    def energy(self, u):
        return fa.assemble(self._energy_density(u) * fa.dx)

    def solve_problem(self,
                      boundary_fn=None,
                      boundary_point_fn=None,
                      boundary_fn_dic=None,
                      initial_guess=None,
                      enable_fast_solve=False,
                      enable_dynamic_solve=False):

        u = fa.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)

        if initial_guess is not None:
            # We shouldn't need this assert, but Fenics lets you pass iterables
            # of the wrong size, and just sets the elems that match, which
            # blows up solve if you pass an initial guess vector in the wrong
            # basis (as it sets the wrong elements)
            assert len(initial_guess) == len(u.vector())
            u.vector().set_local(initial_guess)

        E = self._energy_density(u) * fa.dx

        bcs = []

        # If boundary functions are defined using one global function
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            bcs = bcs + [boundary_bc]

        if boundary_point_fn is not None:
            for corner in self.corner_dic:
                # for corner in self.points_dic:
                boundary_bc = fa.DirichletBC(self.V,
                                             boundary_point_fn,
                                             corner,
                                             method='pointwise')
                bcs = bcs + [boundary_bc]

        # If boundary functions are defined separately for four edges
        if boundary_fn_dic is not None:
            for key in boundary_fn_dic:
                boundary_bc = fa.DirichletBC(self.V, boundary_fn_dic[
                                             key], self.exteriors_dic[key])
                bcs = bcs + [boundary_bc]

        bcs_homo = []
        if boundary_fn_dic is not None:
            for key in boundary_fn_dic:
                boundary_bc = fa.DirichletBC(
                    self.V, fa.Constant((0, 0)), self.exteriors_dic[key])
                bcs_homo = bcs_homo + [boundary_bc]

        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du)

        fa.parameters["form_compiler"]["cpp_optimize"] = True
        ffc_options = {"optimize": True,
                       "eliminate_zeros": True,
                       "precompute_basis_const": True,
                       "precompute_ip_const": True}

        if enable_fast_solve:
            if enable_dynamic_solve:
                if self.args.fluctuation:
                    nIters, convergence = DynamicRelaxSolve(dE, u, bcs, jacE)
                else:
                    nIters, convergence = dynamic_relaxation_solve(dE, u, jacE, bcs, bcs_homo)
                if not convergence:
                    assert(False)
            else:
                fa.solve(dE == 0, u, bcs, J=jacE,
                         form_compiler_parameters=ffc_options)
        else:
            fa.solve(dE == 0, u, bcs, J=jacE,
                     form_compiler_parameters=ffc_options,
                     solver_parameters={'newton_solver': {'relaxation_parameter': self.args.relaxation_parameter,
                                         'maximum_iterations': self.args.max_newton_iter}})
        return u
