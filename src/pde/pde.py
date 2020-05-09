"""Base class for PDEs"""
import fenics as fa
import time
from .dr_solve import DynamicRelaxSolve


class PDE(object):
    """Base class for PDEs.
    """

    def __init__(self, args, mesh=None):
        self.args = args
        # self.args.fluctuation = False
        # self.args.F_list = None
        if mesh is None:
            start = time.time()
            self._build_mesh()
            end = time.time()
            self.time_elapsed = end - start 
        else:
            self.mesh = mesh
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
                boundary_bc = fa.DirichletBC(self.V, 
                                             boundary_point_fn, 
                                             corner, 
                                             method='pointwise')
                bcs = bcs + [boundary_bc]

        # If boundary functions are defined separately for four edges
        if boundary_fn_dic is not None:
            for key in boundary_fn_dic:
                boundary_bc = fa.DirichletBC(self.V, boundary_fn_dic[key], self.exteriors_dic[key])
                bcs = bcs + [boundary_bc]

        bcs_homo = []
        if boundary_fn_dic is not None:
            for key in boundary_fn_dic:
                boundary_bc = fa.DirichletBC(self.V, fa.Constant((0, 0)), self.exteriors_dic[key])
                bcs_homo = bcs_homo + [boundary_bc]

        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du)

        newton_args = {
            'relaxation_parameter': self.args.relaxation_parameter,
            'linear_solver': self.args.linear_solver,
            'maximum_iterations': self.args.max_newton_iter,
            'relative_tolerance': 1e-5,
            'absolute_tolerance': 1e-5
        }

        solver_args = {
            'nonlinear_solver': self.args.nonlinear_solver,
            'newton_solver': newton_args
        }


        solver_args_snes = \
        {
            "nonlinear_solver": "snes",
            "snes_solver": 
            {
                "linear_solver": "umfpack",
                "maximum_iterations": self.args.max_snes_iter,
                "report": True,
                "error_on_nonconvergence": False,
                "line_search":"basic",
                "relative_tolerance":1.e-8,
                "absolute_tolerance":1.e-8,
                "preconditioner" : "default",
                "krylov_solver":
                {
                    "report":True,
                    "nonzero_initial_guess" : False
                }
            },
        }


        fa.parameters["form_compiler"]["cpp_optimize"] = True

        ffc_options = {"optimize": True, \
                       "eliminate_zeros": True, \
                       "precompute_basis_const": True, \
                       "precompute_ip_const": True}

        if enable_fast_solve:
            if enable_dynamic_solve:
                nIters, convergence = DynamicRelaxSolve(dE, u, bcs, jacE)
                if not convergence:
                    assert(False)
            else:
                fa.solve(dE == 0, u, bcs, J=jacE,
                        form_compiler_parameters=ffc_options)                
        else:
            fa.solve(dE == 0, u, bcs, J=jacE,
                     form_compiler_parameters=ffc_options, 
                     solver_parameters=solver_args)

        return u
