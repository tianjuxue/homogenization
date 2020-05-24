"""Metamaterial PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .pde import PDE
from .strain import NeoHookeanEnergy


class Metamaterial(PDE):
    def _build_mesh(self):
        """Create mesh with pores defined by c1, c2 a la Overvelde&Bertoldi"""
        args = self.args

        (L0, 
         porosity,
         c1,
         c2, 
         resolution, 
         n_cells, 
         min_feature_size, 
         pore_radial_resolution) = (args.L0, 
                                    args.porosity, 
                                    args.c1,
                                    args.c2,
                                    args.metamaterial_mesh_size, 
                                    args.n_cells,
                                    args.min_feature_size,
                                    args.pore_radial_resolution)

        material_domain = None
        pore_domain = None

        for i in range(n_cells):
            for j in range(n_cells):
                if args.gradient:
                    c1 = (j + 0.5) / n_cells * -0.2
                    c2 = (j + 0.5) / n_cells * 0.2

                r0 = L0 * math.sqrt(2 * porosity) / math.sqrt(math.pi *
                                                              (2 + c1**2 + c2**2))
                def coords_fn(theta):
                    return r0 * (1 + c1 * fa.cos(4 * theta) + c2 * fa.cos(8 * theta))

                base_pore_points, radii, thetas = build_base_pore(
                            coords_fn, pore_radial_resolution)

                pore = build_pore_polygon(
                    base_pore_points, offset=(L0 * (i + 0.5), L0 * (j + 0.5)))

                pore_domain = pore if not pore_domain else pore + pore_domain

                cell = mshr.Rectangle(
                    fa.Point(L0 * i, L0 * j),
                    fa.Point(L0 * (i + 1), L0 * (j + 1)))
                material_in_cell = cell - pore
                material_domain = (material_in_cell if not material_domain else
                                   material_in_cell + material_domain)

        if args.padding:
            for j in range(n_cells):
                # TODO (Tianju): Likely to have a bug here, but should be easy to fix
                # Given padding is not important anyway, I will leave it here
                base_pore_points, radii, thetas = build_base_pore(
                            coords_fn, pore_radial_resolution)

                pore = build_pore_polygon(
                    base_pore_points, offset=(L0 * (-0.5), L0 * (j + 0.5)))
                cell = mshr.Rectangle(
                    fa.Point(L0 * -0.5, L0 * j),
                    fa.Point(0, L0 * (j + 1)))
                material_in_cell = cell - pore
                material_domain += material_in_cell

                pore = build_pore_polygon(
                    base_pore_points, offset=(L0 * (n_cells + 0.5), L0 * (j + 0.5)))
                cell = mshr.Rectangle(
                    fa.Point(L0 * n_cells, L0 * j),
                    fa.Point(L0 * (n_cells + 0.5), L0 * (j + 1)))
                material_in_cell = cell - pore
                material_domain += material_in_cell              

        mesh = mshr.generate_mesh(material_domain, resolution * n_cells)
        self.mesh = mesh

        # print(mesh.num_cells())
        # print(mesh.num_vertices())
        # exit()


    def _build_function_space(self):
        """Create 2d VectorFunctionSpace and an exterior domain"""
        L0 = self.args.L0
        n_cells = self.args.n_cells
        fluctuation = self.args.fluctuation

        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], L0 * n_cells) or
                    fa.near(x[0], L0 * n_cells) or
                    fa.near(x[0], 0) or
                    fa.near(x[1], 0))

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], 0)

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], L0 * n_cells)

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], 0)

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], L0 * n_cells)

        class LowerLeft(fa.SubDomain):
            def inside(self, x, on_boundary):                    
                return fa.near(x[0], 0) and fa.near(x[1], 0)

        class LowerRight(fa.SubDomain):
            def inside(self, x, on_boundary):                    
                return fa.near(x[0], L0 * n_cells) and fa.near(x[1], 0)

        class UpperLeft(fa.SubDomain):
            def inside(self, x, on_boundary):                    
                return fa.near(x[0], 0) and fa.near(x[1], L0 * n_cells)

        class UpperRight(fa.SubDomain):
            def inside(self, x, on_boundary):                    
                return fa.near(x[0], L0 * n_cells) and fa.near(x[1], L0 * n_cells)            

        class PeriodicDomain(fa.SubDomain):

            def inside(self, x, on_boundary):
                is_lower_left = fa.near(x[0], 0) and fa.near(x[1], 0)
                is_lower_right = fa.near(x[0], L0 * n_cells) and fa.near(x[1], 0)
                is_upper_left = fa.near(x[0], 0) and fa.near(x[1], L0 * n_cells)
                is_upper_right = fa.near(x[0], L0 * n_cells) and fa.near(x[1], L0 * n_cells)
                is_corner = is_lower_left or is_lower_right or is_upper_right
                is_left = fa.near(x[0], 0)
                is_bottom = fa.near(x[1], 0)
                return (is_left or is_bottom) and (not is_corner)

            def map(self, x, y):
                if fa.near(x[0], L0 * n_cells):
                    y[0] = x[0] - L0 * n_cells
                    y[1] = x[1]
                elif fa.near(x[1], L0 * n_cells):
                    y[0] = x[0] 
                    y[1] = x[1] - L0 * n_cells
                else:
                    y[0] = 1000
                    y[1] = 1000
 
        self.corner_dic = [LowerLeft(), LowerRight(), UpperLeft(), UpperRight()]

        self.exteriors_dic = {'left': Left(), 'right': Right(), 'bottom': Bottom(), 'top': Top()}
        self.exterior = Exterior()

        if fluctuation:
            self.V = fa.VectorFunctionSpace(self.mesh, 'P', 1, constrained_domain=PeriodicDomain())
            self.V_non_periodic = fa.VectorFunctionSpace(self.mesh, 'P', 1)
        else:
            self.V = fa.VectorFunctionSpace(self.mesh, 'P', 1)

        self.sub_domains = fa.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.sub_domains.set_all(0)

        self.boundaries_id_dic = {'left': 1, 'right': 2, 'bottom': 3, 'top': 4}
        self.left = Left()
        self.left.mark(self.sub_domains, 1)
        self.right = Right()
        self.right.mark(self.sub_domains, 2)
        self.bottom = Bottom()
        self.bottom.mark(self.sub_domains, 3)
        self.top = Top()
        self.top.mark(self.sub_domains, 4)

        self.normal = fa.FacetNormal(self.mesh)
        self.ds = fa.Measure("ds")(subdomain_data=self.sub_domains)

    def _energy_density(self, u, return_stress=False):
        """Energy density is NeoHookean strain energy. See strain.py for def."""
        return NeoHookeanEnergy(u, self.args.young_modulus,
                                self.args.poisson_ratio, return_stress, 
                                self.args.fluctuation, self.args.F_list)

    def force(self, u):
        # fa.parameters['allow_extrapolation'] = True
        u = fa.interpolate(u, self.V)
        _, first_pk_stress = self._energy_density(u, True)
        stress_11 = fa.assemble(fa.dot(first_pk_stress, self.normal)[0]*self.ds(2))
        stress_12 = fa.assemble(fa.dot(first_pk_stress, self.normal)[1]*self.ds(2))
        stress_21 = fa.assemble(fa.dot(first_pk_stress, self.normal)[0]*self.ds(4))
        stress_22 = fa.assemble(fa.dot(first_pk_stress, self.normal)[1]*self.ds(4))    

        # Change log
        # stress_22 = fa.assemble(first_pk_stress[1, 1]*fa.dx)

        return np.array([[stress_11, stress_12], [stress_21, stress_22]])


''' Helper functions '''
def build_base_pore(coords_fn, n_points):
    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [
        coords_fn(float(i) * 2 * math.pi / n_points) for i in range(n_points)
    ]
    points = [(rtheta * np.cos(theta), rtheta * np.sin(theta))
              for rtheta, theta in zip(radii, thetas)]
    return np.array(points), np.array(radii), np.array(thetas)


def build_pore_polygon(base_pore_points, offset):
    points = [
        fa.Point(p[0] + offset[0], p[1] + offset[1]) for p in base_pore_points
    ]
    pore = mshr.Polygon(points)
    return pore

