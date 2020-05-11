
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dolfin as dl
import numpy as np
import time


def corner(x, on_boundary):
    return bool((dl.near(x[0], 0., 1e-6) and dl.near(x[1], 0.0, 1e-6)))


def corner2(x, on_boundary):
    return bool((dl.near(x[0], 1.0, 1e-6) and dl.near(x[1], 0., 1e-6)))


def top(x, on_boundary):
    return dl.near(x[1], W, 1e-6) and on_boundary


def bottom(x, on_boundary):
    return dl.near(x[1], 0.0, 1e-6) and on_boundary


def PK(F):
    E, nu = material_parameters
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = E / 3 / (1 - 2 * nu)

    F = dl.variable(F)
    C = F.T * F
    J = dl.det(F)
    Ic = dl.tr(C)

    psi = (mu / 2) * ((Ic + 1) * J**(-2.0 / 3.0) - 3) + kappa / 2 * (J - 1)**2

    return dl.diff(psi, F)


if __name__ == '__main__':

    dl.set_log_level(20)

    meshfile = "plots/new_data/mesh/DNS_size16_pore0"
    mesh = dl.Mesh(meshfile + ".xml")
    # subdomains = dl.MeshFunction("size_t", mesh, meshfile + "_physical_region.xml")

    E = 100.
    nu = 0.3

    W = mesh.coordinates()[:, 1].max()

    material_parameters = [E, nu]

    P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    V = dl.FunctionSpace(mesh, P2)

    u = dl.Function(V)
    v = dl.TestFunction(V)

    dx = dl.dx
    I = dl.Identity(2)
    F = I + dl.grad(u)

    Form = dl.inner(PK(F), dl.grad(v)) * dx

    deltaU = 0.0

    outfile = "nu-%.2f/" % (nu)

    uFile = dl.File(outfile + "displacement.pvd")

    uTop = dl.Constant(deltaU)

    # bcs = [ dl.DirichletBC(V.sub(0).sub(0), dl.Constant(0.), corner, method='pointwise'),
    #        dl.DirichletBC(V.sub(0).sub(1), 0.0, bottom),
    #        dl.DirichletBC(V.sub(0).sub(1), uTop, top) ]
    bcs = [dl.DirichletBC(V, (0.0, 0.0), bottom),
           dl.DirichletBC(V.sub(1), uTop, top),
           dl.DirichletBC(V.sub(0), 0.0, top)]

    deltas = []
    PKs = []
    PKForm = (PK(F))[1, 1] * dx  # + (PK(1, I, F, p, nFiber))[0][1, 1]*dx(1)

    # steps = np.concatenate((np.linspace(0, 0.06, 7), np.linspace(0.06, 0.1, 15)))

    for i in range(20):
        if deltaU < 0.06*W:
            deltaU += 0.01*W
        else:
            deltaU += 0.04 / 2

    # for i, s in enumerate(steps):
    #     deltaU = s * W

        print("\nStep {}".format(i))
        # print("Relative disp {}".format(-s))

        uTop.assign(-deltaU)

        try:
            dl.solve(Form == 0, u, bcs, solver_parameters={"newton_solver":
                                                           {"linear_solver": "mumps",
                                                            "maximum_iterations": 50}},
                     form_compiler_parameters={"optimize": True})
        except RuntimeError:
            break

        uFile << (u, deltaU)

        deltas.append(deltaU)
        PKs.append(dl.assemble(PKForm))
        plt.clf()
        # PK22 = sum([PK(i, F0, F) for i in range(nphases)])

        # p1 = dl.plot(pSol, cmap="jet")
        # plt.colorbar(p1)
        # plt.savefig(outfile + "p-%d.png" % i )


plt.clf()
deltas = np.array(deltas)
# eps = 1 + (deltas-1)/W
# plt.plot(deltas,  Er/2/(1+nur)*(eps - 1.0/eps**3))
plt.plot(-deltas / W, PKs / W / W, 'o')
plt.savefig("curve.png")
# np.save("PK1-1.npy", PKs)
