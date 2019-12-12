from .pde.static import Metamaterial
from . import arguments
import numpy as np
import fenics as fa
import matplotlib.pyplot as plt


def anealing_solver(data=1, anneal_steps=10):

    pde = Metamaterial(args)
    anneal_factors = np.linspace(0, 0.1, anneal_steps)
    guess = fa.Function(pde.V).vector()

    for i, factor in enumerate(anneal_factors):
        print("Now at step", i)
        weighted_data = data * factor

        boundary_fn = fa.Expression(('weighted_data*x[1]', 'weighted_data*x[0]'), weighted_data=weighted_data, degree=2)
        # boundary_fn = fa.Constant((1, 1))

        u = pde.solve_problem(boundary_fn=boundary_fn,
                              initial_guess=guess)
        guess = u.vector()

    return u, pde.energy(u)

def plot_f_val(n_cells_list, energy_list):
    # plt.figure()
    plt.plot(n_cells_list, energy_list, '*-', label='newly evaluated')
    plt.legend(loc='lower right')
    # plt.xlabel('number of samples')
    # plt.ylabel('f value')
    # axes = plt.gca()
    # axes.set_ylim(0, 0.3)
    # plt.show()

def energy_fit(n_cells):
    E = 5e-05
    C = 1e-4
    return E + C/n_cells

if __name__ == '__main__':
    args = arguments.args
    # u, energy = anealing_solver()
    # file = fa.File("u.pvd")
    # file << u

    # print(energy/(args.n_cells*args.n_cells))

    # exit()

    n_cells_list = []
    energy_list = []
    # plot_f_val(n_cells_list, energy_list)

    plt.figure()

    # n_cells_continuous = np.linspace(1, 10, 1000)
    # energy_continuous = energy_fit(n_cells_continuous)
    # plot_f_val(n_cells_continuous, energy_continuous)

    # n_cells_list = [3, 3, 4]
    # energy_list = [0.00012777641751276432, 9.241765337561646e-05, 7.811781575515235e-05]

    # for i in range(3):
    #     n_cells = i + 2
    #     args.n_cells = n_cells
    #     print("n_cells is", n_cells)
    #     n_cells_list.append(n_cells)
    #     u, energy = anealing_solver()
    #     energy_list.append(energy/(args.n_cells*args.n_cells))
    #     print(energy/(args.n_cells*args.n_cells), '\n\n\n\n\n\n\n')

    # # plot_f_val(n_cells_list, energy_list)

    # # print(energy_list)

    n_cells_list = [1/4, 1/5, 1/6, 1/7, 1/8, 1/9]
    energy_list = [7.811781575515235e-05, 
                   7.050917171685857e-05, 
                   6.590148453276008e-05, 
                   6.28545335151527e-05, 
                   6.07276671263034e-05,
                   5.91260180411421e-05]
    
    plot_f_val(n_cells_list, energy_list)

    plt.show()






