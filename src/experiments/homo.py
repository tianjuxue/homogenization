'''这个文件是一个独立的应用
'''
import fenics as fa
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ..trainer.loader import Net
from .. import arguments

# computational domain: 10*10
# K = 10

K = 10
N_CELLS = K**2
N = K + 1
N_DOFS = N**2
N_DOFS_TOTAL = 2 * N_DOFS
DIM = 2
DOFS_PER_CELL = 4
SIDE_LEN = 10
CELL_EDGE_LEN = SIDE_LEN / K
CELL_AREA = CELL_EDGE_LEN**2

def visualize(solution):
    # solution: np array of shape (N,) with N = 2 * (K + 1)**2

    X = generate_X()
    solution = solution + X
    solution = solution.reshape(DIM, -1)
    plt.figure()
    for cell_id in range(N_CELLS):
        points = get_cell_points(cell_id, solution)
        points = transfer_points(points)
        plt.plot(points[0], points[1], marker ='o', markersize=1, linewidth=0.5, color='b')

    plt.axis('equal')
    plt.show()

def transfer_points(points):
    points_t = points.copy()
    points_t[:, -1] = points[:, -2]
    points_t[:, -2] = points[:, -1]
    points_t = np.concatenate((points_t, points[:, [0]]), axis=1)
    return points_t

def get_cell_points(cell_id, solution):
    dof_ids = get_cell_dof_ids(cell_id)
    points = np.array([[solution[0][dof_id] for dof_id in dof_ids],
                       [solution[1][dof_id] for dof_id in dof_ids]])
    return points

def get_cell_dof_ids(cell_id):
    dof_id1 = cell_id // K + cell_id
    dof_id2 = dof_id1 + 1
    dof_id3 = dof_id1 + N
    dof_id4 = dof_id3 + 1
    return [dof_id1, dof_id2, dof_id3, dof_id4]

def generate_X():
    X1 = np.linspace(0, SIDE_LEN, N)
    X2 = np.linspace(0, SIDE_LEN, N)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.flatten()
    X2 = X2.flatten()
    X = np.concatenate((X1, X2))
    return X

def neohookean(F_list):

    row_1 = torch.stack([F_list[0], F_list[1]])
    row_2 = torch.stack([F_list[2], F_list[3]])
    F_fake = torch.stack([row_1, row_2])
    I = torch.eye(DIM)
    F = F_fake + I
    #TODO(Tianju): Bug to fix
    C = torch.mm(torch.t(F), F)
    J = torch.det(F)
    shear_mod = args.young_modulus / (2 * (1 + args.poisson_ratio))
    bulk_mod = args.young_modulus / (3 * (1 - 2*args.poisson_ratio))

    Jinv = J**(-2 / DIM)
    I1 = torch.trace(C)

    energy = (shear_mod / 2) * (Jinv * I1 - DIM) + (bulk_mod / 2) * (J - 1)**2

    return energy

def compute_energy(u_points, requires_grad, print_out=False):
    # u_00 = torch.tensor(u_points[0][0], requires_grad=requires_grad)

    u = [[torch.tensor(u_points[i][j], requires_grad=requires_grad) 
            for j in range(DOFS_PER_CELL)]
                for i in range(DIM)]

    F00 = 1/(2*CELL_EDGE_LEN)*(-u[0][0]+u[0][1]-u[0][2]+u[0][3])
    F01 = 1/(2*CELL_EDGE_LEN)*(-u[0][0]-u[0][1]+u[0][2]+u[0][3])
    F10 = 1/(2*CELL_EDGE_LEN)*(-u[1][0]+u[1][1]-u[1][2]+u[1][3])
    F11 = 1/(2*CELL_EDGE_LEN)*(-u[1][0]-u[1][1]+u[1][2]+u[1][3])

    F_list = [F00, F01, F10, F11]
    net_output = neohookean(F_list)

    # # Warning: def_grad is actually F - I
    # def_grad = torch.stack(F_list)
    # void_shape = torch.tensor([-0.2, 0.2])
    # net_input = torch.cat((def_grad, void_shape))
    # net_output = network(net_input)

    energy = net_output*CELL_AREA

    if requires_grad:
        energy.backward()
        energy_gradient = np.zeros_like(u_points)
        for i in range(DIM):
            for j in range(DOFS_PER_CELL):
                energy_gradient[i][j] = u[i][j].grad
        return energy_gradient
    else:
        # Check scalar
        if print_out:
            # print(def_grad.data.numpy())
            energy_val = energy.data.numpy()
            if energy_val < 0:
                print(energy_val)
                print(F_list)
                exit()
        energy = energy.data.numpy()
        return energy

def cell_energy_gradient(cell_id, solution):
    solution = solution.reshape(DIM, -1)
    u_points = get_cell_points(cell_id, solution)
    return compute_energy(u_points, True)

def cell_energy(cell_id, solution, flag=False):
    solution = solution.reshape(DIM, -1)
    u_points = get_cell_points(cell_id, solution)
    return compute_energy(u_points, False, flag)

def add_energy_gradient(energy_gradient, energy_gradient_cell, cell_id):
    dof_ids = get_cell_dof_ids(cell_id)
    energy_gradient = energy_gradient.reshape(DIM, -1)
    assert(energy_gradient[:, dof_ids].shape == energy_gradient_cell.shape)
    energy_gradient[:, dof_ids] = energy_gradient[:, dof_ids] + energy_gradient_cell
    return energy_gradient

def objective(x):
    energy = 0.
    for cell_id in range(N_CELLS):
        energy = energy + cell_energy(cell_id, x)
    print(energy)
    return energy

def objective_der(x):
    energy_gradient = np.zeros_like(x)
    for cell_id in range(N_CELLS):
        energy_gradient_cell = cell_energy_gradient(cell_id, x)
        energy_gradient = add_energy_gradient(energy_gradient, energy_gradient_cell, cell_id)

    energy_gradient = energy_gradient.flatten()

    # print(energy_gradient)

    return energy_gradient

def impose_strong_bcs(x):
    x = x.reshape(DIM, -1)
    x[0][:N] = 0
    x[0][-N:] = 0
    x[1][:N] = 0
    x[1][-N:] = -CELL_EDGE_LEN*0.8
    x = x.flatten()
    return x

def call_back(x):
    # for cell_id in range(N_CELLS):
    #     print("cell is is {}".format(cell_id))
    #     energy_cell = cell_energy(cell_id, x, True)
    #     print("\n")

    impose_strong_bcs(x)

    # visualize(x)

    global Nfeval 
    Nfeval = Nfeval + 1
    if Nfeval % 10 == 0:
        print("\n\n\nStep", Nfeval)
        
Nfeval = 1

if __name__ == '__main__':
    args = arguments.args
    STEP = 0
    model_path = args.checkpoints_path_dummy + '/model_step_' + str(100*99)
    network =  torch.load(model_path)

    test_input = [torch.tensor(0.0248), torch.tensor(0.1413),
                  torch.tensor(-0.0607), torch.tensor(-0.0468)]  
                      
    test_output = neohookean(test_input)

    # exit()

    X = generate_X()
    solution_X = X.flatten()
    u_initial = np.zeros_like(solution_X)
    u_initial = impose_strong_bcs(u_initial)

    options={'xtol': 1e-15, 'eps': 1e-15, 'maxiter': 1000, 'disp': True, 'return_all': False}

    res = opt.minimize(objective,
                       u_initial, 
                       method='CG', 
                       jac=objective_der,
                       callback=call_back,
                       options=None)

    # CG > BFGS > Newton-CG

    # res = opt.minimize(objective,
    #                    u_initial, 
    #                    method='Nelder-Mead', 
    #                    callback=call_back,
    #                    options=None)


    visualize(res.x)

    exit()

    obj = objective(u_initial)
    obj_grad = objective_der(u_initial)

    print("obj is", obj)
    print("obj_grad is", obj_grad)

    print(np.where(np.abs(obj_grad) > 0))


    # test_t1 = torch.tensor(3., requires_grad=True)
    # test_t2 = torch.tensor(4., requires_grad=True)
    # test_t3 = test_t1**2
    # test_t4 = test_t2

    # test_t5 = torch.stack((test_t3, test_t4))
    # test_sum = test_t5.sum()

    # test_result = torch.autograd.grad(test_sum, test_t1, create_graph=True,
    #                                   retain_graph=True)

    # print(test_result)

    # exit()








