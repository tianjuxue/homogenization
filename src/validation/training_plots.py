"""Collects data for gripper case"""

import numpy as np
import os
from ..collector.generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import matplotlib.pyplot as plt
import torch
from ..trainer.loader import Net
from ..trainer.loader import Trainer

def neohookean(F_fake, flag_pore_shape):

    I = torch.eye(2)
    F = F_fake + I
    C = torch.mm(torch.t(F), F)
    C_11 = C[0][0]
    C_12 = C[0][1]
    C_22 = C[1][1]

    if flag_pore_shape == 1:
        xi_1 = torch.tensor(0.)
        xi_2 = torch.tensor(0.)
    else:
        xi_1 = torch.tensor(-0.2)
        xi_2 = torch.tensor(0.2)

    input_tensor = torch.stack([C_11, C_12, C_22, xi_1, xi_2]) 

    return input_tensor

def true_energy(x_vals, flag_pore_shape, flag_normal_shear):

    if flag_pore_shape == 1:
        generator.void_shape = np.array([-0., 0.])
    else:
        generator.void_shape = np.array([-0.2, 0.2])

    x_val = x_vals[-1]

    if flag_normal_shear == 1:
        generator.args.F_list_fixed = [[0, 0], [0., x_val]]
        generator.def_grad = np.array([0, 0, 0, x_val])  
        energy_density, force = generator._anealing_solver_fluctuation(return_force=True)
        energy = energy_density
        stress = [f[1][1] for f in force]           
    else:
        generator.args.F_list_fixed = [[0, x_val], [0., 0]]
        generator.def_grad = np.array([0, x_val, 0, 0])
        energy_density, force = generator._anealing_solver_fluctuation(return_force=True)
        energy = energy_density
        stress = [f[0][1] for f in force] 

        # print(energy_density)
        # print(generator.probe_all)        
        # print(force)

    energy = np.asarray(energy)
    stress = np.asarray(stress)

    return energy, stress

def nn_energy(x_vals, flag_pore_shape, flag_normal_shear):

    energy = []
    stress = []

    for x_val in x_vals:

        if flag_normal_shear == 1:
            F = torch.tensor([[0, 0.], [0., x_val]], requires_grad=True)
            input_tensor = neohookean(F, flag_pore_shape)
            output_tensor = network(input_tensor)
            output_tensor.backward()
            force = F.grad.data.numpy()
            energy.append(output_tensor.data.numpy())
            stress.append(force[1][1])
        else:
            F = torch.tensor([[0, x_val], [0., 0]], requires_grad=True)
            input_tensor = neohookean(F, flag_pore_shape)
            output_tensor = network(input_tensor)
            output_tensor.backward()
            force = F.grad.data.numpy()
            energy.append(output_tensor.data.numpy())
            stress.append(force[0][1])

    energy = np.asarray(energy)
    energy = energy - energy[0]
    stress = np.asarray(stress)

    # print("nn_energy is")
    # print(energy)

    return energy, stress


def plot_energy(x_vals, energy_nn, energy_true):
    plt.plot(x_vals, energy_nn, ':', color='b',label='energy_nn_p1_normal') 
    plt.plot(x_vals, energy_true, '-', color='b', label='energy_p1_normal')          


if __name__ == '__main__':


    args = arguments.args
    PATH = args.data_path_integrated_regular
    generator = Generator(args)

    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 2000
    generator.enable_fast_solve = True
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = True
    generator.void_shape = np.array([-0., 0.]) 
    generator.anneal_factors = np.linspace(0, 1, 11)

    # generator.args.F_list_fixed = [[0., 0.1], [0.1, 0.125]]
    # generator.def_grad = np.array([0, 0, 0, -0.125])
    # energy_density, force = generator._anealing_solver_fluctuation(return_force=True)
    # print(energy_density)
    # print(force)
    # exit()


    # model_path = 'saved_checkpoints_tmp/normal'
    model_path = args.checkpoints_path_shear + '/model_step_' + str(1000*49)
    network =  torch.load(model_path)

   
    flag_pore_shape_list = [1, 2]
    x_vals_list = [np.linspace(0, 0.2, 11), np.linspace(0, -0.2, 11), np.linspace(0, 0.2, 11)]
    flag_normal_shear_list = [1, 1, 2]

    energy_nn_list = []
    stress_nn_list = []
    energy_true_list = []
    stress_true_list = []


    normal_shear_label = ['tensile', 'compressive', 'shear']
    energy_stress_label = ['energy', 'stress']
    pore_label = ['p1', 'p2']  
    pore_color = ['b', 'r']
 

    for i, flag_normal_shear in enumerate(flag_normal_shear_list):
        energy_nn_sub_list = []
        energy_true_sub_list = []
        stress_nn_sub_list = []
        stress_true_sub_list = []

        for j, flag_pore_shape in enumerate(flag_pore_shape_list):
            energy_nn, stress_nn = nn_energy(x_vals=x_vals_list[i], 
                                             flag_pore_shape=flag_pore_shape, 
                                             flag_normal_shear=flag_normal_shear)
            energy_true, stress_true = true_energy(x_vals=x_vals_list[i], 
                                                   flag_pore_shape=flag_pore_shape, 
                                                   flag_normal_shear=flag_normal_shear) 
            energy_nn_sub_list.append(energy_nn)  
            energy_true_sub_list.append(energy_true)      
            stress_nn_sub_list.append(stress_nn)  
            stress_true_sub_list.append(stress_true)  

        energy_nn_list.append(energy_nn_sub_list)
        energy_true_list.append(energy_true_sub_list)
        stress_nn_list.append(stress_nn_sub_list)
        stress_true_list.append(stress_true_sub_list)


    fig, axes = plt.subplots(nrows=2, ncols=3)

    for i, normal_shear in enumerate(normal_shear_label):
        for j, energy_stress in enumerate(energy_stress_label):
            ax = axes[j, i]
            if energy_stress == 'energy':
                ax.title.set_text(normal_shear)
            ax.set(xlabel='strain', ylabel=energy_stress)
            if energy_stress == 'energy':
                for k, pore in enumerate(pore_label):
                    ax.plot(x_vals_list[i], energy_true_list[i][k], '-', color=pore_color[k], label='DNS '+ pore) 
                    ax.plot(x_vals_list[i], energy_nn_list[i][k], ':', color=pore_color[k], label='NN '+ pore)             
            else:
                for k, pore in enumerate(pore_label):
                    ax.plot(x_vals_list[i], stress_true_list[i][k], '-', color=pore_color[k], label='DNS '+ pore) 
                    ax.plot(x_vals_list[i], stress_nn_list[i][k], ':', color=pore_color[k], label='NN '+ pore)              

            if energy_stress == 'energy' and i == 1:
                ax.legend(loc='upper right')
            elif energy_stress == 'stress' and i == 1:
                ax.legend(loc='lower right')
            else:
                ax.legend(loc='upper left')


    plt.show()
    # fig.savefig("energy.pdf", bbox_inches='tight')
    exit()
  
