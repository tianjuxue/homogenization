"""Collects data for gripper case"""

import numpy as np
import os
from .generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import sobol_seq
import sys

def func():
    print(test_name)
    # test_name = test_name + 1

def save_fn(fname, def_grad, void_shape, predicted_energy, probe=None, energy_density=None):

    # # Find a unique filename
    # while (True):
    #     fname = os.path.join(PATH,
    #                          str(np.random.randint(999999999)) + '.npy')
    #     if not os.path.exists(fname):
    #         break

    git_num = os.popen('git rev-parse HEAD').read().split('\n')[0]

    fname = os.path.join(PATH, fname + '.npy')

    if os.path.exists(fname):
        to_save = np.load(fname).any()
    else:
        to_save = {
            'git_id': git_num,
            'args': args,
            'data': [['def_grad', 'void_shape', 'predicted_energy', 'energy_density']]
        }


    if probe is None:
        to_save['data'].append([def_grad, void_shape, predicted_energy])
    else:
        to_save['data'].append([def_grad, void_shape, predicted_energy, probe])

    # if energy_density is None:
    #     to_save['data'].append([def_grad, void_shape, predicted_energy])
    # else:
    #     to_save['data'].append([def_grad, void_shape, predicted_energy, energy_density])
    np.save(fname, to_save)


def get_parameters_sobol(sobol_vec):

    parameters = [0.5*(sobol_vec[0] - 0.5),
                  1.6*(sobol_vec[1] - 0.5),
                  1.6*(sobol_vec[2] - 0.5),
                  0.5*(sobol_vec[3] - 0.5),
                  -0.2,
                  0.2]

    return parameters


def run_trial_sobol(enable_fast_solve, seed_number, parameters):

    generator.enable_fast_solve = enable_fast_solve
    generator.args.F_list_fixed = [[parameters[0], parameters[1]], [parameters[2], parameters[3]]]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])


    if not enable_fast_solve:
        try:
            generator.generate_data_single(def_grad, void_shape)
            predicted_energy_all = generator.energy_density
            def_grad_all =  generator.def_grad_all
            void_shape_all = generator.void_shape_all
            probe_all = generator.probe_all
        except:
            predicted_energy_all = generator.energy_density
            def_grad_all =  generator.def_grad_all
            void_shape_all = generator.void_shape_all
            probe_all = generator.probe_all
    else:
        generator.generate_data_single(def_grad, void_shape)
        predicted_energy_all = generator.energy_density
        def_grad_all =  generator.def_grad_all
        void_shape_all = generator.void_shape_all
        probe_all = generator.probe_all        

    for i, _ in enumerate(predicted_energy_all):
        save_fn('seed_number_' + str(seed_number) + '_anneal' + str(i), 
                def_grad_all[i], 
                void_shape_all[i], 
                predicted_energy_all[i],
                probe=probe_all[i])

    print(def_grad_all)
    print(void_shape_all)
    print(predicted_energy_all)
    print(probe_all)

    print(" success! enable_fast_solve = {}".format(enable_fast_solve))

def generate_vec(division_array):
    # return numpy array (N, 4)
    interval = 1./ (division_array - 1)
    vec = []

    for index in range(4):
        for i in range(division_array[(index + 1) % 4]):
            for j in range(division_array[(index + 2) % 4]):
                for k in range(division_array[(index + 3) % 4]):
                    sub_vec = np.zeros(4)
                    sub_vec[index] = 0
                    sub_vec[(index + 1) % 4] = i * interval[(index + 1) % 4]
                    sub_vec[(index + 2) % 4] = j * interval[(index + 2) % 4]
                    sub_vec[(index + 3) % 4] = k * interval[(index + 3) % 4]
                    vec.append(sub_vec)
                    sub_vec = np.zeros(4)
                    sub_vec[index] = 1
                    sub_vec[(index + 1) % 4] = i * interval[(index + 1) % 4]
                    sub_vec[(index + 2) % 4] = j * interval[(index + 2) % 4]
                    sub_vec[(index + 3) % 4] = k * interval[(index + 3) % 4]
                    vec.append(sub_vec)                    

    vec = np.asarray(vec)

    # # only loop of 2

    # vec = np.zeros((total_number_samples, parameters_len))
    # for i in range(division):
    #     for j in range(division):
    #         vec[division*i + j] = np.array([interval*i, interval*j])

    return vec

def collect_sobol():
    counter = 0
    start = 0
    parameters_len = 2
    division = 4
    total_number_samples = np.power(division, parameters_len)

    division_array = np.array([3, 3, 3, 3])

    vec = generate_vec(division_array)
    total_number_samples = len(vec)

    print("total_number_samples is", total_number_samples)
    # np.set_printoptions(threshold=sys.maxsize)
    # vec = sobol_seq.i4_sobol_generate(parameters_len, total_number_samples)

    # for i in range(start, 3):    
    for i in range(start, total_number_samples): 
        print("\n\n\n")
        parameters = get_parameters_sobol(vec[i])

        print("Simulation for round", i)
        print("parameters is", parameters)

        solver_setting = [True, False]

        for setting in solver_setting:
            try:
                print(" try setting of enable_fast_solve = {}".format(setting))
                run_trial_sobol(enable_fast_solve=setting, seed_number=i, parameters=parameters)
                break
            except Exception as e:
                print(e)
                continue

if __name__ == '__main__':

    # test_name = 0
    # func()

    args = arguments.args
    PATH = args.data_path_integrated_regular
    generator = Generator(args)
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 1000
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 11)

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    collect_sobol()



