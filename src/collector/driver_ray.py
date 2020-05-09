import numpy as np
import os
from .generator import Generator
from .. import arguments
import fenics as fa
import sobol_seq
import sys


def save_fn(args, fname, path, def_grad, void_shape, predicted_energy):
    to_save = {
        'args': args,
        'data': [['def_grad', 'void_shape', 'predicted_energy']]
    }
    to_save['data'].append([def_grad, void_shape, predicted_energy])
    np.save(path + '/' + fname + '.npy', to_save)


def get_parameters_pore0(vec):
    parameters = [0.8 * (vec[0] - 0.5),
                  0.4 * (vec[1] - 0.5),
                  0.4 * (vec[2] - 0.5),
                  0.8 * (vec[3] - 0.5),
                  -0.,
                  0.]
    return parameters


def get_parameters_pore1(vec):
    parameters = [0.6 * (vec[0] - 0.5),
                  1. * (vec[1] - 0.5),
                  1. * (vec[2] - 0.5),
                  0.6 * (vec[3] - 0.5),
                  -0.1,
                  0.1]
    return parameters


def get_parameters_pore2(vec):
    parameters = [0.5 * (vec[0] - 0.5),
                  1.6 * (vec[1] - 0.5),
                  1.6 * (vec[2] - 0.5),
                  0.5 * (vec[3] - 0.5),
                  -0.2,
                  0.2]
    return parameters


def simulate(generator, def_grad, void_shape, solve=True):
    if solve:
        generator.generate_data_single(def_grad, void_shape)
    def_grad_all = generator.def_grad_all
    void_shape_all = generator.void_shape_all
    predicted_energy_all = generator.energy_density
    return def_grad_all, void_shape_all, predicted_energy_all


def run_trial_ray(path, index, parameters):
    generator.args.F_list_fixed = [
        [parameters[0], parameters[1]], [parameters[2], parameters[3]]]
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])

    generator.enable_fast_solve = True
    try:
        print("Try fast solve...")
        def_grad_all, void_shape_all, predicted_energy_all = simulate(
            generator, def_grad, void_shape)
    except Exception as e:
        # print(e)
        print("Fast solve fails, start slow solve...")
        generator.enable_fast_solve = False
        try:
            def_grad_all, void_shape_all, predicted_energy_all = simulate(
                generator, def_grad, void_shape)
        except Exception as e:
            # print(e)
            print("Slow solve fails, save obtained results.")
            def_grad_all, void_shape_all, predicted_energy_all = simulate(
                generator, def_grad, void_shape, False)

    for i in range(len(predicted_energy_all)):
        save_fn(generator.args,
               'number_' + str(index) + '_anneal' + str(i),
                path,
                def_grad_all[i],
                void_shape_all[i],
                predicted_energy_all[i])

    if (len(predicted_energy_all) > 0):
        print("Successfully saved these data")
    else:
        print("Fail to save any data for this configuration.")

    print("\ndef_grad_all", def_grad_all)
    print("\nvoid_shape_all", void_shape_all)
    print("\npredicted_energy_all", predicted_energy_all)


def generate_vec(division_array):
    # return numpy array (N, 4)
    interval = 1. / (division_array - 1)
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
    return vec


def collect_ray(path, get_parameters):
    if not os.path.exists(path):
        os.mkdir(path)

    start = 0
    division_array = np.array([3, 3, 3, 3])
    vec = generate_vec(division_array)
    total_number_samples = len(vec)
    print("total_number_samples is", total_number_samples)

    # for i in range(584, total_number_samples):
    for i in range(start, total_number_samples):
        print("\n#####################################################################")
        parameters = get_parameters(vec[i])
        print("Simulation for round", i)
        print("parameters is", parameters)
        run_trial_ray(path=path, index=i, parameters=parameters)


if __name__ == '__main__':
    args = arguments.args
    generator = Generator(args)
    # Original 0.1 and 1000
    generator.args.relaxation_parameter = 0.1
    generator.args.max_newton_iter = 1000
    generator.args.n_cells = 2
    generator.args.metamaterial_mesh_size = 15
    generator.args.fluctuation = True
    generator.anneal_factors = np.linspace(0, 1, 11)
    # collect_ray(path='saved_data_pore0', get_parameters=get_parameters_pore0)
    collect_ray(path='saved_data_middle', get_parameters=get_parameters_pore1)
    # collect_ray(path='saved_data_compare2', get_parameters=get_parameters_pore2)
