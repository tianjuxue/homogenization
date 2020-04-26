import numpy as np
import os
from .generator import Generator, GeneratorDummy
from .. import arguments
import fenics as fa
import sobol_seq

def func():
    print(test_name)
    # test_name = test_name + 1

def save_fn(fname, def_grad, void_shape, predicted_energy, energy_density=None):

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
    if energy_density is None:
        to_save['data'].append([def_grad, void_shape, predicted_energy])
    else:
        to_save['data'].append([def_grad, void_shape, predicted_energy, energy_density])
    np.save(fname, to_save)

def find_tenary(num):
    quotient = num // 3
    remainder = num % 3
    if quotient == 0:
        return str(remainder)
    else:
        return find_tenary(quotient) + str(remainder)

def get_parameters_grid(num):
    tenary_code = find_tenary(num)
    complement = ""
    for _ in range(6 - len(tenary_code)):
        tenary_code = '0' + tenary_code 
    parameters = [(float(tenary_code[0]) - 1),
                  (float(tenary_code[1]) - 1),
                  (float(tenary_code[2]) - 1),
                  (float(tenary_code[3]) - 1),
                  -0.1*float(tenary_code[4]),
                  0.1*float(tenary_code[5])]
    return tenary_code, parameters


def run_trial_grid(tenary_code, enable_fast_solve, level, parameters):

    generator.enable_fast_solve = enable_fast_solve
    def_grad = 0.05*np.power(2, level)*np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])
    def_grad_all, void_shape_all, predicted_energy_all = generator.generate_data_single(def_grad, void_shape)
    for i, _ in enumerate(generator.anneal_factors):
        save_fn(tenary_code + '_anneal' + str(i) + '_level' + str(level), def_grad_all[i], void_shape_all[i], predicted_energy_all[i])

    print(def_grad_all)
    print(void_shape_all)
    print(predicted_energy_all)

    print(" success! enable_fast_solve = {} and level = {}".format(enable_fast_solve, level))
    # counter = counter + 1

def func():
    print(test_name)
    # test_name = test_name + 1

def collect_grid():
    counter = 0
    start = 0
    for i in range(start, np.power(3, 6)):   
    # for i in range(0, 1):   
        print("\n\n\n")
        tenary_code, parameters = get_parameters_grid(i)
        print("Simulation for round", i)
        print("tenary_code is", tenary_code)
        print("parameters is", parameters)

        solver_setting = [(True, 2 - i) if j == 0 else (False, 2 - i) for i in range(3) for j in range(2)]

        for setting in solver_setting:
            try:
                print(" try setting of enable_fast_solve = {} and level = {}".format(setting[0], setting[1]))
                run_trial_grid(tenary_code=tenary_code, enable_fast_solve=setting[0], level=setting[1], parameters=parameters)
                break
            except Exception as e:
                print(e)
                continue

        print("Total number of rounds: {} and successful number: {}".format(i - start + 1, counter))    


def get_parameters_sobol(sobol_vec):

    parameters = [1*(sobol_vec[0] - 0.5),
                  1*(sobol_vec[1] - 0.5),
                  1*(sobol_vec[2] - 0.5),
                  1*(sobol_vec[3] - 0.5),
                  0.4*(sobol_vec[4] - 1) + 0.2,
                  0.4*(sobol_vec[5] - 1) + 0.3]

    return parameters

def is_qualified(parameters):
    if parameters[4] + parameters[5] > 0.1:
        return False
    else:
        return True

def run_trial_sobol(enable_fast_solve, seed_number, parameters):

    generator.enable_fast_solve = enable_fast_solve
    def_grad = np.array(parameters[0:4])
    void_shape = np.array(parameters[4:6])
    def_grad_all, void_shape_all, predicted_energy_all = generator.generate_data_single(def_grad, void_shape)
    
    for i, _ in enumerate(generator.anneal_factors):
        save_fn('seed_number_' + str(seed_number) + '_anneal' + str(i), def_grad_all[i], void_shape_all[i], predicted_energy_all[i])

    print(def_grad_all)
    print(void_shape_all)
    print(predicted_energy_all)

    print(" success! enable_fast_solve = {}".format(enable_fast_solve))


def collect_sobol():
    counter = 0
    start = 0

    parameters_len = 6
    division = 3
    total_number_samples = np.power(division, parameters_len)

    vec = sobol_seq.i4_sobol_generate(parameters_len, total_number_samples)

    # for i in range(start, 3):    
    for i in range(start, total_number_samples): 
        print("\n\n\n")
        parameters = get_parameters_sobol(vec[i])

        if not is_qualified(parameters):
            continue

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
    generator.args.relaxation_parameter = 0.05
    generator.args.max_newton_iter = 10000
    generator.args.n_cells = 1
    generator.args.metamaterial_mesh_size = 15

    if not os.path.exists(PATH):
        os.mkdir(PATH)

    collect_sobol()
    # collect_grid()

    exit()
    def_grad = np.random.uniform(-0.1, 0.1, 4)
    void_shape = np.random.uniform(-0.2, 0, 2) 
    void_shape[1] = void_shape[1] + 0.2
    def_grad_all, void_shape_all, predicted_energy_all = generator.generate_data_series(def_grad, void_shape)
    for i, _ in enumerate(generator.anneal_factors):
        save_fn(def_grad_all[i], void_shape_all[i], predicted_energy_all[i], tenary_code + '_anneal' + str(i) + '_level' + level)


