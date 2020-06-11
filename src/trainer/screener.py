import glob
import numpy as np
import os
import functools
from .. import arguments


def get_features_C(Xin):
    Xin[:, 0] = Xin[:, 0] + 1
    Xin[:, 3] = Xin[:, 3] + 1
    X_new = Xin.copy()
    X_new[:, 0] = Xin[:, 0] * Xin[:, 0] + Xin[:, 2] * Xin[:, 2]
    X_new[:, 1] = Xin[:, 0] * Xin[:, 1] + Xin[:, 2] * Xin[:, 3]
    X_new[:, 2] = Xin[:, 1] * Xin[:, 1] + Xin[:, 3] * Xin[:, 3]
    # X_new = X_new[:, [0, 1, 2, 4, 5]]
    X_new = X_new[:, [0, 1, 2, 4]]
    return X_new


def load_data_all(args, rm_dup=False, middle=False):

    DATA_PATH_normal = 'saved_data_normal'
    DATA_PATH_middle = 'saved_data_middle'
    DATA_PATH_shear = 'saved_data_shear'

    # DATA_PATH_normal = 'saved_data_pore0'
    # DATA_PATH_middle = 'saved_data_pore1'
    # DATA_PATH_shear = 'saved_data_pore2'

    DATA_PATH_normal = 'saved_data_pore0_sobol_dr'
    DATA_PATH_shear = 'saved_data_pore2_sobol_dr'

    Xin_shear, Xout_shear = load_data_single(DATA_PATH_shear, rm_dup)
    Xin_normal, Xout_normal = load_data_single(DATA_PATH_normal, rm_dup)

    if middle:
        Xin_middle, Xout_middle = load_data_single(DATA_PATH_middle, rm_dup)
        Xin = np.concatenate((Xin_shear, Xin_normal, Xin_middle))
        Xout = np.concatenate((Xout_shear, Xout_normal, Xout_middle))
    else:
        Xin = np.concatenate((Xin_shear, Xin_normal))
        Xout = np.concatenate((Xout_shear, Xout_normal))
 
    args.input_dim = Xin.shape[1]

    print("\nTotal number of samples:", len(Xin))
    np.save('saved_data_sobol/Xin_H.npy', Xin)
    np.save('saved_data_sobol/Xout_H.npy', Xout)

    return Xin, Xout


def load_data_single(file_path, rm_dup):
    files = glob.glob(os.path.join(file_path, '*.npy'))
    X_vec = []
    y_vec = []
    print("\nprocessing", file_path)
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print("processed ", i, " files")
        data = np.load(f).item()['data'][1]
        X = np.concatenate((data[0], data[1]))
        y = data[2]
        X_vec.append(X)
        y_vec.append(y)

    X_input = np.asarray(X_vec)
    y_input = np.asarray(y_vec)
    X_input = get_features_C(X_input)
    XY = np.concatenate((X_input, y_input.reshape(-1, 1)), axis=1)

    if rm_dup:
        XY = np.asarray(sorted(XY, key=functools.cmp_to_key(compare)))
        XY = remove_dup(XY)

    return XY[:, :-1], XY[:, -1]


def compare(x, y):
    if np.isclose(x[0], y[0]):
        if np.isclose(x[1], y[1]):
            if np.isclose(x[2], y[2]):
                return 0
            elif x[2] < y[2]:
                return 1
            else:
                return -1
        elif x[1] < y[1]:
            return 1
        else:
            return -1
    elif x[0] < y[0]:
        return 1
    else:
        return -1


def close(x, y):
    return np.isclose(x[0], y[0]) and np.isclose(x[1], y[1]) and np.isclose(x[2], y[2])


def remove_dup(data):
    result = []
    i = 0
    j = 0
    while j < len(data):
        j += 1
        flag = False
        if (j < len(data)):
            if (not close(data[j], data[i])):
                flag = True
        else:
            flag = True
        if flag:
            result.append(np.average(data[i:j], axis=0))
            i = j
    return np.asarray(result)

if __name__ == '__main__':
    args = arguments.args
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
