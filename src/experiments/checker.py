import numpy as np
import glob
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
    X_new = X_new[:, [0, 1, 2, 4, 5]]
    return X_new


def load_data_single(file_path):
    files = glob.glob(os.path.join(file_path, '*.npy'))
    X_vec = []
    y_vec = []

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

    result = np.concatenate((X_input[:, :-2], y_input.reshape(-1, 1)), axis=1)
    return result


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


if __name__ == '__main__':
    args = arguments.args
    result_old = load_data_single('saved_data_shear')
    result_new = load_data_single('saved_data_shear_cmp')

    result_old = np.asarray(sorted(result_old, key=functools.cmp_to_key(compare)))
    result_new = np.asarray(sorted(result_new, key=functools.cmp_to_key(compare)))

    print(result_old.shape)
    print(result_new.shape)
    print(result_old[:20])
    print('\n')
    print(result_new[:20])
