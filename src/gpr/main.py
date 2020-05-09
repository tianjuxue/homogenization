import numpy as np
from .plot_gp import plot_gp, plot_gp_2D
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
from ..trainer.screener import load_data_all
from .. import arguments
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def kernel_anisotropic(X1, X2, l, sigma_f):
    X1 = X1 / l
    X2 = X2 / l
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
        np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 * sqdist)


def posterior_predictive_anisotropic(X_s, X_train, Y_train, l, sigma_f, sigma_y):
    K = kernel_anisotropic(X_train, X_train, l, sigma_f) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = kernel_anisotropic(X_train, X_s, l, sigma_f)
    K_ss = kernel_anisotropic(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def ufl_pre_test(C_list, l, sigma_f, X_train, v):
    result = 0
    for i, x in enumerate(X_train):
        tmp = 0
        for j in range(len(l)):
            tmp += -0.5 * (x[j] / l[j] - C_list[j] / l[j]) * \
                (x[j] / l[j] - C_list[j] / l[j])
        result += sigma_f**2 * np.exp(tmp) * v[i]
    print(result)
    return result


def test_custom(X_test, Y_test, X_train, Y_train, sigma_y):
    for i in range(10):
        ufl_pre_test(X_test[i], l, sigma_f, X_train, v)


def gpr(args):
    Xin, Xout = load_data_all(args, rm_dup=True, middle=False)
    X_total = Xin[:, :]
    Y_total = Xout[:]
    X_train, Y_train, X_test, Y_test = train_test_split(X_total, Y_total)
    noise = 0.01

    # scikit learn
    rbf = ConstantKernel(1.0) * RBF(length_scale=[0.1, 0.1, 0.1, 0.1])
    gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)

    gpr.fit(X_train, Y_train)

    # Obtain optimized kernel parameters
    l = gpr.kernel_.k2.get_params()['length_scale']
    sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

    print(l, sigma_f)

    print("\nY_test\n", Y_test)

    # scikit learn test
    mu_s = gpr.predict(X_test)
    print("\nscikit learn predicts\n", mu_s)
    print("MSE test", np.sum((mu_s - Y_test)**2) / len(mu_s))

    # custom test
    mu_s, _ = posterior_predictive_anisotropic(
        X_test, X_train, Y_train, l, sigma_f, noise)
    print("\ncustom code predicts\n", mu_s)
    print("MSE test", np.sum((mu_s - Y_test)**2) / len(mu_s))

    K = kernel_anisotropic(X_train, X_train, l, sigma_f) + \
        noise**2 * np.eye(len(X_train))
    K_inv = inv(K)
    v = K_inv.dot(Y_train)
    np.savez('plots/new_data/numpy/gpr/para.npz', l=np.asarray(l),
             sigma_f=np.asarray(sigma_f), X_train=X_train, v=v)


def train_test_split(X_total, Y_total):
    test_ratio = 0.1
    n_samps = len(X_total)
    n_train = int((1 - test_ratio) * n_samps)
    inds = np.random.permutation(n_samps)
    inds_train = inds[:n_train]
    inds_test = inds[n_train:]
    X_train = X_total[inds_train]
    Y_train = Y_total[inds_train]
    X_test = X_total[inds_test]
    Y_test = Y_total[inds_test]
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    args = arguments.args
    # scikit_learn_gp()
    gpr(args)
