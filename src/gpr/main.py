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


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
        np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + \
        sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X_train, Y_train, noise, naive=False):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given 
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (7), if 
               False use a numerically more stable implementation. 

    Returns:
        Minimization objective.
    '''
    def nll_naive(theta):
        # Naive implementation of Eq. (7). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
            0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
            0.5 * len(X_train) * np.log(2 * np.pi)

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
            0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
            0.5 * len(X_train) * np.log(2 * np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable


def gp_exp_2d():
    noise_2D = 0.01

    rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
    gx, gy = np.meshgrid(rx, rx)

    X_2D = np.c_[gx.ravel(), gy.ravel()]

    X_2D_train = np.random.uniform(-4, 4, (100, 2))
    Y_2D_train = np.sin(0.5 * np.linalg.norm(X_2D_train, axis=1)) + \
        noise_2D * np.random.randn(len(X_2D_train))

    plt.figure(figsize=(14, 7))

    print(X_2D_train.shape)
    print(Y_2D_train.shape)

    mu_s, _ = posterior_predictive(
        X_2D, X_2D_train, Y_2D_train, sigma_y=noise_2D)
    plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train,
               f'Before parameter optimization: l={1.00} sigma_f={1.00}', 1)

    res = minimize(nll_fn(X_2D_train, Y_2D_train, noise_2D), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')

    mu_s, _ = posterior_predictive(
        X_2D, X_2D_train, Y_2D_train, *res.x, sigma_y=noise_2D)
    plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train,
               f'After parameter optimization: l={res.x[0]:.2f} sigma_f={res.x[1]:.2f}', 2)
    plt.show()


def gp_exp_1d():
    noise = 0.01
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    X_train = np.arange(-3, 4, 1).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

    print(X_train.shape)
    print(Y_train.shape)

    # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
    # We should actually run the minimization several times with different
    # initializations to avoid local minima but this is skipped here for
    # simplicity.
    res = minimize(nll_fn(X_train, Y_train, noise), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')

    # Store the optimization results in global variables so that we can
    # compare it later with the results from other implementations.
    l_opt, sigma_f_opt = res.x
    print(l_opt, sigma_f_opt)

    # Compute the prosterior predictive statistics with optimized kernel
    # parameters and plot the results
    mu_s, cov_s = posterior_predictive(
        X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise)
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
    # plt.show()


def scikit_learn_gp():
    noise = 0.01
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    X_train = np.arange(-3, 4, 1).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)

    # Reuse training data from previous 1D example
    gpr.fit(X_train, Y_train)

    # Compute posterior predictive mean and covariance
    mu_s, cov_s = gpr.predict(X, return_cov=True)

    # Obtain optimized kernel parameters
    l = gpr.kernel_.k2.get_params()['length_scale']
    sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

    print(l, sigma_f)
    # Plot the results
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
    # plt.show()


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
            tmp += -0.5 * (x[j]/l[j] - C_list[j]/l[j])*(x[j]/l[j] - C_list[j]/l[j])
        result += sigma_f**2 * np.exp(tmp) * v[i]    
    print(result)
    return result

def test_custom(X_test, Y_test, X_train, Y_train, sigma_y):

    l = np.array([0.421, 0.688, 0.463, 0.52 ])
    sigma_f = 2.492785052785189
    K = kernel_anisotropic(X_train, X_train, l, sigma_f) + \
        sigma_y**2 * np.eye(len(X_train))
    K_inv = inv(K)
    v = K_inv.dot(Y_train)

    np.savez('plots/new_data/numpy/gpr/para.npz', l=l, sigma_f=sigma_f, X_train=X_train, v=v)
    exit()

    # for i in range(10):
    #     ufl_pre_test(X_test[i], l, sigma_f, X_train, v)
 
    mu_s, _ = posterior_predictive_anisotropic(
        X_test, X_train, Y_train, l, sigma_f, sigma_y)
    print(np.sum((mu_s - Y_test)**2) / len(mu_s))
    print(mu_s)


def gp_exp_homo(args):

    Xin, Xout = load_data_all(args, rm_dup=True, middle=False)
    X_total = Xin[:, :]
    Y_total = Xout[:]
    X_train, Y_train, X_test, Y_test = train_test_split(X_total, Y_total)
    noise = 0.01

    test_custom(X_test, Y_test, X_train, Y_train, noise)
 
    # scikit learn
    rbf = ConstantKernel(1.0) * RBF(length_scale=[0.1, 0.1, 0.1, 0.1])
    gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)

    gpr.fit(X_train, Y_train)

    # Obtain optimized kernel parameters
    l = gpr.kernel_.k2.get_params()['length_scale']
    sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

    print(l, sigma_f)

    mu_s = gpr.predict(X_test)
    print(mu_s)
    print(Y_test)
    print(np.sum((mu_s - Y_test)**2) / len(mu_s))


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
    gp_exp_homo(args)
