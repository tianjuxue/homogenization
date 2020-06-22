import matplotlib.pyplot as plt
import numpy as np


def plot_reg():
    plt.figure(0)
    train_MSE = np.load('plots/new_data/numpy/polynomial/train_MSE.npy')
    test_MSE = np.load('plots/new_data/numpy/polynomial/test_MSE.npy')
    poly_degree = np.load('plots/new_data/numpy/polynomial/poly_degree.npy')
    plt.plot(poly_degree, train_MSE, linestyle='--', marker='o', color='red')
    plt.plot(poly_degree, test_MSE, linestyle='--', marker='o', color='blue')
    plt.plot(poly_degree, 7.67*1e-5*np.ones_like(poly_degree), color='black')
    # plt.xlabel("Polynomial Degree", fontsize=14)
    # plt.ylabel("MSE", fontsize=14)
    plt.tick_params(labelsize=14)
    plt.yscale("log")
    plt.show()


if __name__ == '__main__':
    plot_reg()
