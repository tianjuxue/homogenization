import numpy as np
import os
import fenics as fa
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path_prefix_energy = 'plots/new_data/numpy/energy/'
    path_prefix_force = 'plots/new_data/numpy/force/'
    path_prefix_time = 'plots/new_data/numpy/time/'

    DNS_energy_com_pore0 = np.load(path_prefix_energy + 'DNS_energy_com_pore0.npy') # 309.06
    DNS_energy_ten_pore0 = np.load(path_prefix_energy + 'DNS_energy_ten_pore0.npy') # 315.3
    NN_energy_com_pore0 = np.load(path_prefix_energy + 'NN_energy_com_pore0.npy') # 6.720
    NN_energy_ten_pore0 = np.load(path_prefix_energy + 'NN_energy_ten_pore0.npy') # 7.103
    DNS_energy_com_pore2 = np.load(path_prefix_energy + 'DNS_energy_com_pore2.npy') # 247.9
    DNS_energy_ten_pore2 = np.load(path_prefix_energy + 'DNS_energy_ten_pore2.npy') # 246.2
    NN_energy_com_pore2 = np.load(path_prefix_energy + 'NN_energy_com_pore2.npy') # 8.753
    NN_energy_ten_pore2 = np.load(path_prefix_energy + 'NN_energy_ten_pore2.npy') # 7.443

    DNS_force_com_pore0 = np.load(path_prefix_force + 'DNS_force_com_pore0.npy')
    DNS_force_ten_pore0 = np.load(path_prefix_force + 'DNS_force_ten_pore0.npy')
    NN_force_com_pore0 = np.load(path_prefix_force + 'NN_force_com_pore0.npy')
    NN_force_ten_pore0 = np.load(path_prefix_force + 'NN_force_ten_pore0.npy')
    DNS_force_com_pore2 = np.load(path_prefix_force + 'DNS_force_com_pore2.npy')
    DNS_force_ten_pore2 = np.load(path_prefix_force + 'DNS_force_ten_pore2.npy')
    NN_force_com_pore2 = np.load(path_prefix_force + 'NN_force_com_pore2.npy')
    NN_force_ten_pore2 = np.load(path_prefix_force + 'NN_force_ten_pore2.npy')

    # strain_ten = np.linspace(0,  0.1, len(DNS_energy_ten_pore0))
    # strain_com = np.linspace(0, -0.1, len(DNS_energy_com_pore0))
    # print("strain_com is", strain_com)


    fig = plt.figure(0)
    plt.tick_params(labelsize=14)
    plt.plot(-0.1*np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11))), (DNS_energy_com_pore0 - DNS_energy_com_pore0[0]), '-', color='blue', label='DNS ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  0.1, len(DNS_energy_ten_pore0)), (DNS_energy_ten_pore0 - DNS_energy_ten_pore0[0]), '-', color='blue', label='DNS ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  -0.1, len(NN_energy_com_pore0)), (NN_energy_com_pore0 - NN_energy_com_pore0[0]), '--', color='blue', label='NN ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  0.1, len(NN_energy_ten_pore0)), (NN_energy_ten_pore0 - NN_energy_ten_pore0[0]), '--', color='blue', label='NN ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  -0.1, len(DNS_energy_com_pore2)), (DNS_energy_com_pore2 - DNS_energy_com_pore2[0]), '-', color='red', label='DNS ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  0.1, len(DNS_energy_ten_pore2)), (DNS_energy_ten_pore2 - DNS_energy_ten_pore2[0]), '-', color='red', label='DNS ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  -0.1, len(NN_energy_com_pore2)), (NN_energy_com_pore2 - NN_energy_com_pore2[0]), '--', color='red', label='NN ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  0.1, len(NN_energy_ten_pore2)), (NN_energy_ten_pore2 - NN_energy_ten_pore2[0]), '--', color='red', label='NN ' + r'$\xi_b$')

    fig = plt.figure(1)
    plt.tick_params(labelsize=14)
    plt.plot(-0.1*np.concatenate((np.linspace(0, 0.75, 6), np.linspace(0.75, 1., 11))), (DNS_force_com_pore0 - DNS_force_com_pore0[0]), '-', color='blue', label='DNS ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  0.1, len(DNS_force_ten_pore0)), (DNS_force_ten_pore0 - DNS_force_ten_pore0[0]), '-', color='blue', label='DNS ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  -0.1, len(NN_force_com_pore0)), (NN_force_com_pore0 - NN_force_com_pore0[0]), '--', color='blue', label='NN ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  0.1, len(NN_force_ten_pore0)), (NN_force_ten_pore0 - NN_force_ten_pore0[0]), '--', color='blue', label='NN ' + r'$\xi_a$')
    plt.plot(np.linspace(0,  -0.1, len(DNS_force_com_pore2)), (DNS_force_com_pore2 - DNS_force_com_pore2[0]), '-', color='red', label='DNS ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  0.1, len(DNS_force_ten_pore2)), (DNS_force_ten_pore2 - DNS_force_ten_pore2[0]), '-', color='red', label='DNS ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  -0.1, len(NN_force_com_pore2)), (NN_force_com_pore2 - NN_force_com_pore2[0]), '--', color='red', label='NN ' + r'$\xi_b$')
    plt.plot(np.linspace(0,  0.1, len(NN_force_ten_pore2)), (NN_force_ten_pore2 - NN_force_ten_pore2[0]), '--', color='red', label='NN ' + r'$\xi_b$')


    print(np.load(path_prefix_time + 'DNS_time_com_pore0.npy'))
    print(np.load(path_prefix_time + 'DNS_time_com_pore2.npy'))
    print(np.load(path_prefix_time + 'DNS_time_ten_pore0.npy'))
    print(np.load(path_prefix_time + 'DNS_time_ten_pore2.npy'))
    print(np.load(path_prefix_time + 'NN_time_com_pore0.npy'))
    print(np.load(path_prefix_time + 'NN_time_com_pore2.npy'))
    print(np.load(path_prefix_time + 'NN_time_ten_pore0.npy'))
    print(np.load(path_prefix_time + 'NN_time_ten_pore2.npy'))

    print(NN_force_com_pore0)
    print(NN_force_ten_pore0)
    print(NN_force_com_pore2)
    print(NN_force_ten_pore2)

    plt.show()
