import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Hard code
    neuron_64 = [0.00300116, 0.0030661, 0.00187069, 0.00141373, 0.0033632, 0.00686808, 0.02220321, 0.2071431, 0.23567955]
    neuron_128 = [0.00532514, 0.00141033, 0.00188422, 0.00113992, 0.0021974, 0.00956167, 0.01435242, 0.07035027, 0.22494337]
    neuron_256 = [0.0038742, 0.00293321,0.00123023, 0.00110485, 0.00424966, 0.01013271, 0.00794241, 0.01451418, 0.14174967]
    x = [1,2,3]
    locs = ['upper right', 'lower right', 'lower right']



    for i in range(3):
        fig = plt.figure()
        plt.tick_params(labelsize=12)
        # plt.xlabel('strain')
        # plt.ylabel('stress')
        plt.plot(x, neuron_64[3*i:3*(i+1)], '-*', color='blue', label='64') 
        plt.plot(x, neuron_128[3*i:3*(i+1)], '-*', color='red', label='128') 
        plt.plot(x, neuron_256[3*i:3*(i+1)], '-*', color='orange', label='256') 
        plt.legend(loc=locs[i], prop={'size': 12})
        plt.xticks(x)

    plt.show()
    # fig.savefig("gradient_validation.pdf", bbox_inches='tight')