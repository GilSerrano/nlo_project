import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from utils.data_handling import verify_solution
from utils.plotting import plot2DagentsMap
from src.distributed_augmented_lagrangian import *
from src.problem import *

if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        filename = sys.argv[2] # load file passed by user
    else:
        filename = 'dist-mpc_3agents_2d_simpleInteractions-202263-172825.npy' # default file
    
    prob = Problem(sys.argv[1])

    x = [[] for ii in range(len(prob.agents))]
    u = [[] for ii in range(len(prob.agents))]

    with open('../data/' + filename, 'rb') as file:
        for ii in range(len(prob.agents)):
            x[ii] = np.load(file, allow_pickle=True) 
            u[ii] = np.load(file, allow_pickle=True) 
    
    # Plot data
    for xx in x:
        plt.plot(xx[0,:], xx[1,:], '-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

    plt.suptitle('All agents')
    plt.show()