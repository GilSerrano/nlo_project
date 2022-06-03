'''
Definition of class Agent
'''
import sys
sys.path.append("..")
import numpy as np
import yaml
from utils.file_handling import get_params_file

class Agent(object):

    def __init__(self, data):
        self.idx = data['idx']
        self.n = data['n_i'] # state dim
        self.p = data['p_i'] # input dim
        
        # Initial and final conditions
        self.x_0 = np.array(data['x_0'], dtype='float64').T
        self.x_H = np.array(data['x_H'], dtype='float64').T
        
        # In and Out Neighbourhoods
        self.in_neigh  = data['in_neigh']
        self.out_neigh = data['out_neigh']
        
        # Own dynamics
        self.matA = np.array(data['matA'], dtype='float64')
        self.matB = np.array(data['matB'], dtype='float64')

        # Dynamic interactions with out-neighbours
        self.outA = {}
        self.outB = {}
        for ii, out_idx in enumerate(self.out_neigh):
            self.outA[(out_idx, self.idx)] = np.array(data['matA_inter'][ii], dtype='float64')
            self.outB[(out_idx, self.idx)] = np.array(data['matB_inter'][ii], dtype='float64')

        self.matCost = np.array(data['matCost'], dtype='float64')

    def __repr__(self):
        output = ''
        output += 'Agent id: ' + str(self.idx) + '\n'
        output += '\t Dimensions: n = ' + str(self.n) + ' p = ' + str(self.p) + '\n'
        output += '\t Initial conditions: x0 = ' + str(self.x_0) + '\n'
        output += '\t Final goal: xH = ' + str(self.x_H) + '\n'
        output += '\t In-neighbourhood:  { ' + str(self.in_neigh) + ' }\n'
        output += '\t Out-neighbourhood: { ' + str(self.out_neigh) + ' }\n'
        output += '\t Matrix A = ' + str(self.matA) + '\n'
        output += '\t Matrix B = ' + str(self.matB) + '\n'
        for key, value in self.outA.items():
            output += '\t A'+ str(key) + ' = ' + str(value) + '\n'
        for key, value in self.outB.items():
            output += '\t B'+ str(key) + ' = ' + str(value) + '\n'
        
        return output    