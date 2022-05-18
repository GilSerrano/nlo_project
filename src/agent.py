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
        self.x_0 = np.transpose(np.array(data['x_0']))
        self.x_H = np.transpose(np.array(data['x_H']))
        
        # In and Out Neighbourhoods
        self.in_neigh  = data['in_neigh']
        self.out_neigh = data['out_neigh']
        
        # Own dynamics
        self.matA = np.array(data['matA'])
        self.matB = np.array(data['matB'])

        # Dynamic interactions with out-neighbours
        self.outA = {}
        self.outB = {}
        for ii, out_idx in enumerate(self.out_neigh):
            self.outA[(out_idx, self.idx)] = np.array(data['matA_inter'][ii])
            self.outB[(out_idx, self.idx)] = np.array(data['matB_inter'][ii])

        self.matCost = np.array(data['matCost'])

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
        
if __name__ == "__main__":
    
    file = get_params_file()
    with open(file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print('Loaded data.')
        except yaml.YAMLError as error:
            print(error)
    
    agents = []
    
    for key in data:
        print(key)
        agents.append(Agent(data[key]))       