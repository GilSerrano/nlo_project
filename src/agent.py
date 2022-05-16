'''
Definition of class Agent


'''

import numpy as np
import yaml


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
        
        


if __name__ == "__main__":
    
    with open("C:/Users/gil97/Documents/Repositories/nlo_project/config/mpc_3agents_noInteractions.yaml", 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            print('Loaded data.')
        except yaml.YAMLError as error:
            print(error)
    
    agents = []
    
    for key in data:
        print(key)
        agents.append(Agent(data[key]))       