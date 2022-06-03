import numpy as np
from datetime import datetime

from src.distributed_augmented_lagrangian import *

def save_solution(data, filename, type):
    dateTimeObj = datetime.now()
    fileprefix = '../data/'
    filesuffix = '-' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '-' + str(dateTimeObj.hour) + str(dateTimeObj.minute) + str(dateTimeObj.second)
    save_filename = fileprefix + type + '-' + filename[:-5] + filesuffix 

    with open(save_filename +'.npy', 'ab') as file:
        np.save(file, data, allow_pickle=True)

def save_solution_dict(data, filename, type):
    dateTimeObj = datetime.now()
    fileprefix = '../data/'
    filesuffix = '-' + str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + '-' + str(dateTimeObj.hour) + str(dateTimeObj.minute) + str(dateTimeObj.second)
    save_filename = fileprefix + type + '-' + filename[:-5] + filesuffix 
    
    data_dict = {}
    for ii, agent in enumerate(data.prob.agents):
        key = 'agent' + str(ii+1)
        data_dict[key] = {
            'x': data.x[ii].value,
            'u': data.u[ii].value,
            'x0':agent.x_0,
            'xH':agent.x_H,
            'n': agent.n,
            'p': agent.p,
            'matA': agent.matA,
            'matB': agent.matB,
            'outA': agent.outA,
            'outB': agent.outB,
            'in_neigh': agent.in_neigh,
            'out_neigh': agent.out_neigh,
            'horizon': data.prob.horizon
        }
    
    
    # data is of type Distributed/Centralised AugmentedLagrangian
    with open(save_filename +'.npy', 'ab') as file:
        np.save(file, data_dict, allow_pickle=True)
        
'''
data is of class CentralisedAugmentedLagrangian or DistributedAugmentedLagrangian
'''
def verify_solution(data):

    for tt in data.prob.horizon:
        for ii, agent in enumerate(data.prob.agents):
            residual = data.x[ii].value[:,tt+1]