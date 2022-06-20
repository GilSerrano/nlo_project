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
    residual = np.array([])
    for tt in range(data.prob.horizon):
        for ii, agent in enumerate(data.prob.agents):
            res = data.x[ii].value[:,tt+1] - (agent.matA @ data.x[ii].value[:,tt] + agent.matB @ data.u[ii].value[:,tt])
            for jj in agent.in_neigh:
                in_agent = data.prob.agents[jj-1]
                res -= in_agent.outA[(agent.idx,in_agent.idx)] @ data.x[jj-1].value[:,tt] + in_agent.outB[(agent.idx,in_agent.idx)] @ data.u[jj-1].value[:,tt]

            residual = np.append(residual, np.linalg.norm(res))

    print("Residual")
    print(residual)
    if np.all(np.less_equal(residual, 10**(-4))):
        print("Solution is valid. Max entry is " + str(np.amax(residual)))
    else:
        print("Solution is NOT valid. Max entry is " + str(np.amax(residual)))