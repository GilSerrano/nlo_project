import numpy as np

'''
result must be a numpy array
'''
def get_agent_states(prob, result, idx):
    # figure out number of states of agents before idx
    nb = sum(prob.n_i[0:idx-1])

    states = []
    for ii in range(prob.horizon + 1):
        states.append(result[prob.n*ii+nb:prob.n*ii+nb+prob.n_i[idx-1]])

    return np.array(states).reshape((prob.horizon+1, prob.n_i[idx-1])).T

def get_agent_inputs(prob, result, idx):
    # figure out number of states of agents before idx
    pb = sum(prob.p_i[0:idx-1])

    inputs = []
    for ii in range(prob.horizon):
        inputs.append(result[prob.p*ii+pb:prob.p*ii+pb+prob.p_i[idx-1]])
    
    return np.array(inputs).reshape((prob.horizon, prob.p_i[idx-1])).T