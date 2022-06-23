import numpy as np

def generate_agent(agent_idx, n_state, N_agents):

    lim_x = 4 # to make the interval between -2 and 2
    lim_M = 10
    lim_inter = 0.1

    # Generate initial and final states
    x_0 = lim_x * (np.random.rand(n_state,) - 0.5)
    x_0 = x_0.tolist()
    x_H = lim_x * (np.random.rand(n_state,) - 0.5)
    x_H = x_H.tolist()

    # Generate a positive semidefinite cost matrix
    matRand =  np.random.rand(n_state, n_state)
    matCost = lim_M * (matRand.T @ matRand)
    matCost = matCost.tolist()

    # Generate set of out neighbours
    out_neigh = set()
    for ii in range(N_agents): # go over all agents
        if np.random.randint(0,2) and (ii+1)!=agent_idx: # if we get randomly a 1, we add to the out neighbourhood
            out_neigh.add(ii+1)

    print('agent'+str(agent_idx)+' out neigh is')
    print(out_neigh)

    # Generate diagonal state and input matrices
    matA = np.diag(np.random.rand(n_state,)).tolist()
    matB = np.diag(np.random.rand(n_state,)).tolist()

    # Generate diagonal interaction state and input matrices
    matA_inter = [] 
    matB_inter = []
    for ii in out_neigh:
        matA_inter.append(np.diag(lim_inter * np.random.rand(n_state,)).tolist())
        matB_inter.append(np.diag(lim_inter * np.random.rand(n_state,)).tolist())

    # Create dictionary and return
    return dict(idx = agent_idx,
                n_i = n_state,
                p_i = n_state,
                x_0 = x_0,
                x_H = x_H,
                in_neigh = set(), # in neighbourhood needs to be done after all agents are generated
                out_neigh = out_neigh, 
                matA = matA,
                matB = matB,
                matA_inter = matA_inter,
                matB_inter = matB_inter,
                matCost = matCost)


def update_in_neighbourhood(prob_dict):
    # go over every agent in network
    for _, agent in prob_dict.items():
        # go over agents in its out neighbourhood
        for jj in agent['out_neigh']:
            print(jj)
            # add agent to their in neighbourhood
            prob_dict['agent'+str(jj)]['in_neigh'].add(agent['idx'])
    
    for _, agent in prob_dict.items():
        print('agent'+str(agent['idx'])+' in neigh is')
        print(agent['in_neigh'])