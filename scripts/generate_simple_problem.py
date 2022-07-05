import sys
sys.path.append("..")

import yaml
import numpy as np
from datetime import datetime

from utils.generate_agent import *

if __name__ == '__main__':
    
    # make sure input arguments are passed correctly
    if len(sys.argv) < 4:
        raise Exception('Number of agents, dimension, and number of neighbours are required and were not passed.')
    else:
        N_agents = int(sys.argv[1])
        n_state = int(sys.argv[2])
        N_neigh = int(sys.argv[3])

    if len(sys.argv) == 5:
        filename = 'mpc_' + str(N_agents) + 'agents_'+str(n_state)+'d_randomInteractions-' + sys.argv[4] # name file after string typed by user
    else:
        date = datetime.now()
        filesuffix = str(date.year) + str(date.month) + str(date.day) + '-' + str(date.hour) + str(date.minute) + str(date.second)
        filename = 'mpc_' + str(N_agents) + 'agents_'+str(n_state)+'d_randomInteractions-' + filesuffix 
    
    # create dictionary of problem setup
    prob_dict = dict()
    for ii in range(N_agents):
        key = 'agent'+str(ii+1)
        prob_dict[key] = generate_simple_agent(ii+1, n_state, N_agents, N_neigh)
    
    # fill the in-neighbourhoods of every agent
    update_out_neighbourhood(prob_dict)

    # save to yaml
    with open('../config/'+filename+'.yaml', 'w') as outfile:
        yaml.dump(prob_dict, outfile, default_flow_style=False)