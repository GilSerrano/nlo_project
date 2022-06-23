import sys
sys.path.append("..")

import yaml
import numpy as np
from datetime import datetime

from utils.generate_agent import *

if __name__ == '__main__':
    
    # make sure input arguments are passed correctly
    if len(sys.argv) < 3:
        raise Exception('Number of agents and dimension are required and was not passed.')
    else:
        N_agents = int(sys.argv[1])
        n_state = int(sys.argv[2])

    if len(sys.argv) == 4:
        filename = 'mpc_' + str(N_agents) + 'agents_'+str(n_state)+'d_randomInteractions-' + sys.argv[3] # name file after string typed by user
    else:
        date = datetime.now()
        filesuffix = str(date.year) + str(date.month) + str(date.day) + '-' + str(date.hour) + str(date.minute) + str(date.second)
        filename = 'mpc_' + str(N_agents) + 'agents_'+str(n_state)+'d_randomInteractions-' + filesuffix 
    
    # create dictionary of problem setup
    prob_dict = dict()
    for ii in range(N_agents):
        key = 'agent'+str(ii+1)
        prob_dict[key] = generate_agent(ii+1, n_state, N_agents)
    
    # fill the in-neighbourhoods of every agent
    update_in_neighbourhood(prob_dict)

    # save to yaml
    with open('../config/'+filename+'.yaml', 'w') as outfile:
        yaml.dump(prob_dict, outfile, default_flow_style=False)