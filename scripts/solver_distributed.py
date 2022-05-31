import sys
sys.path.append("..")

from src.problem import *
from src.distributed_augmented_lagrangian import *
from utils.file_handling import get_params_file
from utils.plotting import *
from utils.results_parser import get_agent_states, get_agent_inputs

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        filename = sys.argv[1] # load file passed by user
    else:
        filename = 'mpc_3agents_3d_noInteractions' # default file

    # Get path to the problem configuration file
    file = get_params_file(filename)

    # Set up the problem
    prob = DistributedProblem(file)

    # Begin solving the problem with ADAL
    dist_al = DistributedAugmentedLagrangian(prob)
    for ii in range(len(prob.agents)):
        print("Agent " + str(ii+1))
        print(dist_al.x[ii].value)
        print(dist_al.u[ii].value)


    # Parse x and u, and plot
    for idx, _ in enumerate(prob.agents):
        plotNDagent(dist_al.x[idx].value, idx+1)
        plotNDagent(dist_al.u[idx].value, idx+1)