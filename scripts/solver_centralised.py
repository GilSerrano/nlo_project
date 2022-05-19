import sys
sys.path.append("..")

from src.problem import *
from src.centralised_augmented_lagrangian import *
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
    prob = CentralisedProblem(file)

    # Begin solving the problem with centralised adal
    cent_al = CentralisedAugmentedLagrangian(prob)
    print(cent_al.x.value)

    # Parse x and u, and plot
    for idx, _ in enumerate(prob.agents):
        state = get_agent_states(prob, cent_al.x.value, idx+1)
        input = get_agent_inputs(prob, cent_al.u.value, idx+1)
        plotNDagent(state, idx+1)
        plotNDagent(input, idx+1)