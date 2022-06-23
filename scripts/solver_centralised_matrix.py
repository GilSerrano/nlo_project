import sys
sys.path.append("..")

from src.problem import *
from src.centralised_augmented_lagrangian_matrix import *
from utils.file_handling import get_params_file
from utils.plotting import *
from utils.data_handling import save_solution, verify_solution
from utils.results_parser import *

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
    
    cost = 0
    
    x = [[] for ii in range(len(prob.agents))]

    # Parse x and u, and plot
    for idx, _ in enumerate(prob.agents):
        state = get_agent_states(prob, cent_al.x.value, idx+1)
        input = get_agent_inputs(prob, cent_al.u.value, idx+1)
        plotNDagent(state, idx+1)
        plotNDagent(input, idx+1)
        x[idx] = state
        for tt in range(prob.horizon):
            cost += input[:,tt].T @ prob.agents[idx].matCost @ input[:,tt]
    
    if prob.agents[0].n == 2:
        plot2DagentsMapMatrix(x)


    print('Total cost is ' + str(cost))

    
    # for ii in range(len(cent_al.prob.agents)):
    #     save_solution(cent_al.x[ii].value, filename, 'cent')
    #     save_solution(cent_al.u[ii].value, filename, 'cent')
    
    # verify_solution(cent_al)