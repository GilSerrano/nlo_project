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
    
    # Parse x and u, and plot
    for idx, _ in enumerate(prob.agents):
        state = get_agent_states(prob, cent_al.x.value, idx+1)
        input = get_agent_inputs(prob, cent_al.u.value, idx+1)
        plotNDagent(state, idx+1)
        plotNDagent(input, idx+1)
        for tt in range(prob.horizon):
            cost += input[:,tt].T @ prob.agents[idx].matCost @ input[:,tt]
    
    print('Total cost is ' + str(cost))

    # for ii in range(len(prob.agents)):
    #     print("Agent " + str(ii+1))
    #     print(cent_al.x[ii].value)
    #     print(cent_al.u[ii].value)

    # flag2D = True
    # # Plot results
    # for idx, agent in enumerate(prob.agents):
    #     if agent.n != 2:
    #         flag2D = False
    #     # Plot state
    #     plotNDagent(cent_al.x[idx].value, idx+1)
    #     if agent.n == 2:
    #         plot2DagentMap(cent_al.x[idx].value, idx+1)
        
    #     # Plot input
    #     plotNDagent(cent_al.u[idx].value, idx+1)
    
    # if flag2D:
    #     plot2DagentsMap(cent_al.x)

    # for ii in range(len(cent_al.prob.agents)):
    #     save_solution(cent_al.x[ii].value, filename, 'cent')
    #     save_solution(cent_al.u[ii].value, filename, 'cent')
    
    # verify_solution(cent_al)