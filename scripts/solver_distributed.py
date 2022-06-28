import sys
sys.path.append("..")

from src.problem import *
from src.distributed_augmented_lagrangian import *
from utils.file_handling import get_params_file
from utils.plotting import *
from utils.data_handling import *

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

    
    flag2D = True
    # Plot results
    for idx, agent in enumerate(prob.agents):
        if agent.n != 2:
            flag2D = False
    
    if flag2D:
        plot2DagentsMap(dist_al.x)
        plot2Dagents(dist_al.x, "state")
        plot2Dagents(dist_al.u, "input")
    
    plotMaxConstraintsConvergence(dist_al.max_constraints_convergence, 'log')

    plotObjectiveFunctionConvergence(dist_al.solution_value_convergence)

    # for ii in range(len(dist_al.prob.agents)):
    #     save_solution(dist_al.x[ii].value, filename, 'dist')
    #     save_solution(dist_al.u[ii].value, filename, 'dist')
    
    verify_solution(dist_al)
    verify_cost_function(dist_al)