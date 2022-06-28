import sys
sys.path.append("..")

from src.problem import *
from src.centralised_augmented_lagrangian import *
from utils.file_handling import get_params_file
from utils.plotting import *
from utils.data_handling import *
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

    for ii in range(len(prob.agents)):
        print("Agent " + str(ii+1))
        print(cent_al.x[ii].value)
        print(cent_al.u[ii].value)

    flag2D = True
    # Plot results
    for idx, agent in enumerate(prob.agents):
        if agent.n != 2:
            flag2D = False
    
    if flag2D:
        plot2DagentsMap(cent_al.x)

    plotConstraintsConvergence(cent_al.constraints_convergence)
    plotConstraintsConvergence(cent_al.constraints_convergence, 'log')
    plotMaxConstraintsConvergence(cent_al.max_constraints_convergence)
    plotMaxConstraintsConvergence(cent_al.max_constraints_convergence, 'log')

    verify_solution(cent_al)
    verify_cost_function(cent_al)