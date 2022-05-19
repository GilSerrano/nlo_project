import sys
sys.path.append("..")
import cvxpy as cp
import problem
import numpy as np
import centralised_augmented_lagrangian as cal
from utils.file_handling import get_params_file
from utils.plotting import plot3Dagent
from utils.results_parser import get_agent_states, get_agent_inputs

if __name__ == '__main__':
    
    #filename = 'mpc_3agents_2d_noInteractions'
    filename = 'mpc_3agents_3d_noInteractions'
    #filename = 'mpc_3agents_3d_simpleInteractions'

    # Get path to the problem configuration file
    file = get_params_file(filename)

    # Set up the problem
    prob = problem.CentralisedProblem(file)

    # Begin solving the problem with centralised adal
    cent_al = cal.CentralisedAugmentedLagrangian(prob)
    print(cent_al.x.value)

    ag1_state = get_agent_states(prob, cent_al.x.value, 1)
    ag2_state = get_agent_states(prob, cent_al.x.value, 2)
    ag3_state = get_agent_states(prob, cent_al.x.value, 3)

    ag1_input = get_agent_inputs(prob, cent_al.u.value, 1)
    ag2_input = get_agent_inputs(prob, cent_al.u.value, 2)
    ag3_input = get_agent_inputs(prob, cent_al.u.value, 3)

    plot3Dagent(ag1_state, 1)
    plot3Dagent(ag2_state, 2)
    plot3Dagent(ag3_state, 3)

    plot3Dagent(ag1_input, 1)
    plot3Dagent(ag2_input, 2)
    plot3Dagent(ag3_input, 3)