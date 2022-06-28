import sys
sys.path.append("..")

from src.problem import *
from utils.file_handling import get_params_file
from utils.plotting import *


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        filename = sys.argv[1] # load file passed by user
    else:
        filename = 'mpc_3agents_2d_noInteractions' # default file

    # Get path to the problem configuration file
    file = get_params_file(filename)

    # Set up the problem
    prob = Problem(file)

    plotAgentNetwork(prob)
    