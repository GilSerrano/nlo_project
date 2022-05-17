import os
import sys

def get_params_file():

    curr_dir = os.path.dirname(__file__) 
    path = '../config/mpc_3agents_noInteractions.yaml'
    
    if len(sys.argv) > 1: # pass parameter file as cmd line input
        path = '../config/'+sys.argv[1]
    
    return os.path.join(curr_dir, path)