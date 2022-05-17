import cvxpy as cp
import problem
import file_handling
from pathlib import Path

if __name__ == '__main__':
    
    # Get path to the problem configuration file
    file = file_handling.get_params_file()

    # Set up the problem
    prob = problem.CentralisedProblem(file)
    
    # Set up optimization variables
    x = cp.Variable((prob.n, 1))
    u = cp.Variable((prob.p, 1))
    lam = cp.Variable((prob.n, 1))








    
    print('The total state size is ' + str(prob.n))
