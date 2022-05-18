import sys
sys.path.append("..")
import cvxpy as cp
import problem
import numpy as np
import centralised_augmented_lagrangian as cal
from utils.file_handling import get_params_file

if __name__ == '__main__':
    
    # Get path to the problem configuration file
    file = get_params_file()

    # Set up the problem
    prob = problem.CentralisedProblem(file)

    # Initialise optimisation related constants
    rho = 5
    horizon = 10


    cent_al = cal.CentralisedAugmentedLagrangian(prob, rho, horizon)
    print(cent_al.x.value)
    '''
    # Set up optimisation variables
    x = cp.Variable((prob.n*(horizon+1),1))
    u = cp.Variable((prob.p*horizon,1))
    lam = cp.Parameter((prob.n*horizon,1))
    lam.value = np.ones((prob.n*horizon,1))


    #Compose the cost function
   
    cost_function = 0
    
    # sum the quadratic cost for the input over the horizon 
    for ii in range(horizon):
        cost_function += cp.quad_form(u[ii*prob.p:(ii+1)*prob.p], prob.MatCost) 
    
    # sum the inner product of lambda with the constraints over the horizon
    for ii in range(horizon):
        cost_function += lam[ii*prob.n:(ii+1)*prob.n].T @ (x[(ii+1)*prob.n:(ii+2)*prob.n] - prob.MatA @ x[ii*prob.n:(ii+1)*prob.n] -prob.MatB @ u[ii*prob.p:(ii+1)*prob.p])

    # # sum the squared norm of the constraints over the horizon
    for ii in range(horizon):
        cost_function += (rho/2) * cp.sum_squares(x[(ii+1)*prob.n:(ii+2)*prob.n] - prob.MatA @ x[ii*prob.n:(ii+1)*prob.n] -prob.MatB @ u[ii*prob.p:(ii+1)*prob.p])

    # Add equality constraints to initial and final states and box constraints to state and input
    constraints = [
        x[0:prob.n] == prob.x0,
        x[-prob.n:,] == prob.xH,
        ]

    objective = cp.Minimize(cost_function)
    opt_mpc = cp.Problem(objective, constraints)
    result = opt_mpc.solve() # warm_start=True

    print(x.value)
    '''