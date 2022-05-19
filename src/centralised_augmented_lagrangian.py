import cvxpy as cp
import numpy as np

class CentralisedAugmentedLagrangian(object):
    def __init__(self, problem):
        self.k = 0
        
        # Get problem details
        self.prob = problem

        # Initialise optimization related parameters and functions
        self.cost_function = 0
        self.tau = 1

        # Set up optimisation variables
        self.x = cp.Variable((self.prob.n * (self.prob.horizon + 1), 1))
        self.u = cp.Variable((self.prob.p * self.prob.horizon, 1))

        # Initialise Lagrange multipliers to 1
        self.lam = cp.Parameter((self.prob.n * self.prob.horizon, 1))
        self.lam.value = np.ones((self.prob.n * self.prob.horizon, 1))

        # Define the constraints
        self.constraints = [
            self.x[0:self.prob.n] == self.prob.x0,
            self.x[-self.prob.n:,] == self.prob.xH,
            ]

        # Auxiliary variable to save previous x value
        self.x.value = np.zeros((self.prob.n * (self.prob.horizon + 1), 1))
        self.x_aux = np.zeros((self.prob.n * (self.prob.horizon + 1), 1))

        # Everything set, go to main loop
        self.main_loop()

    '''
    main_loop()
    Main loop of the algorithm that minimises the augmented Lagrangian,
    updates the variables, checks if constraints are met and updates
    the Lagrange multipliers.
    '''
    def main_loop(self):
        print('Creating cost function')
        # Update cost function with new Lagrange multipliers
        self.create_cost_function()
        
        print('Starting main loop')
        while True:
            self.x_aux = self.x.value            

            # Minimise to find new variable values
            self.minimise()

            # Update x and u
            self.update_variables()

            # Check if constraint are met, if so stop, otherwise update lambdas and continue
            if not self.check_constraints():
                self.update_lagrange_multipliers()
            else:
                break
            
            # Increase k
            self.increase_iteration()

    '''
    create_cost_function()
    Creates the cost function by summming the quadratic cost of each agents, the inner product 
    of the Lagrange multipliers with the constraints, and the squared norm of the constraints,
    everything over the MPC horizon.    
    '''
    def create_cost_function(self):
        # sum the quadratic cost for the input over the horizon 
        for ii in range(self.prob.horizon):
            self.cost_function += cp.quad_form(self.u[ii*self.prob.p:(ii+1)*self.prob.p], self.prob.MatCost)
            
        # sum the inner product of lambda with the constraints over the horizon
        for ii in range(self.prob.horizon):
            self.cost_function += self.lam[ii*self.prob.n:(ii+1)*self.prob.n].T @ (self.x[(ii+1)*self.prob.n:(ii+2)*self.prob.n] - self.prob.MatA @ self.x[ii*self.prob.n:(ii+1)*self.prob.n] -self.prob.MatB @ self.u[ii*self.prob.p:(ii+1)*self.prob.p])

        # # sum the squared norm of the constraints over the horizon
        for ii in range(self.prob.horizon):
            self.cost_function += (self.prob.rho/2) * cp.sum_squares(self.x[(ii+1)*self.prob.n:(ii+2)*self.prob.n] - self.prob.MatA @ self.x[ii*self.prob.n:(ii+1)*self.prob.n] - self.prob.MatB @ self.u[ii*self.prob.p:(ii+1)*self.prob.p])

    '''
    minimise()
    Solves the optimisation problem defined by the cost function and constraints.
    cp.Problem.solve() return the optimisation result value but we ignore it.
    '''
    def minimise(self):
        objective = cp.Minimize(self.cost_function)
        opt_prob = cp.Problem(objective, self.constraints)
        opt_prob.solve(warm_start=True)

    '''
    update_variables()
    Update the variables x and u according to equation (6) of Algorithm 1 to be used
    in the warm start of the optimiser
    '''
    def update_variables(self):
        self.x.value = self.x_aux + self.tau * (self.x.value - self.x_aux)

    '''
    check_constraints()
    Verify if the constraints are met to decide if the problem has been solved.
    '''
    def check_constraints(self):
        constraints_sum = 0
        
        # sum the constraints over the horizon
        for ii in range(self.prob.horizon):
            constraints_sum += self.x.value[(ii+1)*self.prob.n:(ii+2)*self.prob.n] - self.prob.MatA @ self.x.value[ii*self.prob.n:(ii+1)*self.prob.n] - self.prob.MatB @ self.u.value[ii*self.prob.p:(ii+1)*self.prob.p]

        if np.linalg.norm(constraints_sum) <= 10**(-8): # consider constraints are met
            return True
        else:
            print('Constraints error is ' + str(np.linalg.norm(constraints_sum)))
            return False
 
    '''
    update_lagrange_multipliers()
    Update the Lagrange multiplier parameters, according to equation (7) in Algorithm 1
    '''
    def update_lagrange_multipliers(self):

        for ii in range(self.prob.horizon):
            # sum the constraints over the horizon
            constraints_value = self.x.value[(ii+1)*self.prob.n:(ii+2)*self.prob.n] - self.prob.MatA @ self.x.value[ii*self.prob.n:(ii+1)*self.prob.n] - self.prob.MatB @ self.u.value[ii*self.prob.p:(ii+1)*self.prob.p]
        
            # update the Langrange multipliers
            self.lam.value[ii*self.prob.n:(ii+1)*self.prob.n] = self.lam.value[ii*self.prob.n:(ii+1)*self.prob.n] + self.prob.rho * self.tau * constraints_value

    '''
    increase_iterations()
    To keep track of the number of iterations until convergence
    '''
    def increase_iteration(self):
        self.k += 1


    