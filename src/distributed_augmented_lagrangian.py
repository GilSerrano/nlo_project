import enum
import cvxpy as cp
import numpy as np

class DistributedAugmentedLagrangian(object):
    def __init__(self, problem):
        self.k = 0
        
        # Get problem details
        self.prob = problem
        self.solution_value = [[] for ii in range(len(self.prob.agents))]
        self.solution_value_convergence = []

        # Initialise optimization related parameters and functions
        self.cost_functions = [0 for ii in range(len(self.prob.agents))]
        
        self.x = [[] for ii in range(len(self.prob.agents))]
        self.u = [[] for ii in range(len(self.prob.agents))]
        self.lam = [[] for ii in range(len(self.prob.agents))]
        
        self.constraints = [[] for ii in range(len(self.prob.agents))]
        self.constraints_norm  = np.zeros((len(self.prob.agents)*self.prob.horizon,))
        self.x_aux = [[] for ii in range(len(self.prob.agents))]
        self.u_aux = [[] for ii in range(len(self.prob.agents))]

        self.constraints_convergence = []
        self.max_constraints_convergence = []

        for ii, agent in enumerate(self.prob.agents):
            # Set up optimisation variables
            self.x[ii] = cp.Variable((agent.n, self.prob.horizon + 1))
            self.u[ii] = cp.Variable((agent.p, self.prob.horizon))

            # Initialise Lagrange multipliers to 1
            self.lam[ii] = cp.Parameter((agent.n, self.prob.horizon))
            self.lam[ii].value = np.ones((agent.n, self.prob.horizon))

            # Define the constraints
            self.constraints[ii] = [
                self.x[ii][:,0] == agent.x_0, # need to change this
                self.x[ii][:,-1] == agent.x_H,
                ]

            # Auxiliary variable to save previous x value
            self.x[ii].value = np.zeros((agent.n, self.prob.horizon + 1))
            self.x_aux[ii] = np.zeros((agent.n, self.prob.horizon + 1))
            self.u[ii].value = np.zeros((agent.p, self.prob.horizon))
            self.u_aux[ii] = np.zeros((agent.p, self.prob.horizon))

        # Everything set, go to main loop
        self.main_loop()

    '''
    main_loop()
    Main loop of the algorithm that minimises the augmented Lagrangian for every agent,
    updates the variables, checks if constraints are met and updates the Lagrange multipliers.
    '''
    def main_loop(self):
        print('Creating cost function')
        # Update cost function with new Lagrange multipliers
        self.create_cost_functions()
        
        print('Starting main loop')
        while True:
            for ii, _ in enumerate(self.prob.agents):          

                # Minimise to find new variable values
                self.minimise(ii)

                # Update x and u
                self.update_variables(ii)

            # Sum all agents solution at present iteration
            self.solution_value_convergence.append(sum(self.solution_value))

            # Check if constraint are met, if so stop, otherwise update lambdas and continue
            if not self.check_constraints():
                self.update_lagrange_multipliers()
            else:
                print('Optimal value is ' + str(self.solution_value))
                break
                
            # Increase k
            self.k += 1

    '''
    create_cost_functions()
    Creates the cost functions of every agent, according to the ADAL algorithm - see eq.(10)    
    '''
    def create_cost_functions(self):
        for ii, _ in enumerate(self.prob.agents):
            # sum the quadratic cost for the input over the horizon - 1st line of eq. (10)
            for tt in range(self.prob.horizon):
                self.cost_functions[ii] += cp.quad_form(self.u[ii][:,tt], self.prob.agents[ii].matCost)
                
            # sum the inner product of lambda with the constraints over the horizon - 2nd line of eq. (10)
            for tt in range(self.prob.horizon):
                self.cost_functions[ii] += self.lam[ii][:,tt].T @ (self.x[ii][:,tt+1] - self.prob.agents[ii].matA @ self.x[ii][:,tt] - self.prob.agents[ii].matB @ self.u[ii][:,tt])
                for jj in self.prob.agents[ii].out_neigh:
                    self.cost_functions[ii] -= self.lam[jj-1][:,tt].T @ (self.prob.agents[ii].outA[(jj,ii+1)] @ self.x[ii][:,tt] + self.prob.agents[ii].outB[(jj,ii+1)] @ self.u[ii][:,tt])

            # sum the squared norm of the constraints of the agent over the horizon - 3rd line of eq. (10)
            for tt in range(self.prob.horizon):
                cost_functions_aux = 0
                for jj in self.prob.agents[ii].in_neigh:
                    cost_functions_aux += (self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x_aux[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u_aux[jj-1][:,tt])
                
                self.cost_functions[ii] += (self.prob.rho/2) * cp.sum_squares(self.x[ii][:,tt+1] - self.prob.agents[ii].matA @ self.x[ii][:,tt] - self.prob.agents[ii].matB @ self.u[ii][:,tt] - cost_functions_aux)

            # sum the squared norm of dynamics of out-neighbourhood of j - 4th and 5th lines of eq. (10)
            for tt in range(self.prob.horizon):
                for jj in self.prob.agents[ii].out_neigh:
                    cost_functions_aux = 0
                    for mm in self.prob.agents[jj-1].in_neigh:
                        if mm == (ii+1): # agent m is the same as i - do not count it (see eq.10)
                            continue
                        cost_functions_aux += (self.prob.agents[mm-1].outA[(jj,mm)] @ self.x_aux[mm-1][:,tt] + self.prob.agents[mm-1].outB[(jj,mm)] @ self.u_aux[mm-1][:,tt]) # here was the bug - instead of +
                    self.cost_functions[ii] += (self.prob.rho/2) * cp.sum_squares(self.x_aux[jj-1][:,tt+1] - self.prob.agents[ii].outA[(jj,ii+1)] @ self.x[ii][:,tt] - self.prob.agents[ii].outB[(jj,ii+1)] @ self.u[ii][:,tt] - cost_functions_aux)


    '''
    minimise()
    Solves the optimisation problem defined by the cost function and constraints.
    cp.Problem.solve() return the optimisation result value but we ignore it.
    '''
    def minimise(self, ii):
        objective = cp.Minimize(self.cost_functions[ii])
        opt_prob = cp.Problem(objective, self.constraints[ii])
        opt_prob.solve(warm_start=True)
        self.solution_value[ii] = opt_prob.value

    '''
    update_variables()
    Update the variables x and u according to equation (6) of Algorithm 1 to be used
    in the warm start of the optimiser
    '''
    def update_variables(self, ii):
        self.x[ii].value = self.x_aux[ii] + self.prob.tau * (self.x[ii].value - self.x_aux[ii])
        self.u[ii].value = self.u_aux[ii] + self.prob.tau * (self.u[ii].value - self.u_aux[ii])
        self.x_aux[ii] = self.x[ii].value 
        self.u_aux[ii] = self.u[ii].value 

    '''
    check_constraints()
    Verify if the constraints are met to decide if the problem has been solved.
    '''
    def check_constraints(self):
        constraints_sum = 0
        constraints_norm_aux  = self.constraints_norm
        self.constraints_norm = np.array([])
        
        # sum the constraints over the horizon
        for ii in range(len(self.prob.agents)):
            for tt in range(self.prob.horizon):
                constraints_sum = self.x[ii].value[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].value[:,tt] - self.prob.agents[ii].matB @ self.u[ii].value[:,tt]
                for jj in self.prob.agents[ii].in_neigh:
                    constraints_sum -= self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x_aux[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u_aux[jj-1][:,tt]
                self.constraints_norm = np.append(self.constraints_norm, np.linalg.norm(constraints_sum))

        # Append constraints value (sum and max) at each iteration
        self.constraints_convergence.append(sum(self.constraints_norm))
        self.max_constraints_convergence.append(max(self.constraints_norm))

        if np.all(np.less_equal(self.constraints_norm, 10**(-4))): # consider constraints are met
            if np.all(np.less_equal(constraints_norm_aux - self.constraints_norm, 10**(-8))): # consider convergence
                print('Constraints met.')
                print('Norms of errors are ')
                print(self.constraints_norm)
                return True
        else:
            if self.k % 100 == 0:
                print("k = " + str(self.k))
                print('Norms of errors are ')
                print(self.constraints_norm)
            return False
    '''
    update_lagrange_multipliers()
    Update the Lagrange multiplier parameters, according to equation (7) in Algorithm 1
    '''
    def update_lagrange_multipliers(self):

        for ii in range(len(self.prob.agents)):
            for tt in range(self.prob.horizon):
                constraints_value = 0
                # sum the constraints at time tt
                constraints_value = self.x[ii].value[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].value[:,tt] - self.prob.agents[ii].matB @ self.u[ii].value[:,tt]
                for jj in self.prob.agents[ii].in_neigh:
                    constraints_value -= self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x_aux[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u_aux[jj-1][:,tt]

                # update the Langrange multipliers
                self.lam[ii].value[:,tt] = self.lam[ii].value[:,tt] + self.prob.rho * self.prob.tau * constraints_value