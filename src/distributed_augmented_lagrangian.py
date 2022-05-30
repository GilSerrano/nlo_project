import enum
import cvxpy as cp
import numpy as np

class DistributedAugmentedLagrangian(object):
    def __init__(self, problem):
        self.k = 0
        
        # Get problem details
        self.prob = problem

        # Initialise optimization related parameters and functions
        self.cost_functions = [0 for ii in range(len(self.prob.agents))]
        
        self.x = [[] for ii in range(len(self.prob.agents))]
        self.u = [[] for ii in range(len(self.prob.agents))]
        self.lam = [[] for ii in range(len(self.prob.agents))]
        
        self.constraints = [[] for ii in range(len(self.prob.agents))]
        self.x_aux = [[] for ii in range(len(self.prob.agents))]
        self.u_aux = [[] for ii in range(len(self.prob.agents))]

        self.constraints_sum = 0
        self.tau = 1
        


        for ii, agent in enumerate(self.prob.agents):
            # Set up optimisation variables
            self.x[ii] = cp.Variable((agent.n_i, self.prob.horizon + 1))
            self.u[ii] = cp.Variable((agent.p_i, self.prob.horizon))

            # Initialise Lagrange multipliers to 1
            self.lam[ii] = cp.Parameter((agent.n_i, self.prob.horizon))
            self.lam[ii].value = np.ones((agent.n_i, self.prob.horizon))

            # Define the constraints
            self.constraints[ii] = [
                self.x[0:self.prob.n] == self.prob.x0, # need to change this
                self.x[-self.prob.n:,] == self.prob.xH,
                ]

            # Auxiliary variable to save previous x value
            self.x[ii].value = np.zeros((agent.n_i, self.prob.horizon + 1))
            self.x_aux[ii] = np.zeros((agent.n_i, self.prob.horizon + 1))
            self.u_aux[ii] = np.zeros((agent.p_i, self.prob.horizon))

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
                self.x_aux[ii] = self.x[ii].value 
                self.u_aux[ii] = self.u[ii].value           

                # Minimise to find new variable values
                self.minimise(ii)

                # Update x and u
                self.update_variables(ii)

            # Check if constraint are met, if so stop, otherwise update lambdas and continue
            if not self.check_constraints():
                self.update_lagrange_multipliers()
            else:
                break
                
            # Increase k
            self.increase_iteration()

    '''
    calculate_tau
    Get the value of tau, dependent on the interactions between agents
    '''
    def calculate_tau(self):
        pass

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
                for kk, jj in enumerate(self.prob.agents[ii].out_neigh):
                    self.cost_functions[ii] -= self.lam[jj][:,tt].T @ (self.prob.agents[ii].matA_inter[kk] @ self.x[ii][:,tt] + self.prob.agents[ii].matB_inter[kk] @ self.u[ii][:,tt])

            # sum the squared norm of the constraints of the agent over the horizon - 3rd line of eq. (10)
            for tt in range(self.prob.horizon):
                cost_functions_aux = 0
                for jj in self.prob.agents[ii].in_neigh:
                    agent_ii_idx = list(self.prob.agents[jj].out_neigh).index(self.prob.agents[ii].idx) - 1 # get the idx of agent i in agent's j out-neighbourhood
                    cost_functions_aux += (self.prob.agents[jj].matA_inter[agent_ii_idx] @ self.x_aux[jj][:,tt] + self.prob.agents[jj].matB_inter[agent_ii_idx] @ self.u_aux[jj][:,tt])
                
                self.cost_function[ii] += (self.prob.rho/2) * cp.sum_squares(self.x[ii][:,tt+1] - self.prob.agents[ii].matA @ self.x[ii][:,tt] - self.prob.agents[ii].matB @ self.u[ii][:,tt] - cost_functions_aux)

            # sum the squared norm of dynamics of out-neighbourhood of j - 4th and 5th lines of eq. (10)
            for tt in range(self.prob.horizon):
                for kk, jj in enumerate(self.prob.agents[ii].out_neigh):
                    cost_functions_aux = 0
                    for mm in self.prob.agents[jj].in_neigh:
                        agent_jj_idx = list(self.prob.agents[mm].out_neigh).index(self.prob.agents[jj].idx) - 1
                        cost_functions_aux += (self.prob.agents[mm].matA_inter[agent_jj_idx] @ self.x_aux[mm][:,tt] - self.prob.agents[mm].matB_inter[agent_jj_idx] @ self.u_aux[mm][:,tt]) 
                    self.cost_function[ii] += (self.prob.rho/2) * cp.sum_squares(self.x_aux[jj][:,tt+1] - self.prob.agents[ii].matA_inter[kk] @ self.x[ii][:,tt] - self.prob.agents[ii].matB_inter[kk] @ self.u[ii][:,tt] - cost_functions_aux)


    '''
    minimise()
    Solves the optimisation problem defined by the cost function and constraints.
    cp.Problem.solve() return the optimisation result value but we ignore it.
    '''
    def minimise(self, ii):
        objective = cp.Minimize(self.cost_functions[ii])
        opt_prob = cp.Problem(objective, self.constraints[ii])
        opt_prob.solve(warm_start=True)

    '''
    update_variables()
    Update the variables x and u according to equation (6) of Algorithm 1 to be used
    in the warm start of the optimiser
    '''
    def update_variables(self, ii):
        self.x[ii].value = self.x_aux[ii] + self.tau * (self.x[ii].value - self.x_aux[ii])

    '''
    check_constraints()
    Verify if the constraints are met to decide if the problem has been solved.
    '''
    def check_constraints(self):
        aux_constraints_sum = self.constraints_sum
        self.constraints_sum = 0
        
        # sum the constraints over the horizon
        for ii in range(len(self.prob.agents)):
            for tt in range(self.prob.horizon):
                self.constraints_sum += self.x[ii].values[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].values[:,tt] - self.prob.agents[ii].matB @ self.u[ii].values[:,tt]
                for kk, _ in enumerate(self.prob.agents[ii].out_neigh):
                    self.constraints_sum -= self.prob.agents[ii].matA_inter[kk] @ self.x[ii].values[:,tt] + self.prob.agents[ii].matB_inter[kk] @ self.u[ii].values[:,tt]

        if np.linalg.norm(self.constraints_sum) <= 10**(-8): # consider constraints are met
            return True
        elif np.linalg.norm(self.constraints_sum - aux_constraints_sum) <= 10**(-8):
            print('Converged but could not find a solution that met the constraints.')
            return True
        else:
            print('Constraints error is ' + str(np.linalg.norm(self.constraints_sum)))
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
                constraints_value = self.x[ii].values[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].values[:,tt] - self.prob.agents[ii].matB @ self.u[ii].values[:,tt]
                for kk, _ in enumerate(self.prob.agents[ii].out_neigh):
                    constraints_value -= self.prob.agents[ii].matA_inter[kk] @ self.x[ii].values[:,tt] + self.prob.agents[ii].matB_inter[kk] @ self.u[ii].values[:,tt]

                # update the Langrange multipliers
                self.lam[ii].value[:,tt] = self.lam[ii].value[:,tt] + self.prob.rho * self.tau * constraints_value

    '''
    increase_iterations()
    To keep track of the number of iterations until convergence
    '''
    def increase_iteration(self):
        self.k += 1