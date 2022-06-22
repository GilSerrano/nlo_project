from cmath import cos
import cvxpy as cp
import numpy as np

class CentralisedAugmentedLagrangian(object):
    def __init__(self, problem):
        self.k = 0
        
        # Get problem details
        self.prob = problem
        self.solution_value = 0

        # Initialise optimization related parameters and functions
        self.cost_function = 0
        # self.constraints_sum = np.zeros((self.prob.n * self.prob.horizon, 1))
        self.constraints_sum = np.zeros((self.prob.n_i[0],))
        self.constraints_b = np.zeros((self.prob.n_i[0],))
        self.constraints_norm  = np.array([])

        # Set up optimisation variables
        self.x = [[] for ii in range(len(self.prob.agents))]
        self.u = [[] for ii in range(len(self.prob.agents))]
        self.lam = [[] for ii in range(len(self.prob.agents))]
        
        # Define the constraints
        self.constraints = []

        # Auxiliary variable to save previous x value
        self.x_aux = [[] for ii in range(len(self.prob.agents))]
        self.u_aux = [[] for ii in range(len(self.prob.agents))]
        
        for ii, agent in enumerate(self.prob.agents):
            # Set up optimisation variables
            self.x[ii] = cp.Variable((agent.n, self.prob.horizon + 1))
            self.u[ii] = cp.Variable((agent.p, self.prob.horizon))

            # Initialise Lagrange multipliers to 1
            self.lam[ii] = cp.Parameter((agent.n, self.prob.horizon))
            self.lam[ii].value = np.ones((agent.n, self.prob.horizon))

            # Define the constraints
            self.constraints.extend([
                self.x[ii][:,0] == agent.x_0,
                self.x[ii][:,-1] == agent.x_H,
                ])
            self.constraints_b += agent.x_0 - agent.x_H

            # Auxiliary variable to save previous x value
            self.x[ii].value = np.zeros((agent.n, self.prob.horizon + 1))
            self.x_aux[ii] = np.zeros((agent.n, self.prob.horizon + 1))
            self.u[ii].value = np.zeros((agent.n, self.prob.horizon))
            self.u_aux[ii] = np.zeros((agent.p, self.prob.horizon))

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

            # Minimise to find new variable values
            self.minimise()

            # Update x and u
            self.update_variables()

            # Check if constraint are met, if so stop, otherwise update lambdas and continue
            if not self.check_constraints():
                self.update_lagrange_multipliers()
            else:
                print('Optimal value is ' + str(self.solution_value))
                break
            
            # Increase k
            self.k += 1
    '''
    create_cost_function()
    Creates the cost function by summming the quadratic cost of each agents, the inner product 
    of the Lagrange multipliers with the constraints, and the squared norm of the constraints,
    everything over the MPC horizon.    
    '''
    def create_cost_function(self):
        cost_function_aux = 0
        for ii, _ in enumerate(self.prob.agents):
            # sum the quadratic cost for the input over the horizon for each agent
            for tt in range(self.prob.horizon):
                self.cost_function += cp.quad_form(self.u[ii][:,tt], self.prob.agents[ii].matCost)
        
        # sum the constraints over the horizon
        # for ii, agent in enumerate(self.prob.agents):
        #     for tt in range(self.prob.horizon):
        #         Ai = (np.eye(agent.n) - agent.matA)
        #         Bi = -agent.matB
        #         for jj in agent.out_neigh:
        #             Ai += -agent.outA[(jj,agent.idx)]
        #             Bi += -agent.outB[(jj,agent.idx)]
        #         cost_function_aux += (Ai) @ (self.x[ii][:,tt]) + (Bi) @ (self.u[ii][:,tt])
        
        # cost_function_aux -= np.reshape(self.constraints_b, (2,))

        #self.cost_function += self.lam.T @ (cost_function_aux)

        for ii, agent in enumerate(self.prob.agents):
            for tt in range(self.prob.horizon):
                cost_function_aux = self.x[ii][:,tt+1] - (agent.matA @ self.x[ii][:,tt] + agent.matB @ self.u[ii][:,tt])
                for jj in agent.in_neigh:
                    cost_function_aux -= self.prob.agents[jj-1].outA[(ii+1, jj)] @ self.x[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1, jj)] @ self.u[jj-1][:,tt]
                
                # Inner product with Lagrange multipliers
                self.cost_function += self.lam[ii][:,tt].T @ (cost_function_aux)

                # Quadratic penalization term
                self.cost_function += (self.prob.rho/2) * cp.sum_squares(cost_function_aux)

    # '''
    # create_cost_function()
    # Creates the cost function by summming the quadratic cost of each agents, the inner product 
    # of the Lagrange multipliers with the constraints, and the squared norm of the constraints,
    # everything over the MPC horizon.    
    # '''
    # def create_cost_function(self):
    #     for ii, _ in enumerate(self.prob.agents):
    #         # sum the quadratic cost for the input over the horizon for each agent
    #         for tt in range(self.prob.horizon):
    #             self.cost_function += cp.quad_form(self.u[ii][:,tt], self.prob.agents[ii].matCost)
        
    #         # inner product with Lagrangian multipliers
    #         for tt in range(self.prob.horizon):
    #             cost_function_aux = 0
    #             for jj in self.prob.agents[ii].in_neigh:
    #                 cost_function_aux += (self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u[jj-1][:,tt])
                
    #             self.cost_function += self.lam[ii][:,tt].T @ (self.x[ii][:,tt+1] - self.prob.agents[ii].matA @ self.x[ii][:,tt] - self.prob.agents[ii].matB @ self.u[ii][:,tt] - cost_function_aux)

    #         # quadratic penalization term
    #         for tt in range(self.prob.horizon):
    #             cost_function_aux = 0
    #             for jj in self.prob.agents[ii].in_neigh:
    #                 cost_function_aux += (self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u[jj-1][:,tt])
                
    #             self.cost_function += (self.prob.rho/2) * cp.sum_squares(self.x[ii][:,tt+1] - self.prob.agents[ii].matA @ self.x[ii][:,tt] - self.prob.agents[ii].matB @ self.u[ii][:,tt] - cost_function_aux)

    '''
    minimise()
    Solves the optimisation problem defined by the cost function and constraints.
    cp.Problem.solve() return the optimisation result value but we ignore it.
    '''
    def minimise(self):
        objective = cp.Minimize(self.cost_function)
        opt_prob = cp.Problem(objective, self.constraints)
        opt_prob.solve(warm_start=True)
        self.solution_value = opt_prob.value

    '''
    update_variables()
    Update the variables x and u according to equation (6) of Algorithm 1 to be used
    in the warm start of the optimiser
    '''
    def update_variables(self):
        for ii in range(len(self.prob.agents)):
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
        constraints_norm_aux = self.constraints_norm
        self.constraints_norm  = np.array([])
        
        # sum the constraints over the horizon
        for ii in range(len(self.prob.agents)):
            for tt in range(self.prob.horizon):
                constraints_sum = self.x[ii].value[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].value[:,tt] - self.prob.agents[ii].matB @ self.u[ii].value[:,tt]
                for jj in self.prob.agents[ii].in_neigh:
                    constraints_sum -= self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x_aux[jj-1][:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u_aux[jj-1][:,tt]
                self.constraints_norm = np.append(self.constraints_norm, np.linalg.norm(constraints_sum))

        if np.all(np.less_equal(self.constraints_norm, 10**(-4))): # consider constraints are met
            if np.all(np.less_equal(constraints_norm_aux - self.constraints_norm, 10**(-8))): # consider convergence
                return True
        else:
            if self.k % 100 == 0:
                print("k = " + str(self.k))
                print('Norms of errors are ')
                print(self.constraints_norm)
            return False

        '''
        aux_constraints_sum = self.constraints_sum
        self.constraints_sum = np.zeros((self.prob.n_i[0], 1)).T

        # sum the constraints over the horizon
        for ii, agent in enumerate(self.prob.agents):
            for tt in range(self.prob.horizon):
                Ai = (np.eye(agent.n) - agent.matA)
                Bi = -agent.matB
                for jj in agent.out_neigh:
                    Ai += -agent.outA[(jj,agent.idx)]
                    Bi += -agent.outB[(jj,agent.idx)]
                self.constraints_sum += Ai @ self.x[ii].value[:,tt] + Bi @ self.u[ii].value[:,tt]


        if np.linalg.norm(self.constraints_sum - self.constraints_b) <= 10**(-4): # consider constraints are met
            return True
        elif np.linalg.norm(self.constraints_sum - aux_constraints_sum) <= 10**(-4):
            print('Converged but could not find a solution that met the constraints.')
            return True
        else:
            print('Constraints error is ' + str(np.linalg.norm(self.constraints_sum)))
            return False
    '''

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

        '''
        lam_constraints_aux = 0
        for ii, agent in enumerate(self.prob.agents):
            for tt in range(self.prob.horizon):
                Ai = (np.eye(agent.n) - agent.matA)
                Bi = -agent.matB
                for jj in agent.out_neigh:
                    Ai += -agent.outA[(jj,agent.idx)]
                    Bi += -agent.outB[(jj,agent.idx)]
                lam_constraints_aux += (Ai) @ (self.x[ii].value[:,tt]) + (Bi) @ (self.u[ii].value[:,tt])
        
        lam_constraints_aux -= np.reshape(self.constraints_b, (2,))
        self.lam.value = self.lam.value + self.prob.rho * self.prob.tau * lam_constraints_aux
        '''
        # for ii in range(len(self.prob.agents)):
        #     for tt in range(self.prob.horizon):
        #         constraints_value = np.array([])
        #         # sum the constraints at time tt
        #         constraints_value = self.x[ii].value[:,tt+1] - self.prob.agents[ii].matA @ self.x[ii].value[:,tt] - self.prob.agents[ii].matB @ self.u[ii].value[:,tt]
        #         for jj in self.prob.agents[ii].in_neigh:
        #             constraints_value -= self.prob.agents[jj-1].outA[(ii+1,jj)] @ self.x[jj-1].value[:,tt] + self.prob.agents[jj-1].outB[(ii+1,jj)] @ self.u[jj-1].value[:,tt]

        #         # update the Langrange multipliers
        #         self.lam[ii].value[:,tt] = self.lam[ii].value[:,tt] + self.prob.rho * self.prob.tau * constraints_value
 