import yaml
import agent
import numpy as np

class Problem(object):
    
    def __init__(self, file):
        self.agents = []
        self.n = 0
        self.p = 0
        self.n_i = []
        self.p_i = []
        self.cost_matrix = []
        
        self.loadProblem(file)
        
    '''
    Initialize the problem by reading the agents input file
    '''
    def loadProblem(self, file):
        with open(file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as error:
                print(error)        
        
        for key in data:
            self.agents.append(agent.Agent(data[key]))
            self.n_i.append(data[key]['n_i'])
            self.p_i.append(data[key]['p_i'])

        self.n = sum(self.n_i)
        self.p = sum(self.p_i)

    '''
    Check if matrices A and B are of correct size for each agent.
    If incogruences are found, exits the program.
    '''
    def checkProblem(self):
        flag = 0
        for agent in self.agents:
            if agent.matA.shape is not (agent.n, agent.n):
                flag =1
            elif agent.matB.shape is not (agent.n, agent.p):
                flag = 1
        
        if flag:
            try:
                raise SystemExit('Error in agent\'s input data. Exiting...')
            except:
                print('Program tried to exit, but still running.')



'''
Class for the Centralised version of the optimisation problem.
It inherits from the previously defined Problem class.
'''
class CentralisedProblem(Problem):
    def __init__(self,file):
        super().__init__(self, file)

        self.MatA = []
        self.MatB = []
        self.getMatrices()


    '''
    Concatenate all matrices A and B into large system matrices MatA and MatB.
    This function should only be used in the centralised version of the algorithm
    '''
    def getMatrices(self):
        # Initialize matrices to zero matrices of proper size
        self.MatA = np.zeros((self.n, self.n))
        self.MatB = np.zeros((self.n, self.p))
        
        # Go over every agent and load the big matrices with the interactions
        ii_aux = 0
        pp_aux = 0
        for ii, agent in enumerate(self.agents):
            jj_aux =0
            for jj in range(len(self.agents)):
                if ii == jj: # own dynamics
                    self.MatA[jj_aux:jj_aux+agent.n, ii_aux:ii_aux+agent.n] = agent.matA
                    self.MatB[jj_aux:jj_aux+agent.n, pp_aux:pp_aux+agent.p] = agent.matB

                elif jj+1 in agent.out_neigh:
                    self.MatA[jj_aux:jj_aux+agent.n, ii_aux:ii_aux+agent.n] = agent.outA[(jj+1,ii+1)]
                    self.MatB[jj_aux:jj_aux+agent.n, pp_aux:pp_aux+agent.p] = agent.outB[(jj+1,ii+1)]

                jj_aux += self.n_i[jj] # compensate state dimension in row number

            ii_aux += self.n_i[ii] # compensate state dimension in column number
            pp_aux += self.p_i[ii] # compensate input dimension in column number
