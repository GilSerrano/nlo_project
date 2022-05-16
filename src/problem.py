import yaml
import agent
import numpy as np

class Problem(object):
    
    def __init__(self, file):
        self.agents = []
        self.n = 0
        self.p = 0
        self.cost_matrix = []
        self.MatA = []
        self.MatB = []
        
        self.loadProblem(file)
        
        self.getMatrices()
        

    def loadProblem(self, file):
        with open("C:/Users/gil97/Documents/Repositories/nlo_project/config/"+file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as error:
                print(error)        
        
        for key in data:
            self.agents.append(agent.Agent(data[key]))
            self.n += data[key]['n_i']
            self.p += data[key]['p_i']
    
    def getMatrices(self):
        pass
        '''
        get matrices A B from dimensions ni pi
        initialize to 0 
        fill them according to neighbourhood graphs
        '''
        