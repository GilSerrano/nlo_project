import cvxpy as cp
import problem

if __name__ == '__main__':
    
    file = 'mpc_3agents_noInteractions.yaml'
    
    prob = problem.Problem(file)
    
    x = cp.Variables((prob.n, 1))
    u = cp.Variables((prob.p, 1))
    
    
