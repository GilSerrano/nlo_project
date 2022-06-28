import matplotlib.pyplot as plt
import numpy as np

def plot3Dagent(x, idx):
    
    t = np.arange(x.shape[1])

    # plot
    plt.subplot(3,1,1)
    plt.plot(t, x[0,:])
    plt.xlabel('time t')
    plt.ylabel('x-axis')
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(t, x[1,:])
    plt.xlabel('time t')
    plt.ylabel('y-axis')
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, x[2,:])
    plt.xlabel('time t')
    plt.ylabel('z-axis')
    plt.grid(True)
    
    plt.suptitle('Agent ' + str(idx))
    plt.show()


def plot2Dagent(x, idx):
    
    t = np.arange(x.shape[1])

    # plot
    plt.subplot(2,1,1)
    plt.plot(t, x[0,:])
    plt.xlabel('time t')
    plt.ylabel('x-axis')
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(t, x[1,:])
    plt.xlabel('time t')
    plt.ylabel('y-axis')
    plt.grid(True)

    plt.suptitle('Agent ' + str(idx))
    plt.show()

def plotNDagent(x, idx):
    
    N = x.shape[0]
    t = np.arange(x.shape[1])

    for ii in range(N):
        plt.subplot(N,1,ii+1)
        plt.plot(t, x[ii,:])
        plt.xlabel('time t')
        plt.ylabel('axis ' + str(ii+1))
        plt.grid(True)
        
    plt.suptitle('Agent ' + str(idx))
    plt.show()

def plot2DagentMap(x, idx):
    
    # plot
    plt.plot(x[0,:], x[1,:], 'b-')
    plt.plot(x[0,:], x[1,:], 'b*')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.suptitle('Agent ' + str(idx))
    plt.show()

def plot2Dagents(x, ylabel):
    # type is either "state" or "input"
    if ylabel not in {"state", "input"}:
        ylabel = "state"

    lgd = []
    for ii, xx in enumerate(x):
        lgd.append("agent" + str(ii+1))      
        plt.subplot(2,1,1)
        plt.plot(range(len(xx.value[0,:])), xx.value[0,:]) 
        plt.subplot(2,1,2)
        plt.plot(range(len(xx.value[1,:])), xx.value[1,:])

    plt.subplot(2,1,1)
    plt.legend(lgd)
    plt.xlabel("Time t")
    plt.ylabel(ylabel + " 1")
    plt.grid(True) 
    plt.subplot(2,1,2)
    plt.legend(lgd)
    plt.xlabel("Time t")
    plt.ylabel(ylabel + " 2")
    plt.grid(True)
    plt.show()

def plot2DagentsMap(x):
    
    lgd = []
    lgd.append("start")
    lgd.append("end")

    # Plot just the 1st agent's start and end due to legend reasons
    plt.plot(x[0].value[0,0], x[0].value[1,0], marker="o", color="k")
    plt.plot(x[0].value[0,-1], x[0].value[1,-1], marker="*", color="k")

    for ii, xx in enumerate(x):
        lgd.append("agent" + str(ii+1))   
        plt.plot(xx.value[0,:], xx.value[1,:], '-')
    
    for ii, xx in enumerate(x):
        plt.plot(xx.value[0,0], xx.value[1,0], marker="o", color="k")
        plt.plot(xx.value[0,-1], xx.value[1,-1], marker="*", color="k")
        plt.annotate("ag"+str(ii+1), (xx.value[0,0], xx.value[1,0]))
    
    plt.xlabel('state 1')
    plt.ylabel('state 2')
    plt.grid(True)
    plt.legend(lgd)
    plt.show()

def plot2DagentsMapMatrix(x):
    
    for xx in x:
        # plot
        plt.plot(xx[0,:], xx[1,:], '-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.suptitle('All agents')
    plt.show()

def plotConstraintsConvergence(constraints_convergence, plot_type='normal'):

    if plot_type == 'log':
        plt.plot(range(len(constraints_convergence)), np.log10(constraints_convergence))
        plt.ylabel('Log of Sum of Norms of Constraints Violation')
    else:
        plt.plot(range(len(constraints_convergence)), constraints_convergence)
        plt.ylabel('Sum of Norms of Constraints Violation')

    plt.xlabel('Iterations')
    plt.grid(True)
    plt.show()

def plotMaxConstraintsConvergence(constraints_convergence, plot_type='normal'):

    if plot_type == 'log':
        plt.plot(range(len(constraints_convergence)), np.log10(constraints_convergence))
        plt.ylabel('Log of Maximum Norm of Constraints Violation')
    else:
        plt.plot(range(len(constraints_convergence)), constraints_convergence)
        plt.ylabel('Maximum of Norm of Constraints Violation')

    plt.xlabel('Iterations')
    plt.grid(True)
    plt.show()

def plotObjectiveFunctionConvergence(objective_convergence):
    N = len(objective_convergence)
    plt.plot(range(N), objective_convergence)
    plt.plot(range(N), objective_convergence[-1]*np.ones((N,)))
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Convergence')
    plt.grid(True)
    plt.show()

def plotAgentNetwork(problem):
    
    for _, agent in enumerate(problem.agents):
        plt.plot(agent.x_0[0], agent.x_0[1], marker=".", color="tab:orange")
        plt.annotate("ag"+str(agent.idx), (agent.x_0[0], agent.x_0[1]))
        for neigh in agent.out_neigh:
            x = agent.x_0[0]
            y = agent.x_0[1]
            dx = (problem.agents[neigh-1].x_0[0] - agent.x_0[0])
            dy = (problem.agents[neigh-1].x_0[1] - agent.x_0[1])
            plt.arrow(x, y, dx, dy, head_width=0.05, length_includes_head=True, color="tab:blue")
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()