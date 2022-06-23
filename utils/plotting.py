import matplotlib.pyplot as plt
import numpy as np

def plot3Dagent(x, idx):
    
    t = np.arange(x.shape[1])

    # plot
    plt.subplot(3,1,1)
    plt.plot(t, x[0,:])
    plt.xlabel('time t')
    plt.ylabel('x-axis')
    #plt.title('title')
    plt.grid(True)
    
    plt.subplot(3,1,2)
    plt.plot(t, x[1,:])
    plt.xlabel('time t')
    plt.ylabel('y-axis')
    #plt.title('title')
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, x[2,:])
    plt.xlabel('time t')
    plt.ylabel('z-axis')
    #plt.title('title')
    plt.grid(True)
    
    plt.suptitle('Agent ' + str(idx))
    #plt.tight_layout()
    plt.show()


def plot2Dagent(x, idx):
    
    t = np.arange(x.shape[1])

    # plot
    plt.subplot(2,1,1)
    plt.plot(t, x[0,:])
    plt.xlabel('time t')
    plt.ylabel('x-axis')
    #plt.title('title')
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(t, x[1,:])
    plt.xlabel('time t')
    plt.ylabel('y-axis')
    #plt.title('title')
    plt.grid(True)

    plt.suptitle('Agent ' + str(idx))
    #plt.tight_layout()
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
    #plt.title('title')
    plt.grid(True)

    plt.suptitle('Agent ' + str(idx))
    #plt.tight_layout()
    plt.show()

def plot2DagentsMap(x):
    
    for xx in x:
        # plot
        plt.plot(xx.value[0,:], xx.value[1,:], '-')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.title('title')
        plt.grid(True)

    plt.suptitle('All agents')
    #plt.tight_layout()
    plt.show()

def plot2DagentsMapMatrix(x):
    
    for xx in x:
        # plot
        plt.plot(xx[0,:], xx[1,:], '-')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('title')
    plt.grid(True)

    plt.suptitle('All agents')
    #plt.tight_layout()
    plt.show()