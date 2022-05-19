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
