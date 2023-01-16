




import numpy as np
from numpy import random
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

    
### LMC


### exp correlation function
def rLMC(A, phis, locs):
    
    p = A.shape[0]
    n = locs.shape[0]
    
    D = distance_matrix(locs,locs)
    
    Rs = np.array([ np.exp(-D*phis[j]) for j in range(p) ])
    Zs = np.array([np.matmul( np.linalg.cholesky(Rs[j]), random.normal(size=n) ) for j in range(p)])
    
    Vs = np.matmul(A,Zs) 
        
    return( Vs )



#### target example
n = 1000

locs = random.uniform(0,1,(n,2))

A = np.array([[-1,1,1,1,1],
              [1,-1,1,1,1],
              [1,1,-1,1,1],
              [1,1,1,-1,1],
              [1,1,1,1,-1]])
phis = np.array([2,4,8,16,32])


V = rLMC(A,phis,locs)
####



#### showcase example


#### create grid

locs = np.linspace(0, 1, 1001)

A = np.array([[-1,1],
              [1,1]])
phis = np.array([2,32])


V = rLMC(A,phis,np.transpose([locs]))
####

plt.plot(locs,V[0],locs,V[1])
plt.show()













