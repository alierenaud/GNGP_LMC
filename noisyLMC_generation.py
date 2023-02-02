
import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from LMC_generation import rLMC





def rNLMC(A, phis, taus, locs):
    
    p = A.shape[0]
    n = locs.shape[0]
    
    V = rLMC(A, phis, locs) 
    
    Y = V + random.normal(size=(p,n))*np.outer(taus,np.ones(n))
    
    
    return(Y)


#### showcase example


#### create grid

locs = np.linspace(0, 1, 101)

A = np.array([[-1,1],
              [1,1]])
phis = np.array([2,32])
taus = np.array([1,4])*0.1


Y = rNLMC(A,phis,taus,np.transpose([locs]))
####

plt.plot(locs,Y[0],locs,Y[1])
plt.show()







