# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:07:50 2023

@author: alier
"""

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from LMC_generation import rLMC

def mult(z):
    if np.max(z)<0:
        return(0)
    else:
        return(np.argmax(z)+1)



def rmultiLMC(A, phis, mu, locs, retZV=False):
    
    p = A.shape[0]
    n = locs.shape[0]
    
    V = rLMC(A, phis, locs)  + np.outer(mu,np.ones(n))
    
    Z = V + random.normal(size=(p,n))
    
    Y = np.zeros(n)
    
    for i in range(n):
        Y[i] = mult(Z[:,i])
    
    if retZV:
        return(Y,Z,V)
    else:
        return(Y)
    
    
    
### showcase example


# #### create grid

# locs = np.linspace(0, 1, 101)

# A = np.array([[-1,1],
#               [1,1]])
# phis = np.array([2,32])
# mu = np.array([-0.5,0.5])



# Y,Z,V = rmultiLMC(A,phis,mu,np.transpose([locs]),True)
# plt.plot(locs,V[0],locs,V[1],locs,Y,locs,Z[0],locs,Z[1])

