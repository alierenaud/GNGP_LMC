# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:12:40 2023

@author: alier
"""

import numpy as np
from numpy import random

from LMC_generation import rLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix


from noisyLMC_interweaved import A_move_slice
from LMC_inference import phis_move


def mu_move(A_inv_current,Rs_inv_current,V_current):
    
    n = V_current.shape[1]
    p = V_current.shape[0]
    
    M = np.sum([np.transpose([A_inv_current[j]])@np.ones((1,n))@Rs_inv_current[j]@np.ones((n,1))@np.array([A_inv_current[j]]) for j in range(p)],axis=0)
    b = np.sum([np.transpose([A_inv_current[j]])@np.ones((1,n))@Rs_inv_current[j]@np.transpose(V_current)@np.transpose([A_inv_current[j]]) for j in range(p)],axis=0)
    
    M_inv = np.linalg.inv(M)
    
    mu_current = np.linalg.cholesky(M_inv)@random.normal(size=p) + np.transpose(M_inv@b)[0]
    
    Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    return(mu_current, Vmmu1_current, A_invVmmu1_current)

# ### global parameters
# n = 1000
# p = 2


# ### generate random example
# locs = random.uniform(0,1,(n,2))


# mu = np.array([0,2])
# A = np.array([[1.,0.5],
#               [-1,0.5]])
# phis = np.array([5.,25.])


# V = rLMC(A,phis,locs) + np.outer(mu,np.ones(n))



# ### priors
# sigma_A = 1.
# mu_A = np.zeros((p,p))

# min_phi = 3.
# max_phi = 30.
# range_phi = max_phi - min_phi


# alphas = np.ones(p)
# betas = np.ones(p)



# ### useful quantities 

# D = distance_matrix(locs,locs)


# ### init and current state


# mu_current = np.array([0.,0.])
# Vmmu1_current = V - np.outer(mu_current,np.ones(n))

# A_current = np.identity(p)
# A_inv_current = np.linalg.inv(A_current)
# A_invVmmu1_current = A_inv_current @ Vmmu1_current

# phis_current = np.array([10.,10.])
# Rs_current = np.array([ np.exp(-D*phis_current[j]) for j in range(p) ])
# Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])







# ### proposals


# phis_prop = np.ones(p)*1.0
# sigma_slice = 4




# ### samples

# N = 2000
# tail = 1000


# ### global run containers
# mu_run = np.zeros((N,p))
# A_run = np.zeros((N,p,p))
# phis_run = np.zeros((N,p))



# ### acc vector

# acc_phis = np.zeros((p,N))


# import time
# st = time.time()


# for i in range(N):
    
    
    

    
    
    
#     mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V)

#     A_current, A_inv_current, A_invVmmu1_current = A_move_slice(A_current, A_invVmmu1_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    

#     phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,D,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    
#     mu_run[i] = mu_current
#     A_run[i] = A_current
#     phis_run[i] =  phis_current
    
    
#     if i % 100 == 0:
#         print(i)

# et = time.time()
# print('Execution time:', (et-st)/60, 'minutes')



# print('accept phi_1:',np.mean(acc_phis[0,tail:]))
# print('accept phi_2:',np.mean(acc_phis[1,tail:]))


# plt.plot(phis_run[tail:,0])
# plt.plot(phis_run[tail:,1])
# plt.show()


# plt.plot(mu_run[tail:,0])
# plt.plot(mu_run[tail:,1])
# plt.show()


# print("Posterior Marginal Variance", np.mean([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)],axis=0))
# print("True Marginal Variance", A@np.transpose(A))












