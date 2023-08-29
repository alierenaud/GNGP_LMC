

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:12:40 2023

@author: alier
"""

import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix


from noisyLMC_interweaved import A_move_slice
from LMC_inference import phis_move
from LMC_mean import mu_move
from noisyLMC_inference import V_move_conj
from noisyLMC_inference import taus_move


random.seed(2)


### global parameters
n = 1000
p = 2


### generate random example
locs = random.uniform(0,1,(n,2))


mu = np.array([-0.5,0.5])
A = np.array([[1.,0.5],
              [-1,0.5]])
phis = np.array([5.,25.])
taus_sqrt_inv = np.array([1.,2.]) * 0.1


Y, V_true = rNLMC(A,phis,taus_sqrt_inv,locs, retV=True) 

Y += np.outer(mu,np.ones(n))
V_true += np.outer(mu,np.ones(n))

### priors
sigma_A = 1.
mu_A = np.zeros((p,p))

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.ones(p)
betas = np.ones(p)

a = 5
b = 0.1



### useful quantities 

D = distance_matrix(locs,locs)


### init and current state

V_current = random.normal(size=(p,n))
VmY_current = V_current - Y
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])

mu_current = np.array([0.,0.])
Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))

A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current

phis_current = np.array([10.,10.])
Rs_current = np.array([ np.exp(-D*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


taus_current = np.array([50.,50.])
Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y




### proposals


phis_prop = np.ones(p)*0.5
sigma_slice = 4




### samples

N = 1000
tail = 500

### global run containers
mu_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))



### acc vector

acc_phis = np.zeros((p,N))


import time
st = time.time()


for i in range(N):
    
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current, Vmmu1_current, mu_current)
        
    
    
    
    mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current)

    A_current, A_inv_current, A_invVmmu1_current = A_move_slice(A_current, A_invVmmu1_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    

    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,D,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
    
    mu_run[i] = mu_current
    A_run[i] = A_current
    phis_run[i] =  phis_current
    taus_run[i] = taus_current
    
    
    if i % 100 == 0:
        print(i)

et = time.time()
print('Execution time:', (et-st)/60, 'minutes')



print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))


plt.plot(phis_run[:,0])
plt.plot(phis_run[:,1])
plt.show()


plt.plot(mu_run[:,0])
plt.plot(mu_run[:,1])
plt.show()

plt.plot(taus_run[:,0])
plt.plot(taus_run[:,1])
plt.show()


print("Posterior Marginal Variance", np.mean([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)],axis=0))
print("True Marginal Variance", A@np.transpose(A))












