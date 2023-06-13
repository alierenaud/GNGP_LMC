# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:53:27 2023

@author: alier
"""

import numpy as np
from numpy import random

from LMC_generation import rLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix


from noisyLMC_interweaved import A_move_slice



from LMC_inference import phis_move


def A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice):

    
    p = A_current.shape[0] 
    n = A_invV_current.shape[1]
    
    ### threshold
    z =  -1/2 * np.sum( [A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_current))) - 1/2/sigma_A**2 * np.sum((A_current-mu_A)**2) - random.exponential(1,1)
    
    L = A_current - random.uniform(0,sigma_slice,(p,p))
    # L[0] = np.maximum(L[0],0)
    
    U = L + sigma_slice
    
    L *= A_mask_current
    U *= A_mask_current
        
    while True:
    
        
        
        A_prop = random.uniform(L,U)
        A_inv_prop = np.linalg.inv(A_prop)
        A_invV_prop = A_inv_prop @ V_current
        
        acc = z < -1/2 * np.sum( [A_invV_prop[j] @ Rs_inv_current[j] @ A_invV_prop[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_prop))) - 1/2/sigma_A**2 * np.sum((A_prop-mu_A)**2) 
            
        if acc:
            return(A_prop,A_inv_prop,A_invV_prop)
        else:
            for ii in range(p):
                for jj in range(p):
                    if A_prop[ii,jj] < A_current[ii,jj]:
                        L[ii,jj] = A_prop[ii,jj]
                    else:
                        U[ii,jj] = A_prop[ii,jj]

random.seed(0)

### number of points 
n_obs=400
n_grid=1000

### global parameters



### generate random example
loc_obs = random.uniform(0,1,(n_obs,1))
loc_grid = np.transpose([np.linspace(0, 1, n_grid+1)])

locs = np.concatenate((loc_obs,loc_grid), axis=0)


### parameters
A = np.array([[5,0],
              [0,4]])
p = A.shape[0]
phis = np.array([5.,20.])



### generate rfs

V = rLMC(A,phis,locs)

V_obs = V[:,:n_obs]
V_grid = V[:,n_obs:]


### showcase

plt.plot(loc_grid,V_grid[0],loc_grid,V_grid[1])
plt.show()
plt.plot(loc_obs,V_obs[0],'o',c="tab:blue",markersize=2)
plt.plot(loc_obs,V_obs[1],'o',c="tab:orange",markersize=2)
plt.show()



### priors
sigma_A = 10.
mu_A = np.array([[0.,0.],
                  [0.,0.]])
# mu_A = np.array([[0.,0.,0.],
#                  [0.,0.,0.],
#                  [0.,0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi

alphas = np.ones(p)
betas = np.ones(p)

### RJMCMC

prob_one = 0.5

### RJMCMC

RJMCMC = False

n_ones_current = p**2
A_mask_current = np.ones((p,p))

def pairs(p):
    a = []
    k = 0
    for i in range(p):
        for j in range(p):
            a.append((i,j))
            k += 1
    
    return(a)
        
A_ones_ind_current = pairs(p)
A_zeros_ind_current = []


# ### proposals


phis_prop = np.ones(p)*1.5
sigma_slice = 10


def b(n_ones,p):
    
    if n_ones == p**2:
        return(0)
    elif n_ones == p:
        return(1)
    else:
        return(0.5)

n_jumps = p


### samples
N = 5000
tail = 1000

### global run containers
phis_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,n_grid+1))


### acc vector
acc_phis = np.zeros((p,N))

### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)
Dists_obs_grid = distance_matrix(loc_obs,loc_grid)
Dists_grid = distance_matrix(loc_grid,loc_grid)


### init and current state
phis_current = np.repeat(10.,p)
Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])




A_current = random.normal(size=(p,p))
A_inv_current = np.linalg.inv(A_current)

A_invV_current = A_inv_current @ V_obs

import time

st = time.time()

for i in range(N):
    
    
    
    
                        
    if RJMCMC:
        
        A_current, A_inv_current, A_invV_current = A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_obs, sigma_A, mu_A, sigma_slice)
    
    else:
        
        A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_obs, sigma_A, mu_A, sigma_slice)
    
        
        
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_obs,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)

    
    if RJMCMC:
        ### reversible jumps
        
        for j in range(n_jumps):
            
            
            insert = random.binomial(1, b(n_ones_current,p))
            
            if insert:
                
                rand_int = random.choice(range(p**2 - n_ones_current))
                rand_ind = A_zeros_ind_current[rand_int]
                new_elem = random.normal(mu_A[rand_ind],sigma_A,1)
                
                A_new = np.copy(A_current)
                A_new[rand_ind] = new_elem
                
                A_inv_new = np.linalg.inv(A_new)
                
                A_invV_new = A_inv_new @ V_obs
                
                rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n_obs * (1-b(n_ones_current+1,p))/b(n_ones_current,p) * (p**2 - n_ones_current)/(n_ones_current + 1) * prob_one / (1-prob_one)
                
                if random.uniform() < rat:
                    
                    A_current = A_new
                    A_inv_current = A_inv_new
                    A_invV_current = A_invV_new
                    
                    n_ones_current += 1
                    
                    A_mask_current[rand_ind] = 1.
                    
                    A_ones_ind_current.append(A_zeros_ind_current.pop(rand_int))
                    
                
                
            else:
                
                rand_int = random.choice(range(n_ones_current))
                rand_ind = A_ones_ind_current[rand_int]
                
                A_new = np.copy(A_current)
                A_new[rand_ind] = 0.
                
                if np.linalg.det(A_new) != 0:
                    A_inv_new = np.linalg.inv(A_new)
                    
                    A_invV_new = A_inv_new @ V_obs
                    
                    rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n_obs * b(n_ones_current-1,p)/(1-b(n_ones_current,p)) * (n_ones_current)/(p**2 - n_ones_current + 1) * (1-prob_one)/prob_one
                    
                    if random.uniform() < rat:
                        
                        A_current = A_new
                        A_inv_current = A_inv_new
                        A_invV_current = A_invV_new
                        
                        n_ones_current -= 1
                        
                        A_mask_current[rand_ind] = 0.
                        
                        A_zeros_ind_current.append(A_ones_ind_current.pop(rand_int))



    ### make pred cond on current phis, A
    
    
    
    rs = np.array([ np.exp(-Dists_obs_grid*phis_current[j]) for j in range(p) ])
    Rs_prime = np.array([ np.exp(-Dists_grid*phis_current[j]) for j in range(p) ])
    
    Rinvsrs = np.array([ Rs_inv_current[j]@rs[j] for j in range(p) ])
    
    Cs = np.array([ np.linalg.cholesky(Rs_prime[j] - np.transpose(rs[j])@Rinvsrs[j]) for j in range(p) ])
    
    V_grid_current = A_current @ np.array([ Cs[j]@random.normal(size=n_grid+1) + A_invV_current[j]@Rinvsrs[j] for j in range(p)])
    
    
    ###
    
    V_grid_run[i] = V_grid_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    
    if i % 10 == 0:
        print(i)

et = time.time()

print("TTIME:", (et-st)/60, "min")

print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))
# print('accept phi_3:',np.mean(acc_phis[2,tail:]))

plt.plot(phis_run[tail:,0])
plt.plot(phis_run[tail:,1])
# plt.plot(phis_run[tail:,2])
plt.show()




print("MSE", np.mean([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)]))
