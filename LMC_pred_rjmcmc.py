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
              [-3,4]])
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



# ### proposals


phis_prop = np.ones(p)*1.5
sigma_slice = 10



### samples
N = 1000
tail = 400

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



for i in range(N):
    
    
    
    
                        
    
    A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_obs, sigma_A, mu_A, sigma_slice)
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_obs,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)

    ### make pred cond on current phis, A
    
    
    
    rs = np.array([ np.exp(-Dists_obs_grid*phis_current[j]) for j in range(p) ])
    Rs_prime = np.array([ np.exp(-Dists_grid*phis_current[j]) for j in range(p) ])
    
    Rinvsrs = np.array([ Rs_inv_current[j]@rs[j] for j in range(p) ])
    
    Cs = np.array([ np.linalg.cholesky(Rs_prime[j] - np.transpose(rs[j])@Rinvsrs[j]) for j in range(p) ])
    
    V_grid_current = A_current @ np.array([ Cs[j]@random.normal(size=n_grid+1) + A_invV_current[j]@Rinvsrs[j] for j in range(p)])
    
    V_grid_run[i] = V_grid_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    
    if i % 10 == 0:
        print(i)


print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))
# print('accept phi_3:',np.mean(acc_phis[2,tail:]))

plt.plot(phis_run[tail:,0])
plt.plot(phis_run[tail:,1])
# plt.plot(phis_run[tail:,2])
plt.show()




np.mean([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)])
