# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:40:50 2023

@author: alier
"""




import numpy as np
from numpy import random


import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from LMC_pred_rjmcmc import A_move_slice_mask
from LMC_pred_rjmcmc import A_rjmcmc
from LMC_pred_rjmcmc import V_pred
from LMC_pred_rjmcmc import pairs


from noisyLMC_inference import V_move_conj, taus_move

from LMC_inference import phis_move
from LMC_mean import mu_move

import time

random.seed(0)

air_data = np.loadtxt("air_data.csv",delimiter=",")

loc_obs = air_data[:,5:7]


Y_obs = np.transpose(np.log(air_data[:,:3]))

n_obs = Y_obs.shape[1]
p = Y_obs.shape[0]

#### centered
# Y_obs = Y_obs - np.outer(np.mean(Y_obs,axis=1),np.ones(n_obs))




### markov chain + tail length
N = 20000
tail = 10000

### showcase locations

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black")
plt.show()


### priors
sigma_A = 1.
mu_A = np.zeros((p,p))
# mu_A = np.array([[0.,0.,0.],
#                  [0.,0.,0.],
#                  [0.,0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi

alphas = np.ones(p)
betas = np.ones(p)

### RJMCMC

prob_one = 1/2

## tau

a = 1
b = 1

### mu 

mu_mu = np.zeros(p)
sigma_mu = 10

### proposals


phis_prop = np.ones(p)*1
sigma_slice = 10


def ins_prob(n_ones,p):
    
    if n_ones == p**2:
        return(0)
    elif n_ones == p:
        return(1)
    else:
        return(0.5)

n_jumps = p

### global run containers
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))

n_comps_run = np.zeros((N))


### acc vector
acc_phis = np.zeros((p,N))


### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)


### current values

n_ones_current = p**2
A_mask_current = np.ones((p,p))
        
A_ones_ind_current = pairs(p)
A_zeros_ind_current = []

### init and current state
phis_current = np.repeat(10.,p)
Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


V_current = random.normal(size=(p,n_obs))*1
VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])

mu_current = np.zeros(p)
Vmmu1_current = V_current




A_current = random.normal(size=(p,p))
A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current



taus_current = np.ones(p)
Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs

st = time.time()

for i in range(N):
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current, Vmmu1_current, mu_current)
        
    
    
    
    mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    
    
    A_current, A_inv_current, A_invVmmu1_current = A_move_slice_mask(A_current, A_invVmmu1_current, A_mask_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

    
    
        
    A_current, A_inv_current, A_invVmmu1_current, n_ones_current, A_mask_current, A_ones_ind_current, A_zeros_ind_current = A_rjmcmc(Rs_inv_current, Vmmu1_current, A_current, A_inv_current, A_invVmmu1_current, A_zeros_ind_current, A_ones_ind_current, A_mask_current, n_ones_current, prob_one, mu_A, sigma_A, n_jumps)
    

    mu_run[i] = mu_current
    V_run[i] = V_current
    taus_run[i] = taus_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    n_comps_run[i] = n_ones_current
    
    if i % 100 == 0:
        print(i)

et = time.time()

print("Time Elapsed", (et-st)/60, "min")

print("Accept Rate for phis",np.mean(acc_phis,axis=1))


plt.plot(n_comps_run[tail:])
plt.show()

Sigmas0 = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)])
np.median(Sigmas0,axis=0)




np.mean(Sigmas0==0,axis=0)


reso = 200
dist_cov = np.linspace(0,1,reso)

i = 0
j = 0


C_ij = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

C_ij.shape

# plt.plot(Sigmas0[:,0,0])
# plt.show()
# plt.plot(Sigmas0[:,0,1])
# plt.show()
# plt.plot(Sigmas0[:,0,2])
# plt.show()
# plt.plot(Sigmas0[:,1,1])
# plt.show()
# plt.plot(Sigmas0[:,1,2])
# plt.show()
# plt.plot(Sigmas0[:,2,2])
# plt.show()

# plt.plot(mu_run[tail:,0])
# plt.show()
# plt.plot(mu_run[tail:,1])
# plt.show()
# plt.plot(mu_run[tail:,2])
# plt.show()

# plt.plot(phis_run[tail:,0])
# plt.show()
# plt.plot(phis_run[tail:,1])
# plt.show()
# plt.plot(phis_run[tail:,2])
# plt.show()


# plt.plot(taus_run[tail:,0])
# plt.show()
# plt.plot(taus_run[tail:,1])
# plt.show()
# plt.plot(taus_run[tail:,2])
# plt.show()


