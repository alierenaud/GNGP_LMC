#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:09:47 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from LMC_pred_rjmcmc import A_move_slice_mask
from LMC_pred_rjmcmc import A_rjmcmc
from LMC_pred_rjmcmc import V_pred
from LMC_pred_rjmcmc import pairs


from noisyLMC_interweaved import A_move_slice
from noisyLMC_interweaved import makeGrid
from noisyLMC_interweaved import vec_inv

from noisyLMC_inference import V_move_conj, taus_move

from LMC_inference import phis_move

import time

random.seed(0)

### number of points 
n_obs=100
n_grid=20  ### 2D Grid

### repetitions per category
reps = 20

### markov chain + tail length
N = 2000
tail = 1000

### generate uniform locations
loc_obs = random.uniform(0,1,(n_obs,2))
### grid locations
loc_grid = makeGrid(n_grid)
### all locations
locs = np.concatenate((loc_obs,loc_grid), axis=0)



### 5D examples
p = 5

### Triangular

A1 = np.array([[1,0,0,0,0],
              [-np.sqrt(1/2),-np.sqrt(1/2),0,0,0],
              [np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3),0,0],
              [-np.sqrt(1/4),-np.sqrt(1/4),-np.sqrt(1/4),-np.sqrt(1/4),0],
              [np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5)]])



### Full

A2 = np.ones((5,5))*np.sqrt(1/5)
A2 *= np.array([[1,-1,-1,-1,-1],
                [1,1,-1,-1,-1],
                [1,1,1,-1,-1],
                [1,1,1,1,-1],
                [1,1,1,1,1]])



### Block Diagonal

A3 = np.array([[np.sqrt(2/3),np.sqrt(1/3),0,0,0],
              [-np.sqrt(2/3),np.sqrt(1/3),0,0,0],
              [0,0,1.,0,0],
              [0,0,0,np.sqrt(2/3),np.sqrt(1/3)],
              [0,0,0,np.sqrt(2/3),-np.sqrt(1/3)]])




### Diagonal


A4 = np.identity(p)


phis = np.exp(np.linspace(np.log(5), np.log(25),5))
taus_sqrt_inv = np.array([1.,1.,1.,1.,1.]) 



As = np.array([A1,A2,A3,A4])
n_exes = As.shape[0]



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

prob_one = 0.5

## tau

a = 1
b = 1

### proposals


phis_prop = np.ones(p)*3.0
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
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,n_grid**2))
n_comps_run = np.zeros((N))


### acc vector
acc_phis = np.zeros((p,N))

### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)
Dists_obs_grid = distance_matrix(loc_obs,loc_grid)
Dists_grid = distance_matrix(loc_grid,loc_grid)


#### container of pred errors

MSES = np.zeros((n_exes,2,reps))
n_comps = np.zeros((n_exes,reps,N-tail))

STG = time.time()

for ex in range(n_exes):
    for rep in range(reps):
        
        

        ### generate rfs
        
        Y_true, V_true = rNLMC(As[ex],phis,taus_sqrt_inv,locs, retV=True)
        
        Y_obs = Y_true[:,:n_obs]
        V_grid = V_true[:,n_obs:]
        
        
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
        
        
        A_current = random.normal(size=(p,p))
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/taus_sqrt_inv**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y_obs
        
        st = time.time()
        
        for i in range(N):
            
            
            V_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current)
          
            
                                
            
                
            A_current, A_inv_current, A_invV_current = A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
            
                
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_current,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)
    
            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)
    
            
            
                
            A_current, A_inv_current, A_invV_current, n_ones_current, A_mask_current, A_ones_ind_current, A_zeros_ind_current = A_rjmcmc(Rs_inv_current, V_current, A_current, A_inv_current, A_invV_current, A_zeros_ind_current, A_ones_ind_current, A_mask_current, n_ones_current, prob_one, mu_A, sigma_A, n_jumps)
            
            ### make pred cond on current phis, A
    
            V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invV_current, n_grid)
            
            ###
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            V_grid_run[i] = V_grid_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            n_comps_run[i] = n_ones_current
            
            if i % 100 == 0:
                print(i)

        et = time.time()

        print("Time Elapsed", (et-st)/60, "min")
        print("RJMCMC", ex, rep)
        
        MSES[ex,0,rep] = np.mean([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)])
        n_comps[ex,rep] = n_comps_run[tail:N]
        
        ### init and current state
        phis_current = np.repeat(10.,p)
        Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        
        V_current = random.normal(size=(p,n_obs))*1
        VmY_current = V_current - Y_obs
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        
        A_current = random.normal(size=(p,p))
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/taus_sqrt_inv**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y_obs
        
        st = time.time()

        for i in range(N):
            
            
            V_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current)
          
            
            A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
            
                
                
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_current,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)

            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

            
            
            ### make pred cond on current phis, A

            V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invV_current, n_grid)
            
            ###
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            V_grid_run[i] = V_grid_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            
            if i % 100 == 0:
                print(i)

        et = time.time()
        
        print("Time Elapsed", (et-st)/60, "min")
        print("Standard", ex, rep)
        
        MSES[ex,1,rep] = np.mean([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)])
 
ETG = time.time()

print("GLOBAL TIME", (ETG-STG)/60, "min")

dMSE = MSES[:,0,:] - MSES[:,1,:]
        
np.savetxt("dMSE.csv", dMSE, delimiter=",")  
np.save("n_comps.npy", n_comps)
