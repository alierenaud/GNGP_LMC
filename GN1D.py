#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:41:14 2023

@author: homeboy
"""


import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from talk_exs import matern_kernel
from talk_exs import fct


random.seed(0)

n_obs=200
m=20

n_grid = 200

xlim=10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)
grid_locst = np.transpose([grid_locs])


locs = random.uniform(-xlim,xlim,n_obs)
locs = np.sort(locs)
### distance function
locst = np.transpose([locs])

### parameters

mu = np.ones(n_obs)
mu_grid = np.ones(n_grid+1)
a = 0.1
phi_current = 1000.0
tau = 10

### compute grid neighbors

gNei = np.zeros(n_grid+1,dtype=object)

for i in range(m):
    gNei[i] = np.arange(i)
    
    
    
for j in range(m,n_grid+1):
    gNei[j] = np.arange(j-m,j)



agNei = np.zeros(n_grid+1,dtype=object)

for i in range(n_grid+1):
    agNei[i] = np.array([],dtype = int)
    for j in range(n_grid+1):
        if i in gNei[j]:
            agNei[i] = np.append(agNei[i],j)




### compute B,r,dists

Distg = distance_matrix(grid_locst, grid_locst)

DistgMats = np.zeros(n_grid+1,dtype=object)
Bg_current = np.zeros((n_grid+1,n_grid+1))
rg_current= np.zeros(n_grid+1)

for i in range(n_grid+1):

    
    DistgMat_temp = Distg[np.append(gNei[i], i)][:,np.append(gNei[i], i)]
    
    DistgMats[i] =  DistgMat_temp
    
    cov_mat_temp = matern_kernel(DistgMat_temp,phi_current)
    
    ngNei_temp = gNei[i].shape[0]
    
    R_inv_temp = np.linalg.inv(cov_mat_temp[:ngNei_temp,:ngNei_temp])
    r_temp = cov_mat_temp[ngNei_temp,:ngNei_temp]

    
    b_temp = r_temp@R_inv_temp
    

    Bg_current[i][gNei[i]] = b_temp
    
    rg_current[i] = 1-np.inner(b_temp,r_temp)


### compute obs neighbors on grid

ogNei = np.zeros(n_obs,dtype=object)
aogNei = np.zeros(n_grid+1,dtype=object)

for i in range(n_grid+1):
    aogNei[i] = np.empty(0,dtype=int)

Distog = distance_matrix(locst, grid_locst)




for i in range(n_obs):
    
    
    
    
    
    leftNei = int(np.floor( (locs[i] + xlim)/(2*xlim) * n_grid))
    
    off_left = -min(leftNei - m//2 + 1,0)
    off_right = max(leftNei + m//2,n_grid) - n_grid
    
    leftMostNei = leftNei-m//2+1+off_left-off_right
    rightMostNei = leftNei + m//2 + 1 + off_left - off_right
    
    ogNei[i] = np.arange(leftMostNei,rightMostNei)
    
    for j in np.arange(leftMostNei,rightMostNei):
        aogNei[j] = np.append(aogNei[j],i)
    

    
DistggMats = np.zeros(n_obs,dtype=object)
Bog= np.zeros((n_obs,n_grid+1))
rog = np.zeros(n_obs)


for i in range(n_obs):

    
    DistMatog_temp = Distg[ogNei[i]][:,ogNei[i]]
    DistggMats[i]  = DistMatog_temp
    CovMatgg_temp = matern_kernel(DistMatog_temp,phi_current)
    
    R_inv_temp = np.linalg.inv(CovMatgg_temp)
    
    DistMatog_temp = Distog[i][ogNei[i]]
    r_temp = matern_kernel(DistMatog_temp,phi_current)
    

    
    b_temp = r_temp@R_inv_temp
    

    Bog[i][ogNei[i]] = b_temp
    
    rog[i] = 1-np.inner(b_temp,r_temp)


### simulate an example y


w_true = fct(locs)
w_grid_true = fct(grid_locs)
y = w_true + 1/np.sqrt(tau)*random.normal(size = n_obs)
### showcase data

plt.scatter(locs,y, c="black", s=10)
plt.show()  


# w_current = w_true
w_current = random.normal(size=n_obs)

# w_grid = w_grid_true
w_grid = random.normal(size=n_grid+1)

### priors

# alpha_phi = 10
# beta_phi = 0.001

alpha_phi = (10000/2000)**2
beta_phi = np.sqrt(alpha_phi)/2000

### proposals

alpha_prop = 10


### algorithm


N = 1000

w_grid_run = np.zeros((N,n_grid+1))
phi_run = np.zeros(N)

acc_phi = np.zeros(N)


### containers

Bg_new = np.zeros((n_grid+1,n_grid+1))
rg_new = np.zeros(n_grid+1)

import time

st = time.time()

for i in range(N):
    
    
    
    if i % 100 ==0:
        # plt.scatter(locs,y, c="black", s=10)
        plt.plot(grid_locs,w_grid_true)
        plt.plot(grid_locs,w_grid)
        # plt.scatter(locs,w_current, c="tab:orange", s=10)
        plt.show() 
        print(i)
    
    # # w_grid update
    
    # for ii in random.permutation(range(n)):
    # # for ii in range(n):
        
        
    #     A_temp = a/r_current[ii] + tau + np.sum([a/r_current[jj]*B_current[jj,ii]**2 for jj in aNei[ii]])
        
    #     B_temp = a/r_current[ii]*(mu[ii] + np.inner(B_current[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]) ) + tau*y[ii] + np.sum([a*B_current[jj,ii]/r_current[jj]*(w_current[jj] - mu[jj] - np.inner(B_current[jj,Nei[jj]],w_current[Nei[jj]] - mu[Nei[jj]]) + B_current[jj,ii]*w_current[ii] ) for jj in aNei[ii]])                    
        
    #     w_current[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp


    
    
et = time.time()
