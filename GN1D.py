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

n_obs=800
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
a = 1
phi_current = 100.0
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
Bog_current = np.zeros((n_obs,n_grid+1))
rog_current = np.zeros(n_obs)


for i in range(n_obs):

    
    DistMatog_temp = Distg[ogNei[i]][:,ogNei[i]]
    DistggMats[i]  = DistMatog_temp
    CovMatgg_temp = matern_kernel(DistMatog_temp,phi_current)
    
    R_inv_temp = np.linalg.inv(CovMatgg_temp)
    
    DistMatog_temp = Distog[i][ogNei[i]]
    r_temp = matern_kernel(DistMatog_temp,phi_current)
    

    
    b_temp = r_temp@R_inv_temp
    

    Bog_current[i][ogNei[i]] = b_temp
    
    rog_current[i] = 1-np.inner(b_temp,r_temp)


### simulate an example y


w_true = fct(locs)
w_grid_true = fct(grid_locs)
y = w_true + 1/np.sqrt(tau)*random.normal(size = n_obs)
### showcase data

plt.scatter(locs,y, c="black", s=10)
plt.show()  


# w_current = w_true
w_current = random.normal(size=n_obs) + 1

# w_grid = w_grid_true
w_grid = random.normal(size=n_grid+1) + 1

### priors

# alpha_phi = 10
# beta_phi = 0.001

alpha_phi = (10000/2000)**2
beta_phi = np.sqrt(alpha_phi)/2000

### proposals

alpha_prop = 10


### algorithm


N = 4000

w_grid_run = np.zeros((N,n_grid+1))
phi_run = np.zeros(N)

acc_phi = np.zeros(N)


### containers

Bg_new = np.zeros((n_grid+1,n_grid+1))
rg_new = np.zeros(n_grid+1)

Bog_new = np.zeros((n_grid+1,n_grid+1))
rog_new = np.zeros(n_grid+1)

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
    
    # w_grid update
    
    for ii in random.permutation(range(n_grid+1)):
    # for ii in range(n):
        
        
        A_temp = a/rg_current[ii] + np.sum([a/rg_current[jj]*Bg_current[jj,ii]**2 for jj in agNei[ii]]) + np.sum([a/rog_current[jj]*Bog_current[jj,ii]**2 for jj in aogNei[ii]])
        
        B_temp = a/rg_current[ii]*(mu_grid[ii] + np.inner(Bg_current[ii,gNei[ii]],w_grid[gNei[ii]]-mu_grid[gNei[ii]]) ) + np.sum([a*Bg_current[jj,ii]/rg_current[jj]*(w_grid[jj] - mu_grid[jj] - np.inner(Bg_current[jj,gNei[jj]],w_grid[gNei[jj]] - mu_grid[gNei[jj]]) + Bg_current[jj,ii]*w_grid[ii] ) for jj in agNei[ii]])  +  np.sum([a*Bog_current[jj,ii]/rog_current[jj]*(w_current[jj] - mu[jj] - np.inner(Bog_current[jj,ogNei[jj]],w_grid[ogNei[jj]] - mu_grid[ogNei[jj]]) + Bog_current[jj,ii]*w_grid[ii] ) for jj in aogNei[ii]])             
        
        w_grid[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp

    
    # w_obs update


    for ii in range(n_obs):
        
        a_temp = a/rog_current[ii] + tau 
        b_temp = a/rog_current[ii]*(mu[ii] + np.inner(Bog_current[ii,ogNei[ii]],w_grid[ogNei[ii]] - mu_grid[ogNei[ii]])) + tau*y[ii]
        
        w_current[ii] = 1/np.sqrt(a_temp)*random.normal() + b_temp/a_temp
    
    w_grid_run[i] = w_grid
    
    ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    # for ii in range(n):

        
        
    #     DistMat_temp = DistMats[ii]
        
    #     cov_mat_temp = matern_kernel(DistMat_temp,phi_new)
        
    #     nNei_temp = Nei[ii].shape[0]
        
    #     R_inv_temp = np.linalg.inv(cov_mat_temp[:nNei_temp,:nNei_temp])
    #     r_temp = cov_mat_temp[nNei_temp,:nNei_temp]

        
    #     b_temp = r_temp@R_inv_temp
        

    #     B_new[ii][Nei[ii]] = b_temp
        
    #     r_new[ii] = 1-np.inner(b_temp,r_temp)
    
    
    # ratio = np.exp(- a/2* np.sum([ (w_current[ii] - mu[ii] - np.inner(B_new[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]))/r_new[ii] - (w_current[ii] - mu[ii] - np.inner(B_current[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]))/r_current[ii] for ii in range(n)])) * np.prod([(r_current[ii]/r_new[ii])**(1/2) for ii in range(n)]) * np.exp(alpha_prop*(phi_new/phi_current - phi_current/phi_new)) * np.exp(beta_phi*(phi_current-phi_new)) * (phi_new/phi_current)**(alpha_phi-2*alpha_prop)
    
    # if random.uniform() < ratio:
    #     phi_current = phi_new
    #     B_current = B_new
    #     r_current = r_new
    
    
et = time.time()

print("Total Time:", (et-st)/60, "minutes")

tail = 2000

print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

w_grid_mean = np.mean(w_grid_run[tail:], axis=0)
w_grid_025 = np.quantile(w_grid_run[tail:], 0.025, axis=0)
w_grid_975 = np.quantile(w_grid_run[tail:], 0.975, axis=0)


plt.plot(grid_locs,w_grid_true)
plt.plot(grid_locs,w_grid_mean)
plt.show()

plt.plot(grid_locs,w_grid_true)
plt.plot(grid_locs,w_grid_mean)
plt.fill_between(grid_locs, w_grid_025, w_grid_975, alpha=0.5,color="tab:orange")
plt.show()
