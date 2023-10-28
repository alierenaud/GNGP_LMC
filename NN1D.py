#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:57:06 2023

@author: homeboy
"""

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from talk_exs import matern_kernel
from talk_exs import fct


####
# ARRAYS WITH VARIABLE DIMENSION SHOULD PROBABLY BE REMOVED (where possible) 
####

random.seed(0)

n=800
m=20

n_grid = 200

xlim=10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)
grid_locst = np.transpose([grid_locs])


locs = random.uniform(-xlim,xlim,n)
locs = np.sort(locs)
### distance function
locst = np.transpose([locs])

### parameters

mu = np.ones(n)
mu_grid = np.ones(n_grid+1)
a = 0.1
phi = 1000
tau = 10

### neighbors

Nei = np.zeros(n,dtype=object)

for i in range(m):
    Nei[i] = np.arange(i)
    
    
    
for j in range(m,n):
    Nei[j] = np.arange(j-m,j)
    


aNei = np.zeros(n,dtype=object)

for i in range(n):
    aNei[i] = np.array([],dtype = int)
    for j in range(n):
        if i in Nei[j]:
            aNei[i] = np.append(aNei[i],j)
            
            
# Nei
# aNei      


### compute B,r,dists

Dist = distance_matrix(locst, locst)

DistMats = np.zeros(n,dtype=object)
B = np.zeros((n,n))
r = np.zeros(n)

for i in range(n):

    
    DistMat_temp = Dist[np.append(Nei[i], i)][:,np.append(Nei[i], i)]
    
    DistMats[i] =  DistMat_temp
    
    cov_mat_temp = matern_kernel(DistMat_temp,phi)
    
    nNei_temp = Nei[i].shape[0]
    
    R_inv_temp = np.linalg.inv(cov_mat_temp[:nNei_temp,:nNei_temp])
    r_temp = cov_mat_temp[nNei_temp,:nNei_temp]

    
    b_temp = r_temp@R_inv_temp
    

    B[i][Nei[i]] = b_temp
    
    r[i] = 1-np.inner(b_temp,r_temp)

### compute grid neighbors

gridNei = np.zeros(n_grid+1,dtype=object)

Dist_grid = distance_matrix(grid_locst, locst)




for i in range(n_grid+1):
    
    gridNei[i] = np.sort(np.argpartition(Dist_grid[i],m)[:m])
    

    
DistMats_grid = np.zeros(n_grid+1,dtype=object)
B_grid = np.zeros((n_grid+1,n))
r_grid = np.zeros(n_grid+1)


for i in range(n_grid+1):

    
    DistMatObs_temp = Dist[gridNei[i]][:,gridNei[i]]
    DistMats_grid[i]  = DistMatObs_temp
    CovMatObs_temp = matern_kernel(DistMatObs_temp,phi)
    
    R_inv_temp = np.linalg.inv(CovMatObs_temp)
    
    DistMatGridObs_temp = Dist_grid[i][gridNei[i]]
    r_temp = matern_kernel(DistMatGridObs_temp,phi)
    

    
    b_temp = r_temp@R_inv_temp
    

    B_grid[i][gridNei[i]] = b_temp
    
    r_grid[i] = 1-np.inner(b_temp,r_temp)




### simulate an example y


w_true = fct(locs)
w_grid_true = fct(grid_locs)
y = w_true + 1/np.sqrt(tau)*random.normal(size = n)
### showcase data

plt.scatter(locs,y, c="black", s=10)
plt.show()  


# w_current = w_true
w_current = random.normal(size=n)

# w_grid = w_grid_true
w_grid = random.normal(size=n_grid+1)



N = 1000

w_grid_run = np.zeros((N,n_grid+1))



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
    
    ## w update
    
    for ii in random.permutation(range(n)):
    # for ii in range(n):
        
        
        A_temp = a/r[ii] + tau + np.sum([a/r[jj]*B[jj,ii]**2 for jj in aNei[ii]])
        
        B_temp = a/r[ii]*(mu[ii] + np.inner(B[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]) ) + tau*y[ii] + np.sum([a*B[jj,ii]/r[jj]*(w_current[jj] - mu[jj] - np.inner(B[jj,Nei[jj]],w_current[Nei[jj]] - mu[Nei[jj]]) + B[jj,ii]*w_current[ii] ) for jj in aNei[ii]])                    
        
        w_current[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp


    


    ## w grid update
    
    for ii in range(n_grid+1):
        
        a_temp = r_grid[ii] / a
        
        w_grid[ii] = np.sqrt(r_grid[ii] / a)*random.normal() + mu_grid[ii] + np.inner(B_grid[ii][gridNei[ii]],w_current[gridNei[ii]] - mu[gridNei[ii]])
        
    w_grid_run[i] = w_grid
    
et = time.time()

print("Total Time:", (et-st)/60, "minutes")

tail = 400

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
            
            
            
            