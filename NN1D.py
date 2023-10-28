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

Nei = np.full((n,n),False)

for i in range(m):
    Nei[i,:i] = True
    
    
    
for j in range(m,n):
    Nei[j,j-m:j] = True
    


aNei = np.full((n,n),False)

for i in range(n):
    for j in range(n):
        if Nei[j,i]:
            aNei[i,j] = True
            
            
phi_current = phi     


### compute B,r,dists

Dist = distance_matrix(locst, locst)


B = np.zeros((n,n))
r = np.zeros(n)

for i in range(n):

    
    ### NEED TO STOCK DISTANCE MATRICES
    
    DistMatNei = Dist[Nei[i]][:,Nei[i]]
    DistMatLocNei = Dist[i,Nei[i]]
    
    
    
    CovMatNei = matern_kernel(DistMatNei,phi_current)
    invCovMatNei = np.linalg.inv(CovMatNei)
    
    CovMatLocNei = matern_kernel(DistMatLocNei,phi_current)
    
    
    
   
    


    
    bNei = CovMatLocNei@invCovMatNei
    

    B[i,Nei[i]] = bNei
    
    r[i] = 1-np.inner(bNei,CovMatLocNei)

### compute grid neighbors

gridNei = np.full((n_grid+1,n),False)

Dist_grid = distance_matrix(grid_locst, locst)




for i in range(n_grid+1):
    
    gridNei[i,np.argpartition(Dist_grid[i],m)[:m]] = True
    

    

B_grid = np.zeros((n_grid+1,n))
r_grid = np.zeros(n_grid+1)


for i in range(n_grid+1):
    
    ### NEED TO STOCK DISTANCE MATRICES
    
    DistMatNei = Dist[gridNei[i]][:,gridNei[i]]
    DistMatGridNei = Dist_grid[i,gridNei[i]]
    
    CovMatNei = matern_kernel(DistMatNei,phi_current)
    invCovMatNei = np.linalg.inv(CovMatNei)
    
    CovMatGridNei = matern_kernel(DistMatGridNei,phi_current)
    
    
    bNei = CovMatGridNei@invCovMatNei
    

    B_grid[i,gridNei[i]] = bNei
    
    r_grid[i] = 1-np.inner(bNei,CovMatGridNei)




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
        
        
        A_temp = a/r[ii] + tau + np.sum([a/r[jj]*B[jj,ii]**2 for jj in np.where(aNei[ii])[0]])
        
        B_temp = a/r[ii]*(mu[ii] + np.inner(B[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]) ) + tau*y[ii] + np.sum([a*B[jj,ii]/r[jj]*(w_current[jj] - mu[jj] - np.inner(B[jj,Nei[jj]],w_current[Nei[jj]] - mu[Nei[jj]]) + B[jj,ii]*w_current[ii] ) for jj in np.where(aNei[ii])[0]])                    
        
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
            
            
            
            