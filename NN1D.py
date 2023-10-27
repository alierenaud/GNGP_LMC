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


n=800
m=20

n_grid = 200

xlim=10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)

locs = random.uniform(-xlim,xlim,n)
locs = np.sort(locs)
### distance function
locst = np.transpose([locs])

### parameters

mu = np.ones(n)
a = 0.1
phi = 100
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

for i in range(N):

    
    ## w update
    
    for ii in random.permutation(range(n)):
    # for ii in range(n):
        
        
        A_temp = a/r[ii] + tau + np.sum([a/r[jj]*B[jj,ii]**2 for jj in aNei[ii]])
        
        B_temp = a/r[ii]*(mu[ii] + np.inner(B[ii,Nei[ii]],w_current[Nei[ii]]-mu[Nei[ii]]) ) + tau*y[ii] + np.sum([a*B[jj,ii]/r[jj]*(w_current[jj] - mu[jj] - np.inner(B[jj,Nei[jj]],w_current[Nei[jj]] - mu[Nei[jj]]) + B[jj,ii]*w_current[ii] ) for jj in aNei[ii]])                    
        
        w_current[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp


    if i % 100 ==0:
        # plt.scatter(locs,y, c="black", s=10)
        plt.plot(grid_locs,w_grid_true)
        plt.scatter(locs,w_current, c="tab:orange", s=10)
        plt.show() 
        print(i)


    ## w grid update
    
    


   
            
            
            
            