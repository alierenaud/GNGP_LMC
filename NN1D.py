#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:21:57 2023

@author: homeboy
"""

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from talk_exs import matern_kernel
from talk_exs import fct


random.seed(0)

n_grid = 200
xlim = 10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)
f_grid = fct(grid_locs)


n_obs = 200


### locations

locs = random.uniform(size=n_obs)*20 - 10

f_locs = fct(locs)
tau = 10.0
y = f_locs + 1/np.sqrt(tau)*random.normal(size = n_obs)

### showcase data

plt.plot(grid_locs,f_grid)
plt.scatter(locs,y, c="black", s=10)
plt.show()


### algorithm

mu_current = 1

phi_current = 100.0
a_current = 0.1
tau_current = 10.0

f_currrent = random.normal(n_obs)
f_grid_current = random.normal(n_grid+1)

### useful quantitites

D = distance_matrix(np.transpose([locs]),np.transpose([locs]))

R_current = matern_kernel(D,phi_current)
R_inv_current = np.linalg.inv(R_current)


D_grid_obs = distance_matrix(np.transpose([grid_locs]),np.transpose([locs]))

R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)


D_grid = distance_matrix(np.transpose([grid_locs]),np.transpose([grid_locs]))

R_grid_current = matern_kernel(D_grid,phi_current)


### containers

N = 1000

f_grid_run = np.zeros((N,n_grid+1))


for i in range(N):

    

    ### f update
    
    Sigma_f = np.linalg.inv(a_current*R_inv_current + tau_current*np.identity(n_obs))
    
    mu_f = Sigma_f@(a_current*R_inv_current@(mu_current*np.ones(n_obs)) + tau_current*y)
    
    f_current = np.linalg.cholesky(Sigma_f)@random.normal(size=n_obs) + mu_f
    
    
    # ### f grid update
    
    
    # R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)
    # R_grid_current = matern_kernel(D_grid,phi_current)
    
    
    # tempMat = R_grid_obs_current@R_inv_current
    
    # mu_grid = tempMat@(f_current-mu_current) + mu_current
    # Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
    # f_grid_current = np.linalg.cholesky(Sigma_grid)@random.normal(size=n_grid+1) + mu_grid
    
    # f_grid_run[i] = f_grid_current
    
    if i%100==0:

        # plt.plot(grid_locs,f_grid)
        # plt.plot(grid_locs,f_grid_current)
        # # plt.scatter(locs,y, c="black", s=10)
        # # plt.scatter(locs,f_current, c="tab:orange", s=10)
        # plt.show()

        print(i)



