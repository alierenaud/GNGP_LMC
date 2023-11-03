#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:55:05 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix


import matplotlib.pyplot as plt


from base import matern_kernel, fct

from scipy.stats import norm


random.seed(0)

n_grid = 200
xlim = 10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)
f_grid = fct(grid_locs)

### illustrate function
plt.plot(grid_locs,f_grid)
plt.show()

### illustrate prop dens
plt.plot(grid_locs,norm.cdf(f_grid))
plt.show()


n_1 = 100


### generate data

x_true = np.array([])
g_true = np.array([])
y_true = np.array([])

r=0
while r < n_1:
    x = random.uniform() * 20 - 10
    g = fct(x)
    y = fct(x) + random.normal()
    
    x_true = np.append(x_true,x)
    g_true = np.append(g_true,g)
    y_true = np.append(y_true,y)
    
    if y > 0:
        r+=1

n_0_true = np.sum(y_true<0)
print("n_1 =",np.sum(y_true>0))

x_0_true = x_true[y_true<0]
x_1 = x_true[y_true>0]

g_0_true = g_true[y_true<0]
g_1_true = g_true[y_true>0]

y_0_true = y_true[y_true<0]
y_1_true = y_true[y_true>0]

### showcase data        
plt.hist(x_1,density=True,alpha=0.5)
# plt.plot(grid_locs,f_grid)
plt.show()
plt.violinplot(x_1)
plt.show()   





### priors

# sigma2_mu = 1

alpha_phi = 100
beta_phi = 1

# alpha_a = 0.01
# beta_a = 0.1

# alpha_tau = 1
# beta_tau = 0.1


### proposals

alpha_prop = 100


### algorithm

mu = 0
a = 1.0
tau = 1.0

phi_current = 100.0


### initiiate point process
# n_0_current = n_1
# x_0_current = random.uniform(size=(n_1,1))
# g_0_current = random.normal(size=n_0_current) + mu
# y_0_current = -1/2*np.ones(n_0_current)

n_0_current = n_0_true
x_0_current = x_0_true
g_0_current = g_0_true
y_0_current = y_0_true

g_1_current = g_1_true
y_1_current = y_1_true


n_current = n_1 + n_0_current 
x_current = np.append(x_1,x_0_current)
g_current = np.append(g_1_current,g_0_current)
y_current = np.append(y_1_current,y_0_current)

g_grid_current = random.normal(size=n_grid+1)


### useful quantitites

D_0_current = distance_matrix(np.transpose([x_0_current]),np.transpose([x_0_current]))
D_01_current = distance_matrix(np.transpose([x_0_current]),np.transpose([x_1]))
D_1 = distance_matrix(np.transpose([x_1]),np.transpose([x_1]))

D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

R_current = matern_kernel(D_current,phi_current)
R_inv_current = np.linalg.inv(R_current)


D_grid_obs_current = distance_matrix(np.transpose([grid_locs]),np.transpose([x_current]))

R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)


D_grid = distance_matrix(np.transpose([grid_locs]),np.transpose([grid_locs]))

R_grid_current = matern_kernel(D_grid,phi_current)


### containers

N = 1000

g_grid_run = np.zeros((N,n_grid+1))
phi_run = np.zeros(N)

acc_phi = np.zeros(N)


from time import time

st = time()

for i in range(N):

    if i%1==0:

        plt.plot(grid_locs, norm.cdf(f_grid), c="black")
        # plt.scatter(x_1, np.zeros(n_1), s=10)
        plt.scatter(x_0_current, np.zeros(n_0_current), c="tab:orange", marker="|")
        
        plt.scatter(x_1, norm.cdf(g_1_current), s=10, c="tab:blue")
        plt.scatter(x_0_current, norm.cdf(g_0_current), s=10, c="tab:orange")
        plt.show()

        print(i)

        

    ### point process (update x_0,g_0,y_0)
    
    count = 0
    
    while True:
        
        ### simulate n_1 new variables 
        x_new = random.uniform(size=n_1) * 20 - 10
        
        D_new = distance_matrix(np.transpose([x_new]), np.transpose([x_new]))
        R_new = matern_kernel(D_new, phi_current)
        
        D_new_obs = distance_matrix(np.transpose([x_new]), np.transpose([x_current]))
        R_new_obs = matern_kernel(D_new_obs, phi_current)
        
        B_temp = R_new_obs@R_inv_current
        V_temp = R_new-B_temp @ np.transpose(R_new_obs)
        
        g_new = np.linalg.cholesky(V_temp)@random.normal(size=n_1)+B_temp@(g_current-mu) + mu
        y_new = g_new + random.normal(size=n_1)
        
        count += np.sum(y_new>0)
        
        if count >= n_1:
            
            
            x_current = np.append(x_current,x_new)
            g_current = np.append(g_current,g_new)
            y_current = np.append(y_current,y_new)
            
            n_tail = np.where(y_current[n_current:] > 0)[0][n_1-1]
            
            x_tail = x_current[n_current:n_current+n_tail+1]
            g_tail = g_current[n_current:n_current+n_tail+1]
            y_tail = y_current[n_current:n_current+n_tail+1]
            
            
            n_0_current = np.sum(y_tail<0)
            x_0_current = x_tail[y_tail<0]
            g_0_current = g_tail[y_tail<0]
            y_0_current = y_tail[y_tail<0]
            
            n_current = n_1 + n_0_current 
            x_current = np.append(x_1,x_0_current)
            g_current = np.append(g_1_current,g_0_current)
            y_current = np.append(y_1_current,y_0_current)
            
            
            D_0_current = distance_matrix(np.transpose([x_0_current]),np.transpose([x_0_current]))
            D_01_current = distance_matrix(np.transpose([x_0_current]),np.transpose([x_1]))
            
            D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

            R_current = matern_kernel(D_current,phi_current)
            R_inv_current = np.linalg.inv(R_current)


            D_grid_obs_current = distance_matrix(np.transpose([grid_locs]),np.transpose([x_current]))

            R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)


            
            R_grid_current = matern_kernel(D_grid,phi_current)
            
            
            break
        else:
            x_current = np.append(x_current,x_new)
            g_current = np.append(g_current,g_new)
            y_current = np.append(y_current,y_new)
            
            # DD = np.linalg.inv(V_temp)
            # BB = -DD@B_temp
            # AA = R_inv_current - np.transpose(B_temp)@BB
            # R_inv_current = np.block([[AA,np.transpose(BB)],[BB,DD]])
            
            ##
            
            DDDD_TEMP = distance_matrix(np.transpose([x_current]), np.transpose([x_current]))
            RRRR_TEMP = matern_kernel(DDDD_TEMP,phi_current)
            R_inv_current = np.linalg.inv(RRRR_TEMP)
            # print(R_inv_current @ RRRR_TEMP)
            
            print("hey")
            


    # ### f update
    
    # Sigma_f = np.linalg.inv(a_current*R_inv_current + tau_current*np.identity(n_obs))
    
    # mu_f = Sigma_f@(a_current*R_inv_current@(mu_current*np.ones(n_obs)) + tau_current*y)
    
    # f_current = np.linalg.cholesky(Sigma_f)@random.normal(size=n_obs) + mu_f
    
    
    # ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    # R_new = matern_kernel(D,phi_new)
    # R_inv_new = np.linalg.inv(R_new)
    
    # sus = np.exp( a_current/2 * np.transpose(f_current-mu_current)@(R_inv_current-R_inv_new)@(f_current-mu_current) )
    
    # # print("sus",sus)
    
    # pect = np.linalg.det(R_current@R_inv_new)**(1/2)
    
    # # print("pect",pect)
    
    # prior = (phi_new/phi_current)**(alpha_phi-1) * np.exp(-beta_phi*(phi_new-phi_current))
    
    # # print("prior",prior)
    
    # trans = (phi_current/phi_new)**(alpha_prop-1) * np.exp(-alpha_prop*(phi_current/phi_new - phi_new/phi_current))
    
    # # print("trans",trans)
    
    # ratio = sus * pect * prior * trans
    
    
    # if random.uniform() < ratio:
    #     phi_current = phi_new
    #     R_current = R_new
    #     R_inv_current = R_inv_new
        
    #     acc_phi[i] = 1
    
    
    # phi_run[i] = phi_current
    
    # # ### mu update
    
    # # sigma2_cond = 1/(a_current*np.sum(R_inv_current) + 1/sigma2_mu)
    # # mu_cond = sigma2_cond*np.inner(f_current@R_inv_current, np.ones(n_obs))
    
    # # mu_current = random.normal(mu_cond,sigma2_cond)
    
    # # mu_run[i] = mu_current
    
    # # ### a update
    
    # # alpha_cond = n_obs/2+alpha_a
    # # beta_cond = np.transpose(f_current - mu_current)@R_inv_current@(f_current - mu_current)/2 + beta_a
    
    # # a_current = random.gamma(alpha_cond,1/beta_cond)
    
    # # a_run[i] = a_current
    
    # # ### tau update
    
    # # alpha_cond = n_obs/2+alpha_tau
    # # beta_cond = np.inner(y-f_current,y-f_current)/2 + beta_tau
    
    # # tau_current = random.gamma(alpha_cond,1/beta_cond)
    
    # # tau_run[i] = tau_current
    
    # ### f grid update
    
    
    # R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)
    # R_grid_current = matern_kernel(D_grid,phi_current)
    
    
    # tempMat = R_grid_obs_current@R_inv_current
    
    # mu_grid = tempMat@(f_current-mu_current) + mu_current
    # Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
    # f_grid_current = np.linalg.cholesky(Sigma_grid)@random.normal(size=n_grid+1) + mu_grid
    
    # f_grid_run[i] = f_grid_current
    
    

et = time()






     
        
