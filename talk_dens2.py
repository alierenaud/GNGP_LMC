#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:33:50 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix
from scipy.stats import truncnorm


import matplotlib.pyplot as plt


from base import matern_kernel, fct2, makeGrid, vec_inv

from scipy.stats import norm

# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)


random.seed(0)

n_grid = 20


marg_grid = np.linspace(0,1,n_grid+1)



grid_locs = makeGrid(marg_grid,marg_grid)
f_grid = fct2(grid_locs)

### showcase data

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(f_grid,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.show()


n_1 = 500

mu = 0
a = 0.1
tau = 1.0

phi_current = 1

### priors

# sigma2_mu = 1

alpha_phi = 100
beta_phi = 100

# alpha_a = 0.01
# beta_a = 0.1

# alpha_tau = 1
# beta_tau = 0.1


### proposals

# alpha_prop = 100
sigma_prop = 0.1


### generate data

x_true = np.zeros((0,2))
g_true = np.array([])
y_true = np.array([])

r=0
while r < n_1:
    x = random.uniform(size=(1,2))
    g = fct2(x)
    y = fct2(x) + random.normal()
    
    x_true = np.append(x_true,x,axis=0)
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




fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, norm.cdf(vec_inv(f_grid,n_grid+1)), cmap = "Blues")
ax.scatter(x_1[:,0],x_1[:,1],c="black")
plt.colorbar(c)
plt.show()




### algorithm




### initiiate point process

n_0_current = 0
x_0_current = np.zeros(shape=(0,2))


g_0_current = np.zeros(shape=n_0_current) + mu
y_0_current = -np.ones(n_0_current)

# n_0_current = n_0_true
# x_0_current = x_0_true
# n_0_current = n_1
# x_0_current = random.uniform(size=(n_1,2))


# g_0_current = random.normal(size=n_0_current) + mu
# y_0_current = -np.ones(n_0_current)


# g_0_current = g_0_true
# y_0_current = y_0_true

# g_1_current = g_1_true
# y_1_current = y_1_true


g_1_current = np.zeros(shape=n_1) + mu
# g_1_current = random.normal(size=n_1) + mu
y_1_current = np.ones(n_1)


n_current = n_1 + n_0_current 
x_current = np.append(x_1,x_0_current,axis=0)
g_current = np.append(g_1_current,g_0_current)
y_current = np.append(y_1_current,y_0_current)


g_grid_current = np.zeros(shape=(n_grid+1)**2)
# g_grid_current = random.normal(size=(n_grid+1)**2)
# g_grid_current = f_grid


### useful quantitites

D_0_current = distance_matrix(x_0_current,x_0_current)
D_01_current = distance_matrix(x_0_current,x_1)
D_1 = distance_matrix(x_1,x_1)

D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

R_current = matern_kernel(D_current,phi_current)
R_inv_current = np.linalg.inv(R_current)


D_grid_obs_current = distance_matrix(grid_locs,x_current)

R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)


D_grid = distance_matrix(grid_locs,grid_locs)

R_grid_current = matern_kernel(D_grid,phi_current)


### containers

N = 2000

g_grid_run = np.zeros((N,(n_grid+1)**2))
phi_run = np.zeros(N)
n_0_run = np.zeros(N)

acc_phi = np.zeros(N)


from time import time

st = time()

for i in range(N):

    if i%100==0:

        fig, ax = plt.subplots()
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        ax.set_box_aspect(1)



        c = ax.pcolormesh(xv, yv, norm.cdf(vec_inv(g_grid_current,n_grid+1)), cmap = "Blues")
        ax.scatter(x_0_current[:,0],x_0_current[:,1],c="black")
        plt.colorbar(c)
        plt.show()

        print(i)

        

    ### point process (update x_0,g_0,y_0)
    
    count = 0
    
    while True:
        
        while True:
            ### simulate n_1 new variables 
            x_new = random.uniform(size=(3*n_1,2))
            
            D_new = distance_matrix(x_new, x_new)
            # print("new",np.sum(D_new<1e-4 - D_new.shape[0]))
            R_new = matern_kernel(D_new, phi_current)
            
            D_new_obs = distance_matrix(x_new, x_current)
            # print("new_obs",np.sum(D_new_obs<1e-4))
            R_new_obs = matern_kernel(D_new_obs, phi_current)
            
            
            
            B_temp = R_new_obs@R_inv_current
            V_temp = R_new-B_temp @ np.transpose(R_new_obs)
            
            
                
            try:
                C = np.linalg.cholesky(V_temp)
                g_new = C/np.sqrt(a)@random.normal(size=3*n_1)+B_temp@(g_current-mu) + mu
                y_new = g_new + random.normal(size=3*n_1)
                break
            except np.linalg.LinAlgError:
                print("oof")
            
    
        
                
                
        count += np.sum(y_new>0)
        
        if count >= n_1:
            
            
            x_current = np.append(x_current,x_new,axis=0)
            g_current = np.append(g_current,g_new)
            y_current = np.append(y_current,y_new)
            
            n_tail = np.where(y_current[n_current:] > 0)[0][n_1-1]
            
            x_tail = x_current[n_current:n_current+n_tail+1]
            g_tail = g_current[n_current:n_current+n_tail+1]
            y_tail = y_current[n_current:n_current+n_tail+1]
            
            
            n_0_current = np.sum(y_tail<0)
            
            n_0_run[i] = n_0_current
            # print(n_0_current)
            
            x_0_current = x_tail[y_tail<0]
            g_0_current = g_tail[y_tail<0]
            y_0_current = y_tail[y_tail<0]
            
            n_current = n_1 + n_0_current 
            x_current = np.append(x_1,x_0_current,axis=0)
            g_current = np.append(g_1_current,g_0_current)
            y_current = np.append(y_1_current,y_0_current)
            
            
            D_0_current = distance_matrix(x_0_current,x_0_current)
            D_01_current = distance_matrix(x_0_current,x_1)
            
            D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

            R_current = matern_kernel(D_current,phi_current)
            R_inv_current = np.linalg.inv(R_current)
            
            
            # print(np.prod(eigval>0))


            D_grid_obs_current = distance_matrix(grid_locs,x_current)

            # R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)


            
            # R_grid_current = matern_kernel(D_grid,phi_current)
            
            
            break
        else:
            x_current = np.append(x_current,x_new,axis=0)
            g_current = np.append(g_current,g_new)
            y_current = np.append(y_current,y_new)
            
            # DD = np.linalg.inv(V_temp)
            # BB = -DD@B_temp
            # AA = R_inv_current - np.transpose(B_temp)@BB
            # R_inv_current = np.block([[AA,np.transpose(BB)],[BB,DD]])
            
            # #
            
            D_current = distance_matrix(x_current, x_current)
            R_current = matern_kernel(D_current,phi_current)
            R_inv_current = np.linalg.inv(R_current)
            # print(R_inv_current @ RRRR_TEMP)
            
            print("not enough points")
            
            

    ### f update
    
    Sigma_f = np.linalg.inv(a*R_inv_current + tau*np.identity(n_current))
    
    mu_f = Sigma_f@(a*R_inv_current@(mu*np.ones(n_current)) + tau*y_current)
    
    
    
    try:
        C = np.linalg.cholesky(Sigma_f)
        g_current = C@random.normal(size=n_current) + mu_f
    except np.linalg.LinAlgError:
        print("oof g_current")
    
    g_0_current = g_current[n_1:]
    g_1_current = g_current[:n_1]
    
    
    ### y_update
    
    for ii in range(n_1):
        
        y_current[ii] = truncnorm.rvs(a=-g_current[ii],b=np.inf,loc=g_current[ii])
        
    for ii in range(n_1,n_current):
        
        y_current[ii] = truncnorm.rvs(a=-np.inf,b=-g_current[ii],loc=g_current[ii])
    
    y_0_current = y_current[n_1:]
    y_1_current = y_current[:n_1]
    
    
    ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    
    while True:
        phi_new = sigma_prop*random.normal() + phi_current
        if phi_new > 0:
            break
        
    R_new = matern_kernel(D_current,phi_new)
    R_inv_new = np.linalg.inv(R_new)
    
    sus = np.exp( a/2 * np.transpose(g_current-mu)@(R_inv_current-R_inv_new)@(g_current-mu) )
    
    # print("sus",sus)
    
    pect = np.linalg.det(R_current@R_inv_new)**(1/2)
    
    # print("pect",pect)
    
    prior = (phi_new/phi_current)**(alpha_phi-1) * np.exp(-beta_phi*(phi_new-phi_current))
    
    # print("prior",prior)
    
    # trans = (phi_current/phi_new)**(alpha_prop-1) * np.exp(-alpha_prop*(phi_current/phi_new - phi_new/phi_current))
    
    # print("trans",trans)
    
    ratio = sus * pect * prior 
    
    
    if random.uniform() < ratio:
        phi_current = phi_new
        R_current = np.copy(R_new)
        R_inv_current = np.copy(R_inv_new)
        
        acc_phi[i] = 1
    
    
    phi_run[i] = phi_current
    
    
    
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
    
    ### f grid update
    # print("grid_obs",np.sum(D_grid_obs_current<1e-4))
    
    R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)
    R_grid_current = matern_kernel(D_grid,phi_current)
    
    
    tempMat = R_grid_obs_current@R_inv_current
    
    mu_grid = tempMat@(g_current-mu) + mu
    Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
    try:
        C = np.linalg.cholesky(Sigma_grid)
        g_grid_current = C/np.sqrt(a)@random.normal(size=(n_grid+1)**2) + mu_grid
         
    except np.linalg.LinAlgError:
        print("oof g_grid")
    

    g_grid_run[i] =g_grid_current
    
    

et = time()
print("Time:",(et-st)/60,"minutes")

tail = 1000

Phi_g_grid_run = norm.cdf(g_grid_run)

maxes = np.max(Phi_g_grid_run,axis=1)


for i in range(N):
    Phi_g_grid_run[i]=Phi_g_grid_run[i]/maxes[i]

Phi_g_grid_mean = np.mean(Phi_g_grid_run[tail:], axis=0)
Phi_g_grid_025 = np.quantile(Phi_g_grid_run[tail:], 0.05, axis=0)
Phi_g_grid_975 = np.quantile(Phi_g_grid_run[tail:], 0.95, axis=0)


fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(Phi_g_grid_mean,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.show()



print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

plt.boxplot(phi_run[tail:])
plt.show()

plt.plot(n_0_run[tail:])
plt.show()

plt.boxplot(n_0_run[tail:])
plt.show()


   
print("MSE:", np.mean((norm.cdf(f_grid) - Phi_g_grid_mean)**2))
print("TMSE:", np.mean((Phi_g_grid_run[tail:] - norm.cdf(f_grid))**2)) 