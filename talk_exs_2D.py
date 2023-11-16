#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:09:58 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix


import matplotlib.pyplot as plt


from base import matern_kernel, fct2, makeGrid, vec_inv




random.seed(0)

n_grid = 20
xlim = 10


marg_grid = np.linspace(0,1,n_grid+1)



grid_locs = makeGrid(marg_grid,marg_grid)
f_grid = fct2(grid_locs)


n_obs = 2000

### showcase data

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(f_grid,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.show()

### unif locs

locs = random.uniform(size=(n_obs,2))

f_locs = fct2(locs)
tau = 10.0
y = f_locs + 1/np.sqrt(tau)*random.normal(size = n_obs)

# np.prod(np.abs(locs)<10)
# np.sum(np.abs(locs)>10)





### priors

sigma2_mu = 1

alpha_phi = 100
beta_phi = 100

alpha_a = 0.01
beta_a = 0.1

alpha_tau = 1
beta_tau = 0.1


### proposals

# alpha_prop = 100
phi_prop = 0.02

### algorithm

mu_current = 0

phi_current = 1
a_current = 1
tau_current = 10.0

# f_currrent = random.normal(size=n_obs)
# f_grid_current = random.normal(size=(n_grid+1)**2)

f_currrent = np.zeros(shape=n_obs)
f_grid_current = np.zeros(shape=(n_grid+1)**2)


### useful quantitites

D = distance_matrix(locs,locs)

R_current = matern_kernel(D,phi_current)
R_inv_current = np.linalg.inv(R_current)


D_grid_obs = distance_matrix(grid_locs,locs)

R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)


D_grid = distance_matrix(grid_locs,grid_locs)

R_grid_current = matern_kernel(D_grid,phi_current)


### containers

N = 2000

f_grid_run = np.zeros((N,(n_grid+1)**2))
phi_run = np.zeros(N)
mu_run = np.zeros(N)
a_run = np.zeros(N)
tau_run = np.zeros(N)

acc_phi = np.zeros(N)

from time import time

st = time()

for i in range(N):

    if i%100==0:

        
        



        fig, ax = plt.subplots()
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        ax.set_box_aspect(1)



        c = ax.pcolormesh(xv, yv, vec_inv(f_grid_current,n_grid+1), cmap = "Blues")
        plt.colorbar(c)
        plt.show()

        print(i)
        print(phi_current)

    ### f update
    
    Sigma_f = np.linalg.inv(a_current*R_inv_current + tau_current*np.identity(n_obs))
    
    mu_f = Sigma_f@(a_current*R_inv_current@(mu_current*np.ones(n_obs)) + tau_current*y)
    
    f_current = np.linalg.cholesky(Sigma_f)@random.normal(size=n_obs) + mu_f
    
    
    ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    while True:
        phi_new = phi_prop*random.normal() + phi_current
        if phi_new > 0:
            break
    
    R_new = matern_kernel(D,phi_new)
    R_inv_new = np.linalg.inv(R_new)
    
    sus = a_current/2 * np.transpose(f_current-mu_current)@(R_inv_current-R_inv_new)@(f_current-mu_current) 
    
    # print("sus",sus)
    
    pect = (1/2)*np.log(np.linalg.det(R_current@R_inv_new))
    
    # print("pect",pect)
    
    prior = (alpha_phi-1)*np.log(phi_new/phi_current) + -beta_phi*(phi_new-phi_current)
    
    # print("prior",prior)
    
    # trans = (phi_current/phi_new)**(alpha_prop-1) * np.exp(-alpha_prop*(phi_current/phi_new - phi_new/phi_current))
    
    # print("trans",trans)
    
    ratio = np.exp(sus + pect + prior) 
    
    
    if random.uniform() < ratio:
        phi_current = phi_new
        R_current = np.copy(R_new)
        R_inv_current = np.copy(R_inv_new)
        
        acc_phi[i] = 1
    
    
    phi_run[i] = phi_current
    
    # ### mu update
    
    # sigma2_cond = 1/(a_current*np.sum(R_inv_current) + 1/sigma2_mu)
    # mu_cond = sigma2_cond*np.inner(f_current@R_inv_current, np.ones(n_obs))
    
    # mu_current = random.normal(mu_cond,sigma2_cond)
    
    # mu_run[i] = mu_current
    
    # ### a update
    
    # alpha_cond = n_obs/2+alpha_a
    # beta_cond = np.transpose(f_current - mu_current)@R_inv_current@(f_current - mu_current)/2 + beta_a
    
    # a_current = random.gamma(alpha_cond,1/beta_cond)
    
    # a_run[i] = a_current
    
    # ### tau update
    
    # alpha_cond = n_obs/2+alpha_tau
    # beta_cond = np.inner(y-f_current,y-f_current)/2 + beta_tau
    
    # tau_current = random.gamma(alpha_cond,1/beta_cond)
    
    # tau_run[i] = tau_current
    
    ### f grid update
    
    
    R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)
    R_grid_current = matern_kernel(D_grid,phi_current)
    
    
    tempMat = R_grid_obs_current@R_inv_current
    
    mu_grid = tempMat@(f_current-mu_current) + mu_current
    Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
    f_grid_current = np.sqrt(1/a_current)*np.linalg.cholesky(Sigma_grid)@random.normal(size=(n_grid+1)**2) + mu_grid
    
    f_grid_run[i] = f_grid_current
    
    

et = time()
print("Time:",(et-st)/60,"minutes")

tail = 1000

f_grid_mean = np.mean(f_grid_run[tail:], axis=0)
f_grid_025 = np.quantile(f_grid_run[tail:], 0.025, axis=0)
f_grid_975 = np.quantile(f_grid_run[tail:], 0.975, axis=0)


plt.figure(figsize=(4,4))
fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(f_grid_mean,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.title("GP")
# plt.savefig("mean_GP_2000.pdf", bbox_inches='tight')
plt.show()




print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

plt.boxplot(phi_run[tail:])
plt.show()

# plt.plot(mu_run[tail:])
# plt.show()

# plt.plot(a_run[tail:])
# plt.show()

# plt.plot(tau_run[tail:])
# plt.show()


# #### histograms

# plt.hist(phi_run[tail:][phi_run[tail:]<200],density=True,alpha=0.5,bins=20)
# # plt.show()


# kde = KernelDensity(bandwidth = 10, kernel='gaussian')
# kde.fit(phi_run[tail:,None])


# x_d = np.linspace(0, 200, 200)

# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])

# plt.plot(x_d, np.exp(logprob))
# # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
# # plt.ylim(-0.02, 0.22)
# plt.show()

print("MSE:", np.mean((f_grid - f_grid_mean)**2))
print("TMSE:", np.mean((f_grid_run[tail:] - f_grid)**2))


# np.save("phi_run_2000_GP",phi_run)
