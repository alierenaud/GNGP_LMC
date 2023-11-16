#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:13:27 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix


import matplotlib.pyplot as plt


from base import matern_kernel, fct


random.seed(0)

n_grid = 200
xlim = 10

grid_locs = np.linspace(-xlim,xlim,n_grid+1)
f_grid = fct(grid_locs)


n_obs = 200

### normal locs
# sd_locs = 4
# locs = sd_locs*random.normal(size = n_obs)

### unif locs

locs = random.uniform(size=n_obs)*20 - 10

f_locs = fct(locs)
tau = 10.0
y = f_locs + 1/np.sqrt(tau)*random.normal(size = n_obs)

# np.prod(np.abs(locs)<10)
# np.sum(np.abs(locs)>10)

### showcase data

plt.figure(figsize=(5,3.5))
plt.plot(grid_locs,f_grid)
plt.scatter(locs,y, c="black", s=10)
# plt.savefig("datShow.pdf", bbox_inches='tight')
plt.show()


### exp correlations

# ds = np.linspace(0,1,401)

# plt.figure(figsize=(5,3.5))
# plt.plot(ds,np.exp(-ds*3),label="phi=1/3")
# plt.plot(ds,np.exp(-ds*30),label="phi=1/30")
# plt.xlabel("distance")
# plt.ylabel("correlation")
# plt.legend(loc='upper right')
# # plt.savefig("covShow.pdf", bbox_inches='tight')
# plt.show()




### priors

sigma2_mu = 1

alpha_phi = 100
beta_phi = 1

alpha_a = 0.01
beta_a = 0.1

alpha_tau = 1
beta_tau = 0.1


### proposals

# alpha_prop = 100
# phi_prop = 10

phi_prop = 10

### algorithm

mu_current = 0

phi_current = 100.0
a_current = 1
tau_current = 10.0

# f_currrent = random.normal(size=n_obs)
# f_grid_current = random.normal(size=n_grid+1)

f_currrent = np.zeros(shape=n_obs)
f_grid_current = np.zeros(shape=n_grid+1)

### useful quantitites

D = distance_matrix(np.transpose([locs]),np.transpose([locs]))

R_current = matern_kernel(D,phi_current)
R_inv_current = np.linalg.inv(R_current)


D_grid_obs = distance_matrix(np.transpose([grid_locs]),np.transpose([locs]))

R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)


D_grid = distance_matrix(np.transpose([grid_locs]),np.transpose([grid_locs]))

R_grid_current = matern_kernel(D_grid,phi_current)


### containers

N = 2000

f_grid_run = np.zeros((N,n_grid+1))
phi_run = np.zeros(N)
mu_run = np.zeros(N)
a_run = np.zeros(N)
tau_run = np.zeros(N)

acc_phi = np.zeros(N)

from time import time

st = time()

for i in range(N):

    if i%500==0:
        
        plt.figure(figsize=(5,3.5))
        plt.plot(grid_locs,f_grid)
        plt.plot(grid_locs,f_grid_current)
        # plt.scatter(locs,y, c="black", s=10)
        # plt.scatter(locs,f_current, c="tab:orange", s=10)
        plt.savefig("proc"+str(i)+".pdf", bbox_inches='tight')
        # plt.show()

        print(i)

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
    
    sus = np.exp( a_current/2 * np.transpose(f_current-mu_current)@(R_inv_current-R_inv_new)@(f_current-mu_current) )
    
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
        R_current = R_new
        R_inv_current = R_inv_new
        
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
    
    f_grid_current = np.sqrt(1/a_current)*np.linalg.cholesky(Sigma_grid)@random.normal(size=n_grid+1) + mu_grid
    
    f_grid_run[i] = f_grid_current
    
    

et = time()
print("Time:",(et-st)/60,"minutes")

tail = 1000

f_grid_mean = np.mean(f_grid_run[tail:], axis=0)
f_grid_025 = np.quantile(f_grid_run[tail:], 0.025, axis=0)
f_grid_975 = np.quantile(f_grid_run[tail:], 0.975, axis=0)


plt.figure(figsize=(5,3.5))
plt.plot(grid_locs,f_grid)
plt.plot(grid_locs,f_grid_mean)
# plt.savefig("procMean.pdf", bbox_inches='tight')
plt.show()

plt.figure(figsize=(5,3.5))
plt.plot(grid_locs,f_grid)
plt.plot(grid_locs,f_grid_mean)
plt.fill_between(grid_locs, f_grid_025, f_grid_975, alpha=0.5,color="tab:orange")
# plt.title("n=250, MSE=0.0131, Time=0.3308 min")
plt.savefig("procInt.pdf", bbox_inches='tight')
plt.show()



# nsss = np.array([250,500,1000,2000])
# tsss = np.array([0.288,1.0992,5.6082,38.5860])

# plt.plot(nsss,tsss)


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



