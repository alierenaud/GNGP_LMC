#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:13:27 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix
from scipy.special import gamma, kv

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

def matern_kernel(r, phi = 1, v = 0.4):
    r = np.abs(r)
    r[r == 0] = 1e-8
    part1 = 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * r / phi) ** v
    part3 = kv(v, np.sqrt(2 * v) * r / phi)
    return part1 * part2 * part3


# def matern_kernel(r, phi = 1):
    
#     return 0.5*(np.exp(-r/phi) + np.exp(-r/phi/2))

# def matern_kernel(r, phi = 1, alpha = 0.5):
    
#     return (1+(r/phi)**2)**(-alpha)

def fct(s):
    
    return(np.sin(s)/(0.1*s**2+1) + 1)

# random.seed(0)

# n_grid = 200
# xlim = 10

# grid_locs = np.linspace(-xlim,xlim,n_grid+1)
# f_grid = fct(grid_locs)


# n_obs = 800

# ### normal locs
# # sd_locs = 4
# # locs = sd_locs*random.normal(size = n_obs)

# ### unif locs

# locs = random.uniform(size=n_obs)*20 - 10

# f_locs = fct(locs)
# tau = 10.0
# y = f_locs + 1/np.sqrt(tau)*random.normal(size = n_obs)

# # np.prod(np.abs(locs)<10)
# # np.sum(np.abs(locs)>10)

# ### showcase data

# plt.plot(grid_locs,f_grid)
# plt.scatter(locs,y, c="black", s=10)
# plt.show()



# ### priors

# sigma2_mu = 1

# alpha_phi = 1
# beta_phi = 0.001

# alpha_a = 0.01
# beta_a = 0.1

# alpha_tau = 1
# beta_tau = 0.1


# ### proposals

# alpha_prop = 10


# ### algorithm

# mu_current = 1

# phi_current = 1000.0
# a_current = 0.1
# tau_current = 10.0

# f_currrent = random.normal(n_obs)
# f_grid_current = random.normal(n_grid+1)


# ### useful quantitites

# D = distance_matrix(np.transpose([locs]),np.transpose([locs]))

# R_current = matern_kernel(D,phi_current)
# R_inv_current = np.linalg.inv(R_current)


# D_grid_obs = distance_matrix(np.transpose([grid_locs]),np.transpose([locs]))

# R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)


# D_grid = distance_matrix(np.transpose([grid_locs]),np.transpose([grid_locs]))

# R_grid_current = matern_kernel(D_grid,phi_current)


# ### containers

# N = 1000

# f_grid_run = np.zeros((N,n_grid+1))
# phi_run = np.zeros(N)
# mu_run = np.zeros(N)
# a_run = np.zeros(N)
# tau_run = np.zeros(N)

# acc_phi = np.zeros(N)

# from time import time

# st = time()

# for i in range(N):

    

#     ### f update
    
#     Sigma_f = np.linalg.inv(a_current*R_inv_current + tau_current*np.identity(n_obs))
    
#     mu_f = Sigma_f@(a_current*R_inv_current@(mu_current*np.ones(n_obs)) + tau_current*y)
    
#     f_current = np.linalg.cholesky(Sigma_f)@random.normal(size=n_obs) + mu_f
    
    
#     ### phi update
    
#     phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
#     R_new = matern_kernel(D,phi_new)
#     R_inv_new = np.linalg.inv(R_new)
    
#     ratio = np.exp( a_current/2 * np.transpose(f_current-mu_current)@(R_inv_current-R_inv_new)@(f_current-mu_current) ) * np.linalg.det(R_current@R_inv_new)**(1/2) * np.exp(alpha_prop*(phi_new/phi_current - phi_current/phi_new)) * np.exp(beta_phi*(phi_current-phi_new)) * (phi_new/phi_current)**(alpha_phi-2*alpha_prop)
    
#     if random.uniform() < ratio:
#         phi_current = phi_new
#         R_current = R_new
#         R_inv_current = R_inv_new
        
#         acc_phi[i] = 1
    
    
#     phi_run[i] = phi_current
    
#     # ### mu update
    
#     # sigma2_cond = 1/(a_current*np.sum(R_inv_current) + 1/sigma2_mu)
#     # mu_cond = sigma2_cond*np.inner(f_current@R_inv_current, np.ones(n_obs))
    
#     # mu_current = random.normal(mu_cond,sigma2_cond)
    
#     # mu_run[i] = mu_current
    
#     # ### a update
    
#     # alpha_cond = n_obs/2+alpha_a
#     # beta_cond = np.transpose(f_current - mu_current)@R_inv_current@(f_current - mu_current)/2 + beta_a
    
#     # a_current = random.gamma(alpha_cond,1/beta_cond)
    
#     # a_run[i] = a_current
    
#     # ### tau update
    
#     # alpha_cond = n_obs/2+alpha_tau
#     # beta_cond = np.inner(y-f_current,y-f_current)/2 + beta_tau
    
#     # tau_current = random.gamma(alpha_cond,1/beta_cond)
    
#     # tau_run[i] = tau_current
    
#     ### f grid update
    
    
#     R_grid_obs_current = matern_kernel(D_grid_obs,phi_current)
#     R_grid_current = matern_kernel(D_grid,phi_current)
    
    
#     tempMat = R_grid_obs_current@R_inv_current
    
#     mu_grid = tempMat@(f_current-mu_current) + mu_current
#     Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
#     f_grid_current = np.linalg.cholesky(Sigma_grid)@random.normal(size=n_grid+1) + mu_grid
    
#     f_grid_run[i] = f_grid_current
    
#     if i%100==0:

#         # plt.plot(grid_locs,f_grid)
#         # plt.plot(grid_locs,f_grid_current)
#         # # plt.scatter(locs,y, c="black", s=10)
#         # # plt.scatter(locs,f_current, c="tab:orange", s=10)
#         # plt.show()

#         print(i)

# et = time()
# print("Time:",(et-st)/60,"minutes")

# tail = 400

# f_grid_mean = np.mean(f_grid_run[tail:], axis=0)
# f_grid_025 = np.quantile(f_grid_run[tail:], 0.025, axis=0)
# f_grid_975 = np.quantile(f_grid_run[tail:], 0.975, axis=0)


# plt.plot(grid_locs,f_grid)
# plt.plot(grid_locs,f_grid_mean)
# plt.show()

# plt.plot(grid_locs,f_grid)
# plt.plot(grid_locs,f_grid_mean)
# plt.fill_between(grid_locs, f_grid_025, f_grid_975, alpha=0.5,color="tab:orange")
# plt.show()




# print("Accept rate phi:",np.mean(acc_phi))
# ### trace plots

# # plt.plot(phi_run[tail:])
# # plt.show()

# # plt.plot(mu_run[tail:])
# # plt.show()

# # plt.plot(a_run[tail:])
# # plt.show()

# # plt.plot(tau_run[tail:])
# # plt.show()


# # #### histograms

# # plt.hist(phi_run[tail:][phi_run[tail:]<200],density=True,alpha=0.5,bins=20)
# # # plt.show()


# # kde = KernelDensity(bandwidth = 10, kernel='gaussian')
# # kde.fit(phi_run[tail:,None])


# # x_d = np.linspace(0, 200, 200)

# # # score_samples returns the log of the probability density
# # logprob = kde.score_samples(x_d[:, None])

# # plt.plot(x_d, np.exp(logprob))
# # # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
# # # plt.ylim(-0.02, 0.22)
# # plt.show()





