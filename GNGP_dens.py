#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:58:30 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix
from scipy.stats import truncnorm


import matplotlib.pyplot as plt


from base import matern_kernel, fct

from scipy.stats import norm

random.seed(0)

n_grid = 200
m = 20
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
plt.hist(x_1,density=True,alpha=0.5,bins=20)
# plt.plot(grid_locs,f_grid)
plt.show()
plt.violinplot(x_1)
plt.show()   





### priors

alpha_phi = 100
beta_phi = 100



### proposals

alpha_prop = 100


### algorithm

mu = 0
mu_obs = np.ones(n_1)*mu
mu_grid = np.ones(n_grid+1)*mu
a = 1.0
tau = 1.0

phi_current = 1


### initiiate point process
# n_0_current = n_1
# x_0_current = random.uniform(size=n_1) * 20 -10
# g_0_current = random.normal(size=n_0_current) + mu
# y_0_current = -np.ones(n_0_current)

n_0_current = n_0_true
x_0_current = x_0_true
g_0_current = g_0_true
y_0_current = y_0_true

g_1_current = g_1_true
y_1_current = y_1_true

# g_1_current = random.normal(size=n_1) + mu
# y_1_current = np.ones(n_1)


n_current = n_1 + n_0_current 
x_current = np.append(x_1,x_0_current)
g_current = np.append(g_1_current,g_0_current)
y_current = np.append(y_1_current,y_0_current)

# g_grid_current = random.normal(size=n_grid+1)
g_grid_current = f_grid


###### compute grid neighbors

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

Distg = distance_matrix(np.transpose([grid_locs]), np.transpose([grid_locs]))

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

ogNei = np.zeros(n_current,dtype=object)
aogNei = np.zeros(n_grid+1,dtype=object)

for i in range(n_grid+1):
    aogNei[i] = np.empty(0,dtype=int)

# Distog = distance_matrix(np.transpose([x_current]), np.transpose([grid_locs]))

Distogs = np.zeros(n_current,dtype=object)


for i in range(n_current):
    
    
    
    
    
    leftNei = int(np.floor( (x_current[i] + xlim)/(2*xlim) * n_grid))
    
    off_left = -min(leftNei - m//2 + 1,0)
    off_right = max(leftNei + m//2,n_grid) - n_grid
    
    leftMostNei = leftNei-m//2+1+off_left-off_right
    rightMostNei = leftNei + m//2 + 1 + off_left - off_right
    
    nei_temp = np.arange(leftMostNei,rightMostNei)
    
    ogNei[i] = nei_temp
    
    Distogs[i] = distance_matrix([[x_current[i]]], np.transpose([grid_locs[nei_temp]]))
    
    for j in np.arange(leftMostNei,rightMostNei):
        aogNei[j] = np.append(aogNei[j],i)
    

    
DistggMat = Distg[:m][:,:m]
Bog_current = np.zeros((n_current,n_grid+1))
rog_current = np.zeros(n_current)


for i in range(n_current):

    
    
    CovMatgg_temp = matern_kernel(DistggMat,phi_current)
    
    R_inv_temp = np.linalg.inv(CovMatgg_temp)
    
    DistMatog_temp = Distogs[i]
    r_temp = matern_kernel(DistMatog_temp,phi_current)
    

    
    b_temp = r_temp@R_inv_temp
    

    Bog_current[i][ogNei[i]] = b_temp
    
    rog_current[i] = 1-np.inner(b_temp,r_temp)


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
        plt.scatter(x_0_current, np.zeros(n_0_current), c="black", marker="|")
        plt.plot(grid_locs, norm.cdf(g_grid_current), c="tab:orange")
        
        
        plt.scatter(x_1, norm.cdf(g_1_current), s=10, c="tab:blue")
        plt.scatter(x_0_current, norm.cdf(g_0_current), s=10, c="tab:orange")
        
        
        plt.show()

        print(i)

        

    ### point process (update x_0,g_0,y_0)
    
    count = 0
    x_cont= []
    g_cont= []
    y_cont= []
    
    ogNei_cont = []
    aogNei_cont = []
    
    Distogs_cont = []
    
    Bog_cont = []
    rog_cont = []
    
    while True:
        
        
        ### simulate n_1 new variables 
        x_new = random.uniform(size=3*n_1) * 20 - 10
        n_new = x_new.shape[0]
        
        ### compute obs neighbors on grid

        ogNei_new = np.zeros(n_new,dtype=object)
        aogNei_new = np.zeros(n_grid+1,dtype=object)

        for ii in range(n_grid+1):
            aogNei_new[ii] = np.empty(0,dtype=int)

        # Distog = distance_matrix(np.transpose([x_current]), np.transpose([grid_locs]))

        Distogs_new = np.zeros(n_new,dtype=object)


        for ii in range(n_new):
            
            
            
            
            
            leftNei = int(np.floor( (x_new[ii] + xlim)/(2*xlim) * n_grid))
            
            off_left = -min(leftNei - m//2 + 1,0)
            off_right = max(leftNei + m//2,n_grid) - n_grid
            
            leftMostNei = leftNei-m//2+1+off_left-off_right
            rightMostNei = leftNei + m//2 + 1 + off_left - off_right
            
            nei_temp = np.arange(leftMostNei,rightMostNei)
            
            ogNei_new[ii] = nei_temp
            
            Distogs_new[ii] = distance_matrix([[x_new[ii]]], np.transpose([grid_locs[nei_temp]]))
            
            for jj in np.arange(leftMostNei,rightMostNei):
                aogNei_new[jj] = np.append(aogNei_new[jj],ii)
            

            
        
        Bog_new = np.zeros((n_new,n_grid+1))
        rog_new = np.zeros(n_new)


        for ii in range(n_new):

            
            
            CovMatgg_temp = matern_kernel(DistggMat,phi_current)
            
            R_inv_temp = np.linalg.inv(CovMatgg_temp)
            
            DistMatog_temp = Distogs_new[ii]
            r_temp = matern_kernel(DistMatog_temp,phi_current)
            

            
            b_temp = r_temp@R_inv_temp
            

            Bog_new[ii][ogNei_new[ii]] = b_temp
            
            rog_new[ii] = 1-np.inner(b_temp,r_temp)
        
        
        
        g_new = np.zeros(n_new)
        
                
    
        for ii in range(n_new):
            g_new[ii] = np.sqrt(rog_new[ii])*random.normal() + np.inner(Bog_new[ii,ogNei_new[ii]],g_grid_current[ogNei_new[ii]] - mu_grid[ogNei_new[ii]])
    
        y_new = g_new + random.normal(size=n_new)
        

        x_cont = np.append(x_cont, x_new)  
        g_cont = np.append(g_cont, g_new) 
        y_cont = np.append(y_cont, y_new) 
        
        ogNei_cont = np.append(ogNei_cont, ogNei_new)
        aogNei_cont = np.append(aogNei_cont, aogNei_new)
        
        Distogs_cont = np.append(Distogs_cont, Distogs_new)
        
        Bog_cont = np.append(Bog_cont, Bog_new)
        rog_cont = np.append(rog_cont, rog_new)
                
        count += np.sum(y_new>0)
        
        if count >= n_1:
            
            n_tail = np.where(y_cont > 0)[0][n_1-1]
            
            x_cont = x_cont[:n_tail+1]
            g_cont = g_cont[:n_tail+1]
            y_cont = y_cont[:n_tail+1]
            
            thinned = y_cont < 0 
            
            
            n_0_current = np.sum(thinned)
            x_0_current = x_cont[thinned]
            g_0_current = g_cont[thinned]
            y_0_current = y_cont[thinned]
            
          
            n_current = n_1 + n_0_current 
            x_current = np.append(x_1,x_0_current)
            g_current = np.append(g_1_current,g_0_current)
            y_current = np.append(y_1_current,y_0_current)
            
            
            ogNei = np.append(ogNei[:n_1], ogNei_cont)
            aogNei = np.append(aogNei[:n_1], aogNei_cont)
            
            Distogs = np.append(Distogs[:n_1], Distogs_new)
            
            Bog_current = np.append(Bog_current[:n_1], Bog_new)
            rog_current = np.append(rog_current[:n_1], rog_new)
            
            
            break
        
        else:
            print("not enough points")
            


    # ### f update
    
    # Sigma_f = np.linalg.inv(a*R_inv_current + tau*np.identity(n_current))
    
    # mu_f = Sigma_f@(a*R_inv_current@(mu*np.ones(n_current)) + tau*y_current)
    
    
    
    # try:
    #     C = np.linalg.cholesky(Sigma_f)
    #     g_current = C@random.normal(size=n_current) + mu_f
    # except np.linalg.LinAlgError:
    #     print("oof g_current")
    
    # g_0_current = g_current[n_1:]
    # g_1_current = g_current[:n_1]
    
    
    
    # ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    # R_new = matern_kernel(D_current,phi_new)
    # R_inv_new = np.linalg.inv(R_new)
    
    # sus = np.exp( a/2 * np.transpose(g_current-mu)@(R_inv_current-R_inv_new)@(g_current-mu) )
    
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
    #     R_current = np.copy(R_new)
    #     R_inv_current = np.copy(R_inv_new)
        
    #     acc_phi[i] = 1
    
    
    # phi_run[i] = phi_current
    
    # ### y_update
    
    # for ii in range(n_1):
        
    #     y_current[ii] = truncnorm.rvs(a=-g_current[ii],b=np.inf,loc=g_current[ii])
        
    # for ii in range(n_1,n_current):
        
    #     y_current[ii] = truncnorm.rvs(a=-np.inf,b=-g_current[ii],loc=g_current[ii])
    
    # y_0_current = y_current[n_1:]
    # y_1_current = y_current[:n_1]
    
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
    
    # R_grid_obs_current = matern_kernel(D_grid_obs_current,phi_current)
    # R_grid_current = matern_kernel(D_grid,phi_current)
    
    
    # tempMat = R_grid_obs_current@R_inv_current
    
    # mu_grid = tempMat@(g_current-mu) + mu
    # Sigma_grid = R_grid_current - tempMat@np.transpose(R_grid_obs_current)
    
    # try:
    #     C = np.linalg.cholesky(Sigma_grid)
    #     g_grid_current = C/np.sqrt(a)@random.normal(size=n_grid+1) + mu_grid
         
    # except np.linalg.LinAlgError:
    #     print("oof g_grid")
    

    # g_grid_run[i] =g_grid_current
    


et = time()
print("Time:",(et-st)/60,"minutes")

tail = 400

Phi_g_grid_run = norm.cdf(g_grid_run)

Phi_g_grid_mean = np.mean(Phi_g_grid_run[tail:], axis=0)
Phi_g_grid_025 = np.quantile(Phi_g_grid_run[tail:], 0.05, axis=0)
Phi_g_grid_975 = np.quantile(Phi_g_grid_run[tail:], 0.95, axis=0)


plt.plot(grid_locs,norm.cdf(f_grid))
plt.plot(grid_locs,Phi_g_grid_mean)
plt.show()




plt.plot(grid_locs,norm.cdf(f_grid))
plt.plot(grid_locs,Phi_g_grid_mean)
plt.fill_between(grid_locs, Phi_g_grid_025, Phi_g_grid_975, alpha=0.5,color="tab:orange")
plt.show()



print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

plt.boxplot(phi_run[tail:])
plt.show()

plt.hist(x_1,density=True,alpha=0.5,bins=20)
plt.show()

plt.violinplot(x_1,vert=False)
plt.show()