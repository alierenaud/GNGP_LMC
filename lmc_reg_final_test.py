# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:25:37 2024

@author: alier
"""


from numpy import random
import numpy as np


from scipy.stats import beta
from scipy.spatial import distance_matrix


import matplotlib.pyplot as plt


import time


from base import makeGrid, vec_inv
from noisyLMC_generation import rNLMC_mu
from noisyLMC_interweaved import A_move_slice
from noisyLMC_inference import V_move_conj_scale, taus_move
from LMC_inference import phis_move
from LMC_mean import mu_move
from LMC_pred_rjmcmc import V_pred

random.seed(0)

cols = ["Blues","Oranges","Greens","Reds","Purples"]

### number of points 
n_obs=2000
# n_grid=20
n_grid=int(np.sqrt(n_obs)-1)


### number of dimensions
p = 2
### markov chain + tail length
N = 10000
tail = 0


### generate uniform locations

conc = 1
loc_obs = beta.rvs(conc, conc, size=(n_obs,2))

### grid locations
marg_grid = np.linspace(0,1,n_grid+1)
loc_grid = makeGrid(marg_grid, marg_grid)
### all locations
locs = np.concatenate((loc_obs,loc_grid), axis=0)

### showcase locations

# fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black")
# plt.show()

### parameters

# A #

A = np.ones((p,p))*np.sqrt(1/p)
fac = np.ones((p,p))
for i in range(p):
    for j in range(i+1,p):
        fac[j,i] = -1 
A *= fac

# print(A)


phis = np.exp(np.linspace(np.log(5), np.log(25),p))
# mu = A@np.ones(p)
mu=np.zeros(p)

noise_sd = 0.5
taus_sqrt_inv = np.ones(p)*noise_sd
taus = 1/taus_sqrt_inv**2

Sigma = A@np.transpose(A)
Sigma_0p1 = A@np.diag(np.exp(-phis*0.1))@np.transpose(A)
Sigma_1 = A@np.diag(np.exp(-phis*1))@np.transpose(A)



### generate LMC

Y, V_true = rNLMC_mu(A,phis,taus_sqrt_inv,mu,locs, retV=True) 

Y_obs = Y[:,:n_obs]

V_true_obs = V_true[:,:n_obs]
V_true_grid = V_true[:,n_obs:]


### illustrate processes

# for i in range(p):

#     xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
#     fig, ax = plt.subplots()
#     # ax.set_xlim(0,1)
#     # ax.set_ylim(0,1)
#     ax.set_box_aspect(1)
    
    
    
#     c = ax.pcolormesh(xv, yv, vec_inv(V_true_grid[i],n_grid+1), cmap = cols[i])
#     plt.colorbar(c)
#     # plt.savefig("aaaaa.pdf", bbox_inches='tight')
#     plt.show()






### priors

# A #
sigma_A = 0.2
mu_A = np.zeros((p,p))
# mu_A = np.copy(A)

# phi #
min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi

alphas = np.ones(p)
betas = np.ones(p)

## tau
a = 1
b = 1

### mu 

mu_mu = np.zeros(p)
sigma_mu = 1


### proposals


phis_prop = np.ones(p)*0.5
sigma_slice = 1



### global run containers
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,(n_grid+1)**2))


### acc vector
acc_phis = np.zeros((p,N))

### distance matrix

Dists_obs = distance_matrix(loc_obs,loc_obs)
Dists_obs_grid = distance_matrix(loc_obs,loc_grid)
Dists_grid = distance_matrix(loc_grid,loc_grid)

### init 

# True #

phis_current = np.copy(phis)
V_current = np.copy(V_true_obs)
mu_current = np.copy(mu)
V_grid_current = np.copy(V_true_grid)
taus_current = np.copy(taus)
A_current = np.copy(A)

# Random #

# phis_current = np.repeat(10.,p)
# V_current = np.zeros(shape=(p,n_obs))
# mu_current = np.zeros(p)
# V_grid_current = np.zeros(shape=(p,(n_grid+1)**2))
# taus_current = np.ones(p)
# A_current = np.identity(p)

### current state

Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])



VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])


Vmmu1_current = V_current-np.outer(mu_current,np.ones(n_obs))


A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current


Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs



st = time.time()

for i in range(N):
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj_scale(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y_obs, V_current, Vmmu1_current, A_invVmmu1_current, mu_current)
      
    
    
    
    # mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    
    
    A_current, A_inv_current, A_invVmmu1_current = A_move_slice(A_current, A_invVmmu1_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

    
    V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invVmmu1_current, mu_current, (n_grid+1)**2)
    
        


    mu_run[i] = mu_current
    V_run[i] = V_current
    taus_run[i] = taus_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    V_grid_run[i] = V_grid_current 

    
    if i % 10 == 0:
        ett = time.time()
        est_time = (ett-st)*(N-i-1)/(i+1)
        est_h = int(est_time//(60**2))
        est_m = int(est_time%(60**2)//60)
        print(i, "Est. Time Remaining:", est_h, "h", est_m, "min")

et = time.time()
ela_time = (et-st)
ela_h = int(ela_time//(60**2))
ela_m = int(ela_time%(60**2)//60)
print(N, "Time Elapsed:", ela_h, "h", ela_m, "min")

print("Accept Rate for phis",np.mean(acc_phis,axis=1))



# mu_run = np.load("run2000reg_mu.npy")
# taus_run = np.load("run2000reg_taus.npy")
# phis_run = np.load("run2000reg_phis.npy")
# A_run = np.load("run2000reg_A.npy")
# V_grid_run = np.load("run2000reg_V_grid.npy")
# V_run = np.load("run2000reg_V_current.npy")



### trace plots




for i in range(p):
    plt.plot(mu_run[tail:,i])
plt.show()

print("True mu ",mu)
print("Post Mean mu ",np.mean(mu_run[tail:],axis=0))

for i in range(p):
    plt.plot(taus_run[tail:,i])
plt.show()

print("True taus ",taus)
print("Post Mean taus ",np.mean(taus_run[tail:],axis=0))


for i in range(p):
    plt.plot(phis_run[tail:,i])
plt.show()

# print("True phis ",phis)
# print("Post Mean phis ",np.mean(phis_run[tail:],axis=0))

for i in range(p):
    for j in range(p):
        plt.plot(A_run[tail:,i,j])
plt.show()


## covariance

Sigma_run = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(N)])
print("True Sigma\n",Sigma)
print("Post Median Sigma\n",np.median(Sigma_run[tail:],axis=0))
print("Post 0.05 Sigma\n",np.quantile(Sigma_run[tail:],0.05,axis=0))
print("Post 0.95 Sigma\n",np.quantile(Sigma_run[tail:],0.95,axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_run[tail:,i,j])
plt.show()

Sigma_0p1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*0.1))@np.transpose(A_run[i]) for i in range(N)])
print("True Sigma 0.1\n",Sigma_0p1)
print("Post Median Sigma 0.1\n",np.median(Sigma_0p1_run[tail:],axis=0))
print("Post 0.05 Sigma 0.1\n",np.quantile(Sigma_0p1_run[tail:],0.05,axis=0))
print("Post 0.95 Sigma 0.1\n",np.quantile(Sigma_0p1_run[tail:],0.95,axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_0p1_run[tail:,i,j])
plt.show()

# Sigma_1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*1))@np.transpose(A_run[i]) for i in range(N)])
# print("True Sigma 1\n",Sigma_1)
# print("Post Mean Sigma 1\n",np.mean(Sigma_1_run[tail:],axis=0))

# for i in range(p):
#     for j in range(i,p):
#         plt.plot(Sigma_1_run[tail:,i,j])
# plt.show()

## id parameters

id_param_true = np.zeros((p,p))

for i in range(p):
    for j in range(p):
        id_param_true[i,j] = A[i,j]**2*phis[j]

id_param_run = np.zeros((N,p,p))

for i in range(p):
    for j in range(p):
        id_param_run[:,i,j] = A_run[:,i,j]**2*phis_run[:,j]

print("True id param:\n", id_param_true)
print("Median id param:\n", np.median(id_param_run[tail:],axis=0))
print("0.05 id param:\n", np.quantile(id_param_run[tail:],0.05,axis=0))
print("0.95 id param:\n", np.quantile(id_param_run[tail:],0.95,axis=0))


# mean processes

V_grid_mean = np.mean(V_grid_run[tail:],axis=0)


for i in range(p):
    
    


    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_true_grid[i],n_grid+1), cmap = cols[i%5])
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()

    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_grid_mean[i],n_grid+1), cmap = cols[i%5])
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()

MSE = np.mean((V_grid_run - V_true_grid)**2)
print("MSE = ", MSE)

### CI cover ###

V_grid_05 = np.quantile(V_grid_run[tail:],0.05,axis=0)
V_grid_95 = np.quantile(V_grid_run[tail:],0.95,axis=0)

cover_global = np.mean((V_true_grid > V_grid_05) & (V_true_grid < V_grid_95))

# print("90 cover 1:", cover_1)
# print("90 cover 2:", cover_2)
print("90 cover T:", cover_global)

### CI width ###

width_global = np.mean(V_grid_95 - V_grid_05)

# print("90 width 1:", width_1)
# print("90 width 2:", width_2)
print("90 width T:", width_global)

print("T/I:",(et-st)/N,"n =",n_obs,"p =",p,"N =", N)


### confidence interval C_12(0) and C_12(0.1)

# c0l = np.quantile(Sigma_run[tail:,0,1],0.05)
# c0u = np.quantile(Sigma_run[tail:,0,1],0.95)

# print("C_12(0) : [", c0l,",",c0u,"]")


# c0p1l = np.quantile(Sigma_0p1_run[tail:,0,1],0.05)
# c0p1u = np.quantile(Sigma_0p1_run[tail:,0,1],0.95)

# print("C_12(0.1) : [", c0p1l,",",c0p1u,"]")


# np.save("run2000reg_mu.npy",mu_run)
# np.save("run2000reg_taus.npy",taus_run)
# np.save("run2000reg_phis.npy",phis_run)
# np.save("run2000reg_A.npy",A_run)
# np.save("run2000reg_V_grid.npy",V_grid_run)
# np.save("run2000reg_V_current.npy",V_run)


