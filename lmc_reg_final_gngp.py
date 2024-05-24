# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:40:11 2024

@author: alier
"""

from numpy import random
import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

import time

from base import makeGrid, vec_inv

from multiLMC_generation import rmultiLMC, mult
from noisyLMC_interweaved import A_move_slice
from noisyLMC_inference import V_move_conj_scale
from LMC_multi import Z_move
from LMC_inference import phis_move
from LMC_mean import mu_move
from LMC_pred_rjmcmc import V_pred

def X_move(loc_grid,lam_current,phis_current,A_current,mu_current,A_inv_current,Rs_inv_current,A_invVmmu_current,X_current,V_current,Z_current,Y_current,D_obs,X_obs,Y_obs,n_obs):
    
    p = V_current.shape[0]
    
    V_obs_current = V_current[:,:n_obs]
    Z_obs_current = Z_current[:,:n_obs]
    
    
    
    n_new = random.poisson(lam_current)
    X_new = random.uniform(0,1,(n_new,2))
    Y_new = np.zeros(n_new,dtype=int)
    
    D_0_new = distance_matrix(X_new,X_new)
    D_current_0_new = distance_matrix(X_current,X_new)
    
    V_new = V_pred(D_0_new, D_current_0_new, phis_current, Rs_inv_current, A_current, A_invVmmu_current, mu_current, n_new)
    Z_new = V_new + random.normal(size=(p,n_new))
    
    for ii in range(n_new):
        Y_new[ii] = mult(Z_new[:,ii])
    
    X_0_current = X_new[Y_new==0]
    V_0_current = V_new[:,Y_new==0]
    Z_0_current = Z_new[:,Y_new==0]
    n_0_current = np.sum(Y_new==0)
    Y_0_current = np.zeros(n_0_current,dtype=int)
    
    D_0_current = distance_matrix(X_0_current,X_0_current)
    D_0obs_current = distance_matrix(X_0_current,X_obs)
    
    Dists_obs_current = np.block([[D_obs,np.transpose(D_0obs_current)],[D_0obs_current,D_0_current]])
    
    X_current = np.concatenate((X_obs,X_0_current),axis=0)
    n_current = n_obs + n_0_current
    Y_current = np.concatenate((Y_obs,Y_0_current),axis=0)
    
    Dists_obs_grid_current = distance_matrix(X_current,loc_grid)
    
    Z_current = np.concatenate((Z_obs_current,Z_0_current),axis=1)
    V_current = np.concatenate((V_obs_current,V_0_current),axis=1)
    
    Vmmu1_current = V_current - np.outer(mu_current,np.ones(n_current))
    
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    Rs_current = np.array([ np.exp(-Dists_obs_current*phis_current[j]) for j in range(p) ])
    Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ]) 
    
    
    return(n_current,X_current,V_current,Z_current,Y_current,Dists_obs_current,Dists_obs_grid_current,Rs_current,Rs_inv_current,Vmmu1_current,A_invVmmu1_current)


random.seed(0)
cols = ["Blues","Oranges","Greens","Reds","Purples"]
tab_cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    

### base intensity
lam = 1000
# n_grid=10
n_grid=int(np.sqrt(lam/4)-1)

### number of dimensions
p = 2
### markov chain + tail length
N = 2000
tail = 1000


### generate base poisson process

n_true = random.poisson(lam)
X_true = random.uniform(size=(n_true,2))

### grid locations
marg_grid = np.linspace(0,1,n_grid+1)
loc_grid = makeGrid(marg_grid, marg_grid)
### all locations
locs = np.concatenate((X_true,loc_grid), axis=0)

### showcase locations

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(X_true[:,0],X_true[:,1],color="black")
plt.show()


### parameters

# A #



### independent

# A = np.identity(p)

### positive correlation

line = np.ones(p)

for i in range(p):
    line[i] /= (i+1)
    
A = np.ones((p,p))

for i in range(p):
    A[i] = np.concatenate((line[i:],line[:i]))


### weird correlation

# A = np.ones((p,p))*np.sqrt(1/p)
# fac = np.ones((p,p))
# for i in range(p):
#     for j in range(i+1,p):
#         fac[i,j] = -1 
# A *= fac

### amplify signal 

A *= 2

# print(A)


phis = np.exp(np.linspace(np.log(5), np.log(25),p))
# mu = A@np.ones(p)
# mu = np.zeros(p)
mu = np.ones(p)*-1


taus = np.ones(p)
Dm1 = np.diag(taus)

Sigma = A@np.transpose(A)
Sigma_0p1 = A@np.diag(np.exp(-phis*0.1))@np.transpose(A)
Sigma_1 = A@np.diag(np.exp(-phis*1))@np.transpose(A)

### random example

Y, Z_true_all, V_true_all = rmultiLMC(A,phis,mu,locs, retZV=True) 

Y_true = Y[:n_true]
Y_grid = Y[n_true:]

Z_true = Z_true_all[:,:n_true]


V_true = V_true_all[:,:n_true]
V_grid = V_true_all[:,n_true:]

### move zeros to tail 

X_true = np.concatenate((X_true[Y_true!=0],X_true[Y_true==0]))
Y_true = np.concatenate((Y_true[Y_true!=0],Y_true[Y_true==0]))
V_true = np.concatenate((V_true[:,Y_true!=0],V_true[:,Y_true==0]),axis=1)
Z_true = np.concatenate((Z_true[:,Y_true!=0],Z_true[:,Y_true==0]),axis=1)

### fixed quantities 

n_obs = np.sum(Y_true!=0)
Y_obs = Y_true[:n_obs]
X_obs = X_true[:n_obs]

### illustrate multi process grid

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(X_true[Y_true==0,0],X_true[Y_true==0,1],color="grey")

for i in range(p):
    
    
    ax.scatter(X_true[Y_true==i+1,0],X_true[Y_true==i+1,1],color=tab_cols[i])
    
plt.show()


xv, yv = np.meshgrid(marg_grid, marg_grid)


for i in range(p):

    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_grid[i],n_grid+1), cmap = cols[i])
    plt.colorbar(c)
    plt.show()



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(Y_grid,n_grid+1), cmap = "Greys")
plt.colorbar(c)
plt.show()


### priors

# A #
sigma_A = 1.
mu_A = np.zeros((p,p))

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

### lambda

a_lam = 1000
b_lam = 1


### proposals


phis_prop = np.ones(p)*1
sigma_slice = 1



### global run containers
lam_run = np.zeros(p)
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,(n_grid+1)**2))


### acc vector
acc_phis = np.zeros((p,N))


### distance matrix


Dists_grid = distance_matrix(loc_grid,loc_grid)

### init 

# True #

n_current = np.copy(n_true)
X_current = np.copy(X_true)
Y_current = np.copy(Y_true)
V_current = np.copy(V_true)
Z_current = np.copy(Z_true)
phis_current = np.copy(phis)
mu_current = np.copy(mu)
A_current = np.copy(A)
lam_current = np.copy(lam)
V_grid_current = np.copy(V_grid)




# Arbitrary #

# phis_current = np.repeat(10.,p)
# V_current = np.zeros(shape=(p,n_true))
# mu_current = np.zeros(p)
# V_grid_current = np.zeros(shape=(p,(n_grid+1)**2))
# Z_current = np.zeros(shape=(p,n_true))
# A_current = np.identity(p)


# Random #

# phis_current = random.uniform(size=p) * 27 + 3
# V_current = random.normal(size=(p,n_tot))
# mu_current = random.normal(size=p)
# V_grid_current = random.normal(size=(p,(n_grid+1)**2))
# Z_current = random.normal(size=(p,n_tot))
# A_current = random.normal(size=(p,p))

### current state

Dists_obs_current = distance_matrix(X_current,X_current)
Dists_obs_grid_current = distance_matrix(X_current,loc_grid)

### fixed ###
Dists_obs = Dists_obs_current[:n_obs][n_obs]
###

Rs_current = np.array([ np.exp(-Dists_obs_current*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])



Vmmu1_current = V_current-np.outer(mu_current,np.ones(n_true))


A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current







st = time.time()

for i in range(N):
    
    
    n_current,X_current,V_current,Z_current,Y_current,Dists_obs_current,Dists_obs_grid_current,Rs_current,Rs_inv_current,Vmmu1_current,A_invVmmu1_current = X_move(loc_grid,lam_current,phis_current,A_current,mu_current,A_inv_current,Rs_inv_current,A_invVmmu1_current,X_current,V_current,Z_current,Y_current,Dists_obs,X_obs,Y_obs,n_obs)
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj_scale(Rs_inv_current, A_inv_current, taus, Dm1, Z_current, Z_current, V_current, Vmmu1_current, A_invVmmu1_current, mu_current)
      
    
    
    
    mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    
    
    A_current, A_inv_current, A_invVmmu1_current = A_move_slice(A_current, A_invVmmu1_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs_current,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    Z_current = Z_move(V_current,Z_current,Y_true)
    
    V_grid_current = V_pred(Dists_grid, Dists_obs_grid_current, phis_current, Rs_inv_current, A_current, A_invVmmu1_current, mu_current, (n_grid+1)**2)
    
    lam_current = random.gamma(n_current + a_lam, 1/(b_lam + 1))    

    lam_run[i] = lam_run
    mu_run[i] = mu_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    V_grid_run[i] = V_grid_current 

    
    if i % 100 == 0:
        print(i)

et = time.time()

print("Time Elapsed", (et-st)/60, "min")
print("Accept Rate for phis",np.mean(acc_phis,axis=1))


### trace plots


for i in range(p):
    plt.plot(mu_run[tail:,i])
plt.show()

print("True mu ",mu)
print("Post Mean mu ",np.mean(mu_run[tail:],axis=0))




for i in range(p):
    plt.plot(phis_run[tail:,i])
plt.show()


for i in range(p):
    for j in range(p):
        plt.plot(A_run[tail:,i,j])
plt.show()


## covariance

Sigma_run = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(N)])
print("True Sigma\n",Sigma)
print("Post Mean Sigma\n",np.mean(Sigma_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_run[tail:,i,j])
plt.show()

Sigma_0p1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*0.1))@np.transpose(A_run[i]) for i in range(N)])
print("True Sigma 0.1\n",Sigma_0p1)
print("Post Mean Sigma 0.1\n",np.mean(Sigma_0p1_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_0p1_run[tail:,i,j])
plt.show()

Sigma_1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*1))@np.transpose(A_run[i]) for i in range(N)])
print("True Sigma 1\n",Sigma_1)
print("Post Mean Sigma 1\n",np.mean(Sigma_1_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_1_run[tail:,i,j])
plt.show()




# mean processes

V_grid_mean = np.mean(V_grid_run[tail:],axis=0)


for i in range(p):
    
    


    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_grid[i],n_grid+1), cmap = cols[i%5])
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

MSE = np.mean((V_grid_run - V_grid)**2)
print("MSE = ", MSE)


### confidence interval C_12(0) and C_12(0.1)

c0l = np.quantile(Sigma_run[tail:,0,1],0.05)
c0u = np.quantile(Sigma_run[tail:,0,1],0.95)

print("C_12(0) : [", c0l,",",c0u,"]")


c0p1l = np.quantile(Sigma_0p1_run[tail:,0,1],0.05)
c0p1u = np.quantile(Sigma_0p1_run[tail:,0,1],0.95)

print("C_12(0.1) : [", c0p1l,",",c0p1u,"]")
