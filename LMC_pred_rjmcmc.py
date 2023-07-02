# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:53:27 2023

@author: alier
"""

import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from scipy.stats import beta

from noisyLMC_interweaved import A_move_slice
from noisyLMC_interweaved import makeGrid
from noisyLMC_interweaved import vec_inv

from noisyLMC_inference import V_move_conj, taus_move

from LMC_inference import phis_move

random.seed(100)

RJMCMC = True
def A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice):

    
    p = A_current.shape[0] 
    n = A_invV_current.shape[1]
    
    ### threshold
    z =  -1/2 * np.sum( [A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_current))) - 1/2/sigma_A**2 * np.sum((A_current-mu_A)**2) - random.exponential(1,1)
    
    L = A_current - random.uniform(0,sigma_slice,(p,p))
    # L[0] = np.maximum(L[0],0)
    
    U = L + sigma_slice
    
    L *= A_mask_current
    U *= A_mask_current
        
    while True:
    
        
        
        A_prop = random.uniform(L,U)
        A_inv_prop = np.linalg.inv(A_prop)
        A_invV_prop = A_inv_prop @ V_current
        
        acc = z < -1/2 * np.sum( [A_invV_prop[j] @ Rs_inv_current[j] @ A_invV_prop[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_prop))) - 1/2/sigma_A**2 * np.sum((A_prop-mu_A)**2) 
            
        if acc:
            return(A_prop,A_inv_prop,A_invV_prop)
        else:
            for ii in range(p):
                for jj in range(p):
                    if A_prop[ii,jj] < A_current[ii,jj]:
                        L[ii,jj] = A_prop[ii,jj]
                    else:
                        U[ii,jj] = A_prop[ii,jj]



### number of points 
n_obs=100
# n_grid=400  ### 1D grid
n_grid=20  ### 2D Grid

### global parameters



### generate random example
loc_obs = random.uniform(0,1,(n_obs,2))

### beta locations
# loc_obs = beta.rvs(2, 1, size=(n_obs,2))

### showcase locations

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black")
plt.show()

# loc_grid = np.transpose([np.linspace(0.5, 1, n_grid+1)])
loc_grid = makeGrid(n_grid)

locs = np.concatenate((loc_obs,loc_grid), axis=0)


### parameters
# A = np.array([[np.sqrt(1/2),0,np.sqrt(1/2)],
#               [-np.sqrt(1/2),0,np.sqrt(1/2)],
#               [0,1.,0]])
# p = A.shape[0]
# phis = np.array([5.,10.,20.])
# taus_sqrt_inv = np.array([1.,1.,1.]) 


### 5D example

# A = random.normal(size=(5,5))

### Triangular

A = np.array([[1,0,0,0,0],
              [-np.sqrt(1/2),-np.sqrt(1/2),0,0,0],
              [np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3),0,0],
              [-np.sqrt(1/4),-np.sqrt(1/4),-np.sqrt(1/4),-np.sqrt(1/4),0],
              [np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5),np.sqrt(1/5)]])
p = A.shape[0]


### Full

# A = np.ones((5,5))*np.sqrt(1/5)
# A *= np.array([[1,-1,-1,-1,-1],
#                 [1,1,-1,-1,-1],
#                 [1,1,1,-1,-1],
#                 [1,1,1,1,-1],
#                 [1,1,1,1,1]])
# p = A.shape[0]


### Block Diagonal

# A = np.array([[np.sqrt(2/3),np.sqrt(1/3),0,0,0],
#               [-np.sqrt(2/3),np.sqrt(1/3),0,0,0],
#               [0,0,1.,0,0],
#               [0,0,0,np.sqrt(2/3),np.sqrt(1/3)],
#               [0,0,0,np.sqrt(2/3),-np.sqrt(1/3)]])
# p = A.shape[0]



### Diagonal

# p = 5
# A = np.identity(p)


phis = np.exp(np.linspace(np.log(5), np.log(25),5))
taus_sqrt_inv = np.array([1.,1.,1.,1.,1.]) 


### generate rfs

Y_true, V_true = rNLMC(A,phis,taus_sqrt_inv,locs, retV=True)

Y_obs = Y_true[:,:n_obs]

# V_obs = V_true[:,:n_obs]
V_grid = V_true[:,n_obs:]


### showcase

# plt.plot(loc_grid,V_grid[0],loc_grid,V_grid[1])
# plt.show()
# plt.plot(loc_obs,V_obs[0],'o',c="tab:blue",markersize=2)
# plt.plot(loc_obs,V_obs[1],'o',c="tab:orange",markersize=2)
# plt.show()


locs1D = (np.arange(n_grid) + 0.5)/n_grid
xv, yv = np.meshgrid(locs1D, locs1D)

### process 1

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid[0],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Blues")
plt.show()

### process 2

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid[1],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Oranges")
plt.show()

### process 3

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid[2],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Greens")
plt.show()

### process 4

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid[3],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Reds")
plt.show()

### process 5

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid[4],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Purples")
plt.show()


### showcase crosscovariance

max_d = 1
res = 100

ds = np.linspace(0,max_d,res)

def crossCov(d,A,phis,i,j):
    return(np.sum(A[i] * A[j] * np.exp(-d*phis)))
    
cc = np.zeros(res)

fig, axs = plt.subplots(5, 2, figsize=(8, 12))

i=0
j=1

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[0, 0].plot(ds,cc, c="black")
axs[0, 0].set_title(str(i+1) + " and " + str(j+1))

i=0
j=2

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[0, 1].plot(ds,cc, c="black")
axs[0, 1].set_title(str(i+1) + " and " + str(j+1))

i=0
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[1, 0].plot(ds,cc, c="black")
axs[1, 0].set_title(str(i+1) + " and " + str(j+1))

i=0
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[1, 1].plot(ds,cc, c="black")
axs[1, 1].set_title(str(i+1) + " and " + str(j+1))

i=1
j=2

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[2, 0].plot(ds,cc, c="black")
axs[2, 0].set_title(str(i+1) + " and " + str(j+1))

i=1
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[2, 1].plot(ds,cc, c="black")
axs[2, 1].set_title(str(i+1) + " and " + str(j+1))

i=1
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[3, 0].plot(ds,cc, c="black")
axs[3, 0].set_title(str(i+1) + " and " + str(j+1))

i=2
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[3, 1].plot(ds,cc, c="black")
axs[3, 1].set_title(str(i+1) + " and " + str(j+1))

i=2
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[4, 0].plot(ds,cc, c="black")
axs[4, 0].set_title(str(i+1) + " and " + str(j+1))


i=3
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
axs[4, 1].plot(ds,cc, c="black")
axs[4, 1].set_title(str(i+1) + " and " + str(j+1))


# plt.savefig('crosscov.pdf') 
plt.tight_layout()
plt.show()


### priors
sigma_A = 10.
mu_A = np.zeros((p,p))
# mu_A = np.array([[0.,0.,0.],
#                  [0.,0.,0.],
#                  [0.,0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi

alphas = np.ones(p)
betas = np.ones(p)

### RJMCMC

prob_one = 0.5

### RJMCMC



n_ones_current = p**2
A_mask_current = np.ones((p,p))

def pairs(p):
    a = []
    k = 0
    for i in range(p):
        for j in range(p):
            a.append((i,j))
            k += 1
    
    return(a)
        
A_ones_ind_current = pairs(p)
A_zeros_ind_current = []

## tau

a = 1
b = 1

### proposals


phis_prop = np.ones(p)*3.0
sigma_slice = 10


def ins_prob(n_ones,p):
    
    if n_ones == p**2:
        return(0)
    elif n_ones == p:
        return(1)
    else:
        return(0.5)

n_jumps = p


### samples
N = 10000
tail = 4000

### global run containers
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,n_grid**2))


### acc vector
acc_phis = np.zeros((p,N))

### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)
Dists_obs_grid = distance_matrix(loc_obs,loc_grid)
Dists_grid = distance_matrix(loc_grid,loc_grid)


### init and current state
phis_current = np.repeat(10.,p)
Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


V_current = random.normal(size=(p,n_obs))*1
VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])


A_current = random.normal(size=(p,p))
A_inv_current = np.linalg.inv(A_current)

A_invV_current = A_inv_current @ V_current

taus_current = 1/taus_sqrt_inv**2
Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs

import time

st = time.time()

for i in range(N):
    
    
    V_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current)
  
    
                        
    if RJMCMC:
        
        A_current, A_inv_current, A_invV_current = A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
    
    else:
        
        A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
    
        
        
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_current,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)

    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

    
    if RJMCMC:
        ### reversible jumps
        
        for j in range(n_jumps):
            
            
            insert = random.binomial(1, ins_prob(n_ones_current,p))
            
            if insert:
                
                rand_int = random.choice(range(p**2 - n_ones_current))
                rand_ind = A_zeros_ind_current[rand_int]
                new_elem = random.normal(mu_A[rand_ind],sigma_A,1)
                
                A_new = np.copy(A_current)
                A_new[rand_ind] = new_elem
                
                A_inv_new = np.linalg.inv(A_new)
                
                A_invV_new = A_inv_new @ V_current
                
                rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n_obs * (1-ins_prob(n_ones_current+1,p))/ins_prob(n_ones_current,p) * (p**2 - n_ones_current)/(n_ones_current + 1) * prob_one / (1-prob_one)
                
                if random.uniform() < rat:
                    
                    A_current = A_new
                    A_inv_current = A_inv_new
                    A_invV_current = A_invV_new
                    
                    n_ones_current += 1
                    
                    A_mask_current[rand_ind] = 1.
                    
                    A_ones_ind_current.append(A_zeros_ind_current.pop(rand_int))
                    
                
                
            else:
                
                rand_int = random.choice(range(n_ones_current))
                rand_ind = A_ones_ind_current[rand_int]
                
                A_new = np.copy(A_current)
                A_new[rand_ind] = 0.
                
                if np.linalg.det(A_new) != 0:
                    A_inv_new = np.linalg.inv(A_new)
                    
                    A_invV_new = A_inv_new @ V_current
                    
                    rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n_obs * ins_prob(n_ones_current-1,p)/(1-ins_prob(n_ones_current,p)) * (n_ones_current)/(p**2 - n_ones_current + 1) * (1-prob_one)/prob_one
                    
                    if random.uniform() < rat:
                        
                        A_current = A_new
                        A_inv_current = A_inv_new
                        A_invV_current = A_invV_new
                        
                        n_ones_current -= 1
                        
                        A_mask_current[rand_ind] = 0.
                        
                        A_zeros_ind_current.append(A_ones_ind_current.pop(rand_int))


    
    ### make pred cond on current phis, A
    
    
    
    rs = np.array([ np.exp(-Dists_obs_grid*phis_current[j]) for j in range(p) ])
    Rs_prime = np.array([ np.exp(-Dists_grid*phis_current[j]) for j in range(p) ])
    
    Rinvsrs = np.array([ Rs_inv_current[j]@rs[j] for j in range(p) ])
    
    Cs = np.array([ np.linalg.cholesky(Rs_prime[j] - np.transpose(rs[j])@Rinvsrs[j]) for j in range(p) ])
    
    V_grid_current = A_current @ np.array([ Cs[j]@random.normal(size=n_grid**2) + A_invV_current[j]@Rinvsrs[j] for j in range(p)])
    
    
    ###
    
    V_run[i] = V_current
    taus_run[i] = taus_current
    V_grid_run[i] = V_grid_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    
    if i % 100 == 0:
        print(i)

et = time.time()

print("TTIME:", (et-st)/60, "min")

print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))
print('accept phi_3:',np.mean(acc_phis[2,tail:]))
print('accept phi_4:',np.mean(acc_phis[3,tail:]))
print('accept phi_5:',np.mean(acc_phis[4,tail:]))

plt.plot(phis_run[tail:,0])
plt.plot(phis_run[tail:,1])
plt.plot(phis_run[tail:,2])
plt.plot(phis_run[tail:,3])
plt.plot(phis_run[tail:,4])
plt.show()

plt.plot(taus_run[tail:,0])
plt.plot(taus_run[tail:,1])
plt.plot(taus_run[tail:,2])
plt.plot(taus_run[tail:,3])
plt.plot(taus_run[tail:,4])
plt.show()


print("MSE", np.mean([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)]))


### evaluate independencies

covariances = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)])

print("POSTERIOR MARGINAL COVARIANCE MEAN")
print(np.around(np.mean(covariances,axis=0), 2))

print("TRUE MARGINAL COVARIANCE")
print(np.around(A@np.transpose(A), 2))

indep_post = np.mean(covariances==0,axis=0)
print("POSTERIOR PROBABILITY OF MARGINAL INDEPENDENCE")
print(np.around(indep_post, 2))

### showcase mean predictions

V_grid_mean = np.mean(V_grid_run,axis=0)

### process 1

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid_mean[0],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Blues")
plt.show()

### process 2

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid_mean[1],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Oranges")
plt.show()

### process 3

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid_mean[2],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Greens")
plt.show()

### process 4

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid_mean[3],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Reds")
plt.show()

### process 5

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = vec_inv(V_grid_mean[4],n_grid)
ax.pcolormesh(xv, yv, vv, cmap = "Purples")
plt.show()

### showcase infered crosscovariance

max_d = 1
res = 100

ds = np.linspace(0,max_d,res)

def crossCov(d,A,phis,i,j):
    return(np.sum(A[i] * A[j] * np.exp(-d*phis)))
    
cc = np.zeros(res)
cc_obs = np.zeros((N-tail,res))

fig, axs = plt.subplots(5, 2, figsize=(8, 12))

i=0
j=1

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[0, 0].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[0, 0].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[0, 0].plot(ds,cc)
axs[0, 0].set_title(str(i+1) + " and " + str(j+1))
    

i=0
j=2

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[0, 1].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[0, 1].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[0, 1].plot(ds,cc)
axs[0, 1].set_title(str(i+1) + " and " + str(j+1))

i=0
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[1, 0].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[1, 0].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[1, 0].plot(ds,cc)
axs[1, 0].set_title(str(i+1) + " and " + str(j+1))
    

i=0
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[1, 1].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[1, 1].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[1, 1].plot(ds,cc)
axs[1, 1].set_title(str(i+1) + " and " + str(j+1))

i=1
j=2

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[2, 0].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[2, 0].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[2, 0].plot(ds,cc)
axs[2, 0].set_title(str(i+1) + " and " + str(j+1))

i=1
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[2, 1].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[2, 1].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[2, 1].plot(ds,cc)
axs[2, 1].set_title(str(i+1) + " and " + str(j+1))

i=1
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[3, 0].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[3, 0].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[3, 0].plot(ds,cc)
axs[3, 0].set_title(str(i+1) + " and " + str(j+1))

i=2
j=3

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[3, 1].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[3, 1].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[3, 1].plot(ds,cc)
axs[3, 1].set_title(str(i+1) + " and " + str(j+1))


i=2
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[4, 0].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[4, 0].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[4, 0].plot(ds,cc)
axs[4, 0].set_title(str(i+1) + " and " + str(j+1))



i=3
j=4

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
for ns in range(tail,N):
    for r in range(res):
        cc_obs[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
        
axs[4, 1].fill_between(ds, np.quantile(cc_obs,0.05,axis=0), np.quantile(cc_obs,0.95,axis=0), color="silver")    
axs[4, 1].plot(ds,np.median(cc_obs,axis=0), c="black")
axs[4, 1].plot(ds,cc)
axs[4, 1].set_title(str(i+1) + " and " + str(j+1))


# plt.savefig('crosscov.pdf') 
plt.tight_layout()
plt.show()