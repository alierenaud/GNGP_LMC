# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:36:37 2023

@author: alier
"""

import numpy as np
from numpy import random

from multiLMC_generation import rmultiLMC
from multiLMC_generation import mult
from LMC_multi import mult_vec

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from noisyLMC_interweaved import A_move_slice
from noisyLMC_interweaved import A_move_white
from LMC_inference import phis_move
from LMC_mean import mu_move
from noisyLMC_inference import V_move_conj
from LMC_multi import Z_move
from LMC_multi import probs
from LMC_pred_rjmcmc import V_pred
from noisyLMC_interweaved import vec_inv
from LMC_generation import rLMC


from noisyLMC_interweaved import makeGrid


tab_cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    

def probspp(V_true,V_current,bigN,locs,loc_obs,Y):
    
    order = np.argsort(locs[:,0])
    
    p = V_true.shape[0]
    n = V_true.shape[1]
    
    Z_exs_infV = random.normal(size=(bigN,p,n)) + V_current
    Z_exs_trueV = random.normal(size=(bigN,p,n)) + V_true
    
    
    
    
    Y_exs_infV = np.zeros((bigN,n),dtype=int)
    Y_exs_trueV = np.zeros((bigN,n),dtype=int)
    
    for ns in range(bigN):
        for i in range(n):
            Y_exs_infV[ns,i] = mult(Z_exs_infV[ns,:,i])
            Y_exs_trueV[ns,i] = mult(Z_exs_trueV[ns,:,i])
    
    
    
    Y_vec_exs_infV=np.array([mult_vec(Y_exs_infV[ns],p) for ns in range(bigN)])
    Y_vec_exs_trueV=np.array([mult_vec(Y_exs_trueV[ns],p) for ns in range(bigN)])
    
    probs_infV = np.mean(Y_vec_exs_infV,axis=0)
    probs_trueV = np.mean(Y_vec_exs_trueV,axis=0)
    

    for k in range(p):
    
        plt.plot(locs[order,0],probs_infV[k,order], c=tab_cols[k])
        plt.plot(locs[order,0],probs_trueV[k,order], c=tab_cols[k], alpha=0.5)
        
        plt.scatter(loc_obs[Y==k+1,0],np.zeros(np.sum(Y==k+1))-(k+1)*0.05,s=1)
        
        
    plt.show()

def fct(x,alpha=0.3):
    return(np.exp(-(x[0]**2+x[1]**2)/alpha))

random.seed(1000)

### global parameters
lam = 1000
n = random.poisson(lam)
# n = 500
# p = 1
p = 2


### generate random example
locs = random.uniform(0,1,(n,2))

# n_grid = 500
# locs_grid = np.transpose(np.array([np.linspace(0,1,n_grid)]))
# Dists_grid = distance_matrix(locs_grid,locs_grid)


# locs_tot = np.concatenate((locs,locs_grid))

# obs = np.zeros(n,int)

# for i in range(n):
#     obs[i] = random.binomial(1,fct(locs[i]))




# mu = np.array([-1.,-1.])
# A = np.array([[1.,1.],
#               [1.,0.]])
# phis = np.array([5.,25.])

# mu = np.array([0.])
# A = np.array([[1.]])
# phis = np.array([5.])


# Y, Z_true, V_true = rmultiLMC(A,phis,mu,locs, retZV=True) 

# Y_grid = Y[n:]
# Z_grid = Z_true[:,n:]
# V_grid = V_true[:,n:]

# Y = Y[:n]
# Z_true = Z_true[:,:n]
# V_true = V_true[:,:n]


## grid

n_grid = 56
loc_grid = makeGrid(n_grid)


Dists_grid = distance_matrix(loc_grid,loc_grid)


# ## to add easily noticeable patern ###
# obs = np.zeros(n,int)

# for i in range(n):
#     obs[i] = random.binomial(1,fct(locs[i]))
    
# Y = Y*obs

# n_0_true = np.sum(Y==0)
# n_1 = np.sum(Y!=0)

# Y_0_true = Y[Y==0]
# Y_1 = Y[Y!=0]

# X_0_true = locs[Y==0]
# X_1 = locs[Y!=0]


# Z_0_true = Z_true[:,Y==0]
# Z_1_true = Z_true[:,Y!=0]

# V_0_true = V_true[:,Y==0]
# V_1_true = V_true[:,Y!=0]


# V_true = np.concatenate((V_1_true,V_0_true),axis=1)

#### maple hickory examples

maple = np.loadtxt("maple.csv",delimiter=",")
hickory = np.loadtxt("hickory.csv",delimiter=",")

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]

Y1 = np.ones(n_maple,dtype=int)
Y2 = np.ones(n_hickory,dtype=int)*2

Y_1 = np.concatenate((Y1,Y2))
X_1 = np.concatenate((maple,hickory))
n_1 = X_1.shape[0]

### showcase 2D

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(X_1[Y_1==1,0],X_1[Y_1==1,1])
ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1])
plt.show()

### showcase 1D

# fig, ax = plt.subplots()

# # order = np.argsort(locs[:,0])

# ax.plot(locs[:,0],V_true[0,:],"o")
# ax.plot(locs[:,0],V_true[1,:],"o")
# plt.show()

### priors
sigma_A = 1.
mu_A = np.zeros((p,p))

sigma_mu = 1.
mu_mu = np.zeros((p))


min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.ones(p)
betas = np.ones(p)

# a = 5
# b = 0.1

a_lam = 1000
b_lam = 1


### useful quantities 

# X_0_current = X_0_true
X_0_current = random.uniform(size=(int(n_1/p),2))
n_0_current = X_0_current.shape[0]
Y_0_current = np.zeros(n_0_current,dtype=int)

D_0_current = distance_matrix(X_0_current,X_0_current)
D_01_current = distance_matrix(X_0_current,X_1)
D_1 = distance_matrix(X_1,X_1)

D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

### init and current state

X_current = np.concatenate((X_1,X_0_current),axis=0)
n_current = n_1 + n_0_current
Y_current = np.concatenate((Y_1,Y_0_current),axis=0)



Z_0_current = random.normal(size=(p,n_0_current))
Z_1_current = random.normal(size=(p,n_1))
# Z_0_current = (mult_vec(Y_0_current,p) - 0.5)
# Z_1_current = (mult_vec(Y_1,p) - 0.5)
# Z_0_current = Z_0_true
# Z_1_current = Z_1_true

V_0_current = random.normal(size=(p,n_0_current))
V_1_current = random.normal(size=(p,n_1))
# V_0_current = (mult_vec(Y_0_current,p) - 0.5)*2
# V_1_current = (mult_vec(Y_1,p) - 0.5)*2


# V_0_current = V_0_true
# V_1_current = V_1_true

# Z_current = Z_true
# Z_current = (mult_vec(Y,p) - 0.5)*4
# Z_current = (mult_vec(Y_current,p) - 0.5)*2
Z_current = np.concatenate((Z_1_current,Z_0_current),axis=1)

# V_current = V_true
# V_current = (mult_vec(Y,p) - 0.5)*2
# V_current = random.normal(size=(p,n_current))
V_current = np.concatenate((V_1_current,V_0_current),axis=1)
VmZ_current = V_current - Z_current
VmZ_inner_rows_current = np.array([ np.inner(VmZ_current[j], VmZ_current[j]) for j in range(p) ])








# mu_current = mu
mu_current = np.zeros(p) 
Vmmu_current = V_current - np.outer(mu_current,np.ones(n_current))

# A_current = A
A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)
A_invVmmu_current = A_inv_current @ Vmmu_current

# phis_current = phis
phis_current = np.ones(p)*10.
Rs_current = np.array([ np.exp(-D_current*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])




taus = np.ones(p)
Dm1_current = np.diag(taus)
# Dm1Z_current = Dm1_current @ Z_current

lam_current = lam


# ### proposals


phis_prop = np.ones(p)*1
sigma_slice = 10




### samples

N = 6000
tail = 2000

### global run containers
mu_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
phis_run = np.zeros((N,p))
lam_run = np.zeros((N))
n_0_run = np.zeros(N)
V_grid_run = np.zeros((N,p,n_grid**2))



### acc vector

acc_phis = np.zeros((p,N))





import time
st = time.time()


for i in range(N):
    
    
    ### update X_0,Z_0,V_0
    
    
    Z_1_current = Z_current[:,:n_1]
    V_1_current = V_current[:,:n_1]
    
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
    D_01_current = distance_matrix(X_0_current,X_1)
    
    D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])
    
    X_current = np.concatenate((X_1,X_0_current),axis=0)
    n_current = n_1 + n_0_current
    Y_current = np.concatenate((Y_1,Y_0_current),axis=0)
    
    Z_current = np.concatenate((Z_1_current,Z_0_current),axis=1)
    V_current = np.concatenate((V_1_current,V_0_current),axis=1)
    
    VmZ_current = V_current - Z_current
    VmZ_inner_rows_current = np.array([ np.inner(VmZ_current[j], VmZ_current[j]) for j in range(p) ])

    Vmmu_current = V_current - np.outer(mu_current,np.ones(n_current))
    
    A_invVmmu_current = A_inv_current @ Vmmu_current
    
    Rs_current = np.array([ np.exp(-D_current*phis_current[j]) for j in range(p) ])
    
    ### could be faster (possibly)
    Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ]) 
    
    
    V_current, Vmmu_current, VmZ_current, VmZ_inner_rows_current, A_invVmmu_current = V_move_conj(Rs_inv_current, A_inv_current, taus, Z_current, Z_current, V_current, Vmmu_current, mu_current)
        
    
    
    
    mu_current, Vmmu_current, A_invVmmu_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    A_current, A_inv_current, A_invVmmu_current = A_move_slice(A_current, A_invVmmu_current, Rs_inv_current, Vmmu_current, sigma_A, mu_A, sigma_slice)
    
    # A_current, A_inv_current, V_current = A_move_white(A_invVmmu_current,Dm1_current,Z_current,sigma_A,mu_A)
    

    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,D_current,A_invVmmu_current,Rs_current,Rs_inv_current)
    
    
    Z_current,VmZ_current,VmZ_inner_rows_current = Z_move(V_current,Z_current,Y_current)
    
    lam_current = random.gamma(n_current + a_lam, 1/(b_lam + 1))
    
    Dists_obs_grid = distance_matrix(X_current,loc_grid)
    
    V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invVmmu_current, mu_current,n_grid**2)
        
    

    mu_run[i] = mu_current
    A_run[i] = A_current
    phis_run[i] =  phis_current
    n_0_run[i] = n_0_current
    lam_run[i] = lam_current
    V_grid_run[i] = V_grid_current
    
    
    
    if i % 1 == 0:
        print(i)
        
        
        
        
        ## showcase RFs

        # plt.plot(locs_grid[:,0],V_grid_current[0],c="tab:blue")
        # plt.plot(locs_grid[:,0],V_grid_current[1],c="tab:orange")
        # plt.plot(locs_grid[:,0],V_grid[0],c="tab:blue", alpha=0.5)
        # plt.plot(locs_grid[:,0],V_grid[1],c="tab:orange", alpha=0.5)
        # plt.show()
        
        # diagnostic using probabilities
        
        # probspp(V_grid,V_grid_current,1000,locs_grid,X_1,Y_1)
        
        
        fig, ax = plt.subplots()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_box_aspect(1)

        ax.scatter(X_1[Y_1==1,0],X_1[Y_1==1,1])
        ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1])
        ax.scatter(X_0_current[:,0],X_0_current[:,1],c="grey")
        plt.show()
        
        # fig, ax = plt.subplots()

        

        
        # ax.plot(X_current[:,0],V_current[0,:],"o",c="tab:blue")
        # ax.plot(X_current[:,0],V_current[1,:],"o",c="tab:orange")
        # ax.plot(locs[:,0],V_true[0,:],"o",alpha=0.5,c="blue")
        # ax.plot(locs[:,0],V_true[1,:],"o",alpha=0.5,c="orange")
        # plt.show()
        
        
        
        
        

et = time.time()
print('Execution time:', (et-st)/60, 'minutes')



print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))


plt.plot(phis_run[tail:,0])
plt.show()
plt.plot(phis_run[tail:,1])
plt.show()


plt.plot(mu_run[tail:,0])
plt.show()
plt.plot(mu_run[tail:,1])
plt.show()


plt.plot(A_run[tail:,0,0])
plt.show()
plt.plot(A_run[tail:,0,1])
plt.show()
plt.plot(A_run[tail:,1,0])
plt.show()
plt.plot(A_run[tail:,1,1])
plt.show()

plt.plot(lam_run[tail:])
plt.show()


plt.plot(n_0_run[tail:])
plt.show()




print("Posterior Marginal Variance", np.mean([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)],axis=0))
# print("True Marginal Variance", A@np.transpose(A))



V_grid_mean = np.mean(V_grid_run[tail:],axis=0)

locs1D = (np.arange(n_grid) + 0.5)/n_grid
xv, yv = np.meshgrid(locs1D, locs1D)

### process 1

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = np.transpose(vec_inv(V_grid_mean[0],n_grid))
ax.pcolormesh(xv, yv, vv, cmap = "Blues")
ax.scatter(X_1[Y_1==1,0],X_1[Y_1==1,1],c="black")
plt.show()

### process 2

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = np.transpose(vec_inv(V_grid_mean[1],n_grid))
ax.pcolormesh(xv, yv, vv, cmap = "Oranges")
ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1],c="black")
plt.show()

### computing intensity


Z_grid_run = V_grid_run + random.normal(size=(N,p,n_grid**2))


def intensis(x):
    
    p = x.shape[0]
    
    iis = np.zeros(p+1)
    
    if np.prod(x<0):
        iis[0] = 1
    else:
        iis[np.argmax(x)+1] = 1
        

    return(iis)


intensity_run = np.zeros((N,p+1,n_grid**2))


for ii in range(N):
    for jj in range(n_grid**2):
        intensity_run[ii,:,jj] = lam_run[ii] * intensis(Z_grid_run[ii,:,jj])




intensity_mean = np.mean(intensity_run[tail:],axis=0)

locs1D = (np.arange(n_grid) + 0.5)/n_grid
xv, yv = np.meshgrid(locs1D, locs1D)

### process 1

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = np.transpose(vec_inv(intensity_mean[1],n_grid))
c = ax.pcolormesh(xv, yv, vv, cmap = "Blues")
ax.scatter(X_1[Y_1==1,0],X_1[Y_1==1,1],c="black")
plt.colorbar(c)
plt.show()

### process 2

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = np.transpose(vec_inv(intensity_mean[2],n_grid))
c = ax.pcolormesh(xv, yv, vv, cmap = "Oranges")
ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1],c="black")
plt.colorbar(c)
plt.show()

### process thinned

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)


vv = np.transpose(vec_inv(intensity_mean[0],n_grid))
c = ax.pcolormesh(xv, yv, vv, cmap = "Greys")
# ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1],c="black")
plt.colorbar(c)
plt.show()


np.save("A_run.npy",A_run)
np.save("phis_run.npy",phis_run)
np.save("mu_run.npy",mu_run)
np.save("lam_run.npy",lam_run)
np.save("n_0_run.npy",n_0_run)
np.save("V_grid_run.npy",V_grid_run)

