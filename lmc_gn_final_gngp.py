#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:41:36 2024

@author: homeboy
"""

from numpy import random
import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

import time

from base import makeGrid, vec_inv

from multiLMC_generation import rmultiLMC


from gn_func import phis_move, A_move_slice, mu_move, V_move_conj_scale, V_grid_move_scale
from LMC_multi import Z_move
from base import matern_kernel

def ell(s,n_grid,m):
    
    if s*n_grid < (m+1)/2:
        return(0)
    elif s*n_grid > n_grid - (m+1)/2:
        return(n_grid-m)
    else:
        return(np.ceil(s*n_grid)- (m+1)/2)


def X_move(loc_grid,n_grid,lam_current,phis_current,mu_current,A_current,A_inv_current,A_invV_gridmmu1_current,dist_nei_ogrid,X_current,V_current,Z_current,Y_current,X_obs,Y_obs,n_obs):
    
    p = V_current.shape[0]
    m = int(np.sqrt(dist_nei_ogrid.shape[0])-1)
    
    
    V_obs_current = V_current[:,:n_obs]
    Z_obs_current = Z_current[:,:n_obs]
    
    
    
    n_new = random.poisson(lam_current)
    X_new = random.uniform(0,1,(n_new,2))
    

    ogNei_new = np.zeros((n_new,(m+1)**2),dtype=int)
    dist_pnei_ogrid_new = np.zeros(((n_new,(m+1)**2)))

        

    for i in range(n_new):
            
        left_lim = ell(X_new[i,0],n_grid,m)
        xNei = np.arange(left_lim,left_lim+m+1) 
        
        down_lim = ell(X_new[i,1],n_grid,m)
        yNei = np.arange(down_lim,down_lim+m+1) 

        
        ogNei_new[i] = np.array([ii*(n_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
        
        dist_pnei_ogrid_new[i] = distance_matrix([X_new[i]],loc_grid[ogNei_new[i]])[0]

        

    ogbs_new = np.zeros((p,n_new,(m+1)**2))
    ogrs_new = np.zeros((p,n_new))


    for j in range(p):
        
        R_j_N_inv = np.linalg.inv(matern_kernel(dist_nei_ogrid,phis_current[j]))
        
        
        for i in range(n_new):
        
            r_j_Nii = matern_kernel(dist_pnei_ogrid_new[i],phis_current[j])
        
            ogb = R_j_N_inv@r_j_Nii
            
            ogbs_new[j,i] = ogb
            ogrs_new[j,i] = 1 - np.inner(r_j_Nii,ogb)
    
    outsies = np.array([np.outer(A_current[:,j],A_current[:,j]) for j in range(p)])
    V_new = np.zeros((p,n_new))
    
    
    for i in range(n_new):
        
        
        Sigma_new = np.sum([outsies[j]*ogrs_new[j,i] for j in range(p)],axis=0)
        mu_new = mu_current + np.sum([np.inner(A_invV_gridmmu1_current[j,ogNei_new[i]],ogbs_new[j,i])*A_current[:,j]  for j in range(p)],axis=0)
        
        V_new[:,i] = np.linalg.cholesky(Sigma_new)@random.normal(size=p) + mu_new
    
    
    Z_new = V_new + random.normal(size=(p,n_new))
    
    Y_ind = np.prod(Z_new<0,axis=0)
    
    X_0_current = X_new[Y_ind==1]
    V_0_current = V_new[:,Y_ind==1]
    Z_0_current = Z_new[:,Y_ind==1]
    n_0_current = np.sum(Y_ind==1)
    Y_0_current = np.zeros(n_0_current,dtype=int)
    

    X_current = np.concatenate((X_obs,X_0_current),axis=0)
    n_current = n_obs + n_0_current
    Y_current = np.concatenate((Y_obs,Y_0_current),axis=0)
    
    
    Z_current = np.concatenate((Z_obs_current,Z_0_current),axis=1)
    V_current = np.concatenate((V_obs_current,V_0_current),axis=1)
    
    Vmmu1_current = V_current - np.outer(mu_current,np.ones(n_current))
    
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    ### neighbors observations
       
    ogNei = np.zeros((n_current,(m+1)**2),dtype=int)
       
    aogNei = np.zeros((n_grid+1)**2,dtype=object)
    aogInd = np.zeros((n_grid+1)**2,dtype=object)
    
    dist_pnei_ogrid = np.zeros(((n_current,(m+1)**2)))
       
    for i in range((n_grid+1)**2):
        aogNei[i] = []
        aogInd[i] = []
    
    ### neihgbors obs-grid   
    

    

    for i in range(n_current):
            
        left_lim = ell(X_current[i,0],n_grid,m)
        xNei = np.arange(left_lim,left_lim+m+1) 
        
        down_lim = ell(X_current[i,1],n_grid,m)
        yNei = np.arange(down_lim,down_lim+m+1) 
    
        
        ogNei[i] = np.array([ii*(n_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
        
        ind = 0 
        for j in ogNei[i]:
            aogNei[j].append(i)
            aogInd[j].append(ind)
            ind += 1
        
        dist_pnei_ogrid[i] = distance_matrix([X_current[i]],loc_grid[ogNei[i]])[0]


    
    
    




    ogbs = np.zeros((p,n_current,(m+1)**2))
    ogrs = np.zeros((p,n_current))
    
    
    for j in range(p):
        
        R_j_N_inv = np.linalg.inv(matern_kernel(dist_nei_ogrid,phis_current[j]))
        
        
        for i in range(n_current):
        
            r_j_Nii = matern_kernel(dist_pnei_ogrid[i],phis_current[j])
        
            ogb = R_j_N_inv@r_j_Nii
            
            ogbs[j,i] = ogb
            ogrs[j,i] = 1 - np.inner(r_j_Nii,ogb)
        

    return(n_current,X_current,V_current,Z_current,Y_current,ogNei,aogNei,aogInd,dist_pnei_ogrid,ogbs,ogrs,Vmmu1_current,A_invVmmu1_current)


random.seed(0)
cols = ["Blues","Oranges","Greens","Reds","Purples"]
tab_cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    

### import lansing woods data

# maple = np.loadtxt("maple.csv", delimiter=",")
# hickory = np.loadtxt("hickory.csv", delimiter=",")
# whiteoak = np.loadtxt("whiteoak.csv", delimiter=",")
# redoak = np.loadtxt("redoak.csv", delimiter=",")
# blackoak = np.loadtxt("blackoak.csv", delimiter=",")

# n_maple = maple.shape[0]
# n_hickory = hickory.shape[0]
# n_whiteoak = whiteoak.shape[0]
# n_redoak = redoak.shape[0]
# n_blackoak = blackoak.shape[0]

# X_obs = np.concatenate((maple,hickory,whiteoak,redoak,blackoak))

# n_obs = n_maple + n_hickory + n_whiteoak + n_redoak + n_blackoak
# Y_obs = np.concatenate((np.ones(n_maple,dtype=int)*1,np.ones(n_hickory,dtype=int)*2,np.ones(n_whiteoak,dtype=int)*3,np.ones(n_redoak,dtype=int)*4,np.ones(n_blackoak,dtype=int)*5))

# X_obs += random.uniform(size=(n_obs,2))/10**3

# p = 5

# n_grid=50

# a_lam = n_obs*(p+1)/p

### base intensity
lam = 2500
n_grid=50
# n_grid=int(np.sqrt(lam/4)-1)

### number of dimensions
p = 5

### number of neighbors
m = 3

### markov chain + tail length
N = 4000
tail = 0


### generate base poisson process

n_true = random.poisson(lam)
X_true = random.uniform(size=(n_true,2))

### grid locations
marg_grid = np.linspace(0,1,n_grid+1)
loc_grid = makeGrid(marg_grid, marg_grid)
### all locations
locs = np.concatenate((X_true,loc_grid), axis=0)

### showcase locations

# fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_box_aspect(1)

# ax.scatter(X_true[:,0],X_true[:,1],color="black")
# plt.show()


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
fac = np.ones((p,p))
for i in range(p):
    for j in range(i+1,p):
        fac[i,j] = -1 
A *= fac

### amplify signal 

# A *= 2

### lower correlation

# A += np.identity(p)*1

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

Rho = np.diag(np.diag(Sigma)**(-1/2))@Sigma@np.diag(np.diag(Sigma)**(-1/2))
Rho_0p1 = np.diag(np.diag(Sigma)**(-1/2))@Sigma_0p1@np.diag(np.diag(Sigma)**(-1/2))


### random example

Y, Z_true_all, V_true_all = rmultiLMC(A,phis,mu,locs, retZV=True) 

Y_true = Y[:n_true]
Y_grid = Y[n_true:]

Z_true = Z_true_all[:,:n_true]


V_true = V_true_all[:,:n_true]
V_grid = V_true_all[:,n_true:]

### add noticeable patern

# Y_true[X_true[:,1]>0.5] = 0

### move zeros to tail 

X_true = np.concatenate((X_true[Y_true!=0],X_true[Y_true==0]))
V_true = np.concatenate((V_true[:,Y_true!=0],V_true[:,Y_true==0]),axis=1)
Z_true = np.concatenate((Z_true[:,Y_true!=0],Z_true[:,Y_true==0]),axis=1)
Y_true = np.concatenate((Y_true[Y_true!=0],Y_true[Y_true==0]))


### fixed quantities 

n_obs = np.sum(Y_true!=0)
Y_obs = Y_true[:n_obs]
X_obs = X_true[:n_obs]

### illustrate multi process grid

# fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_box_aspect(1)

# ax.scatter(X_true[Y_true==0,0],X_true[Y_true==0,1],color="grey")

# for i in range(p):
    
    
#     ax.scatter(X_true[Y_true==i+1,0],X_true[Y_true==i+1,1],color=tab_cols[i])
    
# plt.show()


# xv, yv = np.meshgrid(marg_grid, marg_grid)


# for i in range(p):

    
#     fig, ax = plt.subplots()
#     # ax.set_xlim(0,1)
#     # ax.set_ylim(0,1)
#     ax.set_box_aspect(1)
    
    
    
#     c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(V_grid[i],n_grid+1)), cmap = cols[i])
#     plt.colorbar(c)
#     plt.show()



# fig, ax = plt.subplots()
# # ax.set_xlim(0,1)
# # ax.set_ylim(0,1)
# ax.set_box_aspect(1)



# c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(Y_grid,n_grid+1)), cmap = "Greys")
# plt.colorbar(c)
# plt.show()


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



### mu 

mu_mu = np.zeros(p)
sigma_mu = 1

### lambda

a_lam = np.sum(Y_true!=0)*(p+1)/p
b_lam = 1


### proposals


phis_prop = np.ones(p)*0.5
sigma_slice = 1



### global run containers
n_run = np.zeros(N)
lam_run = np.zeros(N)
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,(n_grid+1)**2))


### acc vector
acc_phis = np.zeros((p,N))


### grid neighbors

gNei = np.zeros((n_grid+1)**2,dtype=object)


### anti-neighbors

agNei = np.zeros((n_grid+1)**2,dtype=object)
agInd = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    agNei[i] = []
    agInd[i] = []
    
    
### neighbors grid

for j in range(n_grid+1):
    for i in range(n_grid+1):
        
        xNei = np.arange(np.max([0,i-m]),i+1)
        yNei = np.arange(np.max([0,j-m]),j+1)
    
        gNei[j*(n_grid+1)+i] = np.array([jj*(n_grid+1)+ii for jj in yNei for ii in xNei if (ii != i) | (jj != j )],dtype=int)

        ind = 0 
        for jj in gNei[j*(n_grid+1)+i]:
            agNei[jj].append(j*(n_grid+1)+i)
            agInd[jj].append(ind)
            ind += 1


### distances

npat = (m+1)**2

dist_nei_grid = np.zeros(npat,dtype=object)
dist_pnei_grid = np.zeros(npat,dtype=object)



init = 0

for j in range(m+1):
    for i in range(m+1):
        
        ind = j*(n_grid+1) + i
        
        # print(ind)
        
        dist_nei_grid[init] = distance_matrix(loc_grid[gNei[ind]],loc_grid[gNei[ind]])
        dist_pnei_grid[init] = distance_matrix([loc_grid[ind]],loc_grid[gNei[ind]])[0]
        
        # print(dist_nei_grid[init]@dist_pnei_grid[init])
        
        init += 1
        
### init 

# True #

# n_current = np.copy(n_true)
# X_current = np.copy(X_true)
# Y_current = np.copy(Y_true)
# V_current = np.copy(V_true)
# Z_current = np.copy(Z_true)
# phis_current = np.copy(phis)
# mu_current = np.copy(mu)
# A_current = np.copy(A)
# lam_current = np.copy(lam)
# V_grid_current = np.copy(V_grid)


# Arbitrary #

n_0_init = 0
X_0_init = random.uniform(size=(n_0_init,2))
Y_0_init = np.zeros(n_0_init)

n_current = n_obs + n_0_init
X_current = np.concatenate((X_obs,X_0_init))
Y_current = np.concatenate((Y_obs,Y_0_init))

V_current = np.zeros(shape=(p,n_current))
Z_current = np.zeros(shape=(p,n_current))

V_grid_current = np.zeros(shape=(p,(n_grid+1)**2))

phis_current = np.repeat(10.,p)
mu_current = np.zeros(p)
A_current = np.identity(p)
# A_current = random.normal(size=(p,p))
lam_current = a_lam



### updated quantities

### neighbors observations

ogNei = np.zeros((n_current,(m+1)**2),dtype=int)

aogNei = np.zeros((n_grid+1)**2,dtype=object)
aogInd = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    aogNei[i] = []
    aogInd[i] = []
    
### neihgbors obs-grid


    

for i in range(n_current):
        
    left_lim = ell(X_current[i,0],n_grid,m)
    xNei = np.arange(left_lim,left_lim+m+1) 
    
    down_lim = ell(X_current[i,1],n_grid,m)
    yNei = np.arange(down_lim,down_lim+m+1) 

    
    ogNei[i] = np.array([ii*(n_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
    
    ind = 0 
    for j in ogNei[i]:
        aogNei[j].append(i)
        aogInd[j].append(ind)
        ind += 1
    

### does not get updated
dist_nei_ogrid = distance_matrix(loc_grid[ogNei[0]],loc_grid[ogNei[0]])

dist_pnei_ogrid = np.zeros(((n_current,(m+1)**2)))


for i in range(n_current):
    
    dist_pnei_ogrid[i] = distance_matrix([X_current[i]],loc_grid[ogNei[i]])[0]



### likelihood quantities

gbs = np.zeros((p,npat),dtype=object)
grs = np.zeros((p,npat))

for i in range(npat):
    for j in range(p):
        
        R_j_Ni_inv = np.linalg.inv(matern_kernel(dist_nei_grid[i],phis_current[j]))
        r_j_Nii = matern_kernel(dist_pnei_grid[i],phis_current[j])
        
        gb = R_j_Ni_inv@r_j_Nii
        
        gbs[j,i] = gb
        grs[j,i] = 1 - np.inner(r_j_Nii,gb)
    

ogbs = np.zeros((p,n_current,(m+1)**2))
ogrs = np.zeros((p,n_current))


for j in range(p):
    
    R_j_N_inv = np.linalg.inv(matern_kernel(dist_nei_ogrid,phis_current[j]))
    
    
    for i in range(n_current):
    
        r_j_Nii = matern_kernel(dist_pnei_ogrid[i],phis_current[j])
    
        ogb = R_j_N_inv@r_j_Nii
        
        ogbs[j,i] = ogb
        ogrs[j,i] = 1 - np.inner(r_j_Nii,ogb)
        







        
        
### unchanged quantities from exact




Vmmu1_current = V_current-np.outer(mu_current,np.ones(n_current))
V_gridmmu1_current = V_grid_current-np.outer(mu_current,np.ones((n_grid+1)**2))


A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current
A_invV_gridmmu1_current = A_inv_current @ V_gridmmu1_current




st = time.time()

for i in range(N):
    
    
    
    n_current,X_current,V_current,Z_current,Y_current,ogNei,aogNei,aogInd,dist_pnei_ogrid,ogbs,ogrs,Vmmu1_current,A_invVmmu1_current = X_move(loc_grid,n_grid,lam_current,phis_current,mu_current,A_current,A_inv_current,A_invV_gridmmu1_current,dist_nei_ogrid,X_current,V_current,Z_current,Y_current,X_obs,Y_obs,n_obs)
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj_scale(ogbs, ogrs, ogNei, A_inv_current, taus, Dm1, Z_current, Z_current, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current)
    
    
    
    V_grid_current, V_gridmmu1_current, A_invV_gridmmu1_current = V_grid_move_scale(gbs, ogbs, grs, ogrs, gNei, ogNei, agNei, agInd, aogNei, aogInd, A_inv_current, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current)
    
    mu_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current = mu_move(A_inv_current,gNei,ogNei,gbs,grs,ogbs,ogrs,V_current,V_grid_current,sigma_mu,mu_mu)

    
    A_current,A_inv_current,A_invVmmu1_current,A_invV_gridmmu1_current = A_move_slice(A_current, A_invVmmu1_current, A_invV_gridmmu1_current, Vmmu1_current, V_gridmmu1_current, gNei, ogNei, gbs, grs, ogbs, ogrs, sigma_A, mu_A, sigma_slice)

    

    phis_current,gbs,grs,ogbs,ogrs,acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,A_invVmmu1_current,A_invV_gridmmu1_current,gNei,ogNei,dist_nei_grid,dist_pnei_grid,dist_nei_ogrid,dist_pnei_ogrid,gbs,grs,ogbs,ogrs)
    
    Z_current = Z_move(V_current,Z_current,Y_current)

    lam_current = random.gamma(n_current + a_lam, 1/(b_lam + 1))      

    lam_run[i] = lam_current
    n_run[i] = n_current
    mu_run[i] = mu_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    V_grid_run[i] = V_grid_current 
    
    if i % 10 == 0:
        
        
        ett = time.time()
        est_time = (ett-st)*(N-i-1)/(i+1)
        est_h = int(est_time//(60**2))
        est_m = int(est_time%(60**2)//60)
        print(i, "Est. Time Remaining:", est_h, "h", est_m, "min")
        print(np.linalg.det(A_current))
        
        fig, ax = plt.subplots()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_box_aspect(1)
        ax.scatter(X_current[:n_obs,0],X_current[:n_obs,1],c="black")
        ax.scatter(X_current[n_obs:,0],X_current[n_obs:,1],c="grey")
        plt.show()
    


et = time.time()
ela_time = (et-st)
ela_h = int(ela_time//(60**2))
ela_m = int(ela_time%(60**2)//60)
print(N, "Time Elapsed:", ela_h, "h", ela_m, "min")


print("Accept Rate for phis",np.mean(acc_phis,axis=1))


### trace plots

plt.plot(n_run[tail:])
plt.show()

# print("True n ",n_true)
print("Post Mean n ",np.mean(n_run[tail:]))

plt.plot(lam_run[tail:])
plt.show()

# print("True lambda ",lam)
print("Post Mean lambda ",np.mean(lam_run[tail:]))

for i in range(p):
    plt.plot(mu_run[tail:,i])
plt.show()

# print("True mu ",mu)
print("Post Mean mu ",np.mean(mu_run[tail:],axis=0))



for i in range(p):
    plt.plot(phis_run[tail:,i])
plt.show()


for i in range(p):
    for j in range(p):
        plt.plot(A_run[tail:,i,j])
plt.show()


# covariance

Sigma_run = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(N)])
# print("True Sigma\n",Sigma)
print("Post Mean Sigma\n",np.mean(Sigma_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_run[tail:,i,j])
plt.show()

Sigma_0p1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*0.1))@np.transpose(A_run[i]) for i in range(N)])
# print("True Sigma 0.1\n",Sigma_0p1)
print("Post Mean Sigma 0.1\n",np.mean(Sigma_0p1_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_0p1_run[tail:,i,j])
plt.show()

Sigma_1_run = np.array([A_run[i]@np.diag(np.exp(-phis_run[i]*1))@np.transpose(A_run[i]) for i in range(N)])
# print("True Sigma 1\n",Sigma_1)
print("Post Mean Sigma 1\n",np.mean(Sigma_1_run[tail:],axis=0))

for i in range(p):
    for j in range(i,p):
        plt.plot(Sigma_1_run[tail:,i,j])
plt.show()




### mean processes

V_grid_mean = np.mean(V_grid_run[tail:],axis=0)


for i in range(p):
    
    


    # xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    # fig, ax = plt.subplots()
    # # ax.set_xlim(0,1)
    # # ax.set_ylim(0,1)
    # ax.set_box_aspect(1)
    
    
    
    # c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(V_grid[i],n_grid+1)), cmap = cols[i])
    # plt.colorbar(c)
    # # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    # plt.show()

    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(V_grid_mean[i],n_grid+1)), cmap = cols[i])
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()






# MSE = np.mean((V_grid_run - V_grid)**2)
# print("MSE = ", MSE)


### confidence interval C_12(0) and C_12(0.1)

c0l = np.quantile(Sigma_run[tail:,0,1],0.05)
c0u = np.quantile(Sigma_run[tail:,0,1],0.95)

print("C_12(0) : [", c0l,",",c0u,"]")


c0p1l = np.quantile(Sigma_0p1_run[tail:,0,1],0.05)
c0p1u = np.quantile(Sigma_0p1_run[tail:,0,1],0.95)

print("C_12(0.1) : [", c0p1l,",",c0p1u,"]")


rho_run = np.array([np.diag(np.diag(Sigma_run[i])**(-1/2))@Sigma_run[i]@np.diag(np.diag(Sigma_run[i])**(-1/2)) for i in range(N)])
rho_0p1_run = np.array([np.diag(np.diag(Sigma_run[i])**(-1/2))@Sigma_0p1_run[i]@np.diag(np.diag(Sigma_run[i])**(-1/2)) for i in range(N)])

# print("True Rho\n",Rho)
print("Post Mean Rho\n",np.mean(rho_run[tail:],axis=0))

# print("True Rho 0.1\n",Rho_0p1)
print("Post Mean Rho 0.1\n",np.mean(rho_0p1_run[tail:],axis=0))





### computing intensity


Z_grid_run = V_grid_run + random.normal(size=(N,p,(n_grid+1)**2))


def intensis(x):
    
    p = x.shape[0]
    
    iis = np.zeros(p+1)
    
    if np.prod(x<0):
        iis[0] = 1
    else:
        iis[np.argmax(x)+1] = 1
        

    return(iis)


intensity_run = np.zeros((N,p+1,(n_grid+1)**2))


for ii in range(N):
    for jj in range((n_grid+1)**2):
        intensity_run[ii,:,jj] = lam_run[ii] * intensis(Z_grid_run[ii,:,jj])

intensity_mean = np.mean(intensity_run[tail:],axis=0)

for i in range(p):
    

    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(intensity_mean[i+1],n_grid+1)), cmap = cols[i%5])
    ax.scatter(X_obs[Y_obs==i+1,0],X_obs[Y_obs==i+1,1],c="black")
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()

# np.save("run_lam.npy",lam_run)
# np.save("run_n.npy",n_run)
# np.save("run_mu.npy",mu_run)
# np.save("run_phis.npy",phis_run)
# np.save("run_A.npy",A_run)
# np.save("run_V_grid.npy",V_grid_run)


