# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:26:25 2024

@author: alier
"""

from numpy import random
import numpy as np

from scipy.stats import beta
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from base import makeGrid
from noisyLMC_generation import rNLMC_mu

def phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,A_invV_current,gNei,ogNei,dist_nei_grid,dist_pnei_grid,dist_nei_ogrid,dist_pnei_ogrid,gbs,grs,ogbs,ogrs):
    
    p = phis_current.shape[0]
    range_phi = max_phi - min_phi
    
    acc_phis = np.zeros(p)
    
    gbs_new = np.zeros((p,npat),dtype=object)
    grs_new = np.zeros((p,npat))

    ogbs_new = np.zeros((p,n_obs,(m+1)**2))
    ogrs_new = np.zeros((p,n_obs))
    
    for j in range(p):
        
        phis_new = phis_current[j] + phis_prop[j]*random.normal()
        
        if (phis_new > min_phi)  &  (phis_new < max_phi):
            
            ### prior
            
            
            phis_new_star_j = (phis_new - min_phi)/range_phi
            phis_current_star_j = (phis_current[j] - min_phi)/range_phi
            
            ### grid

            for i in range(npat):
                
                    
                R_j_Ni_inv = np.linalg.inv(np.exp(-dist_nei_grid[i]*phis_new[j]))
                r_j_Nii = np.exp(-dist_pnei_grid[i]*phis_new[j])
                
                gb = R_j_Ni_inv@r_j_Nii
                
                gbs_new[j,i] = gb
                grs_new[j,i] = 1 - np.inner(r_j_Nii,gb)
                

            
            ### obs
            
            R_j_N_inv = np.linalg.inv(np.exp(-dist_nei_ogrid*phis_new[j]))
            
            
            for i in range(n_obs):
            
                r_j_Nii = np.exp(-dist_pnei_ogrid[i]*phis_new[j])
            
                ogb = R_j_N_inv@r_j_Nii
                
                ogbs[j,i] = ogb
                ogrs[j,i] = 1 - np.inner(r_j_Nii,ogb)
                    

            
            
            rat = np.exp( -1/2 * ( A_invV_current[j] @ ( Rs_inv_new - Rs_inv_current[j] ) @ A_invV_current[j] ) ) * np.linalg.det( Rs_inv_new @ Rs_current[j] ) **(1/2) * (phis_new_star_j/phis_current_star_j)**(alphas[j]-1) * ((1-phis_new_star_j)/(1-phis_current_star_j))**(betas[j]-1)                             
            
            
            if random.uniform() < rat:
                phis_current[j] = phis_new
                
                gbs[j] = gbs_new[j]
                grs[j] = grs_new[j]
                ogbs[j] = ogbs_new[j]
                ogrs[j] = ogrs_new[j]
                
                acc_phis[j] = 1
                
    return(phis_current,gbs,grs,ogbs,ogrs,acc_phis)


cols = ["Blues","Oranges","Greens","Reds","Purples"]

# random.seed(0)

### number of points 
n_obs=100
n_grid=11

### number of dimensions
p = 2

### number of neighbors

m = 3

### markov chain + tail length
N = 4000
tail = 2000


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
        fac[i,j] = -1 
A *= fac

# print(A)

lower = 5
upper = 25

phis = np.exp(np.linspace(np.log(lower), np.log(upper),p))
mu = A@np.ones(p)

noise_sd = 0.5
taus_sqrt_inv = np.ones(p)*noise_sd
taus = 1/taus_sqrt_inv**2


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


### proposals


phis_prop = np.ones(p)*1
sigma_slice = 10



### global run containers
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,(n_grid+1)**2))


### acc vector
acc_phis = np.zeros((p,N))

### neighbors

gNei = np.zeros((n_grid+1)**2,dtype=object)
ogNei = np.zeros((n_obs,(m+1)**2),dtype=int)

### neighbors grid

for j in range(n_grid+1):
    for i in range(n_grid+1):
        
        xNei = np.arange(np.max([0,i-m]),i+1)
        yNei = np.arange(np.max([0,j-m]),j+1)
    
        gNei[j*(n_grid+1)+i] = [jj*(n_grid+1)+ii for jj in yNei for ii in xNei if (ii != i) | (jj != j )]
        
### showcase grid neighbors

# for j in range(n_grid+1):
#     for i in range(n_grid+1):
        
#         ind = j*(n_grid+1)+i

#         fig, ax = plt.subplots()
        
#         ax.set_aspect(1)
        
        
#         plt.scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#         plt.scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],c="tab:green")
#         plt.scatter(loc_grid[ind,0],loc_grid[ind,1],c="tab:orange")
        
#         # plt.title(str(i))
        
#         plt.show()

### neihgbors obs-grid

def ell(s,n_grid,m):
    
    if s*n_grid < (m+1)/2:
        return(0)
    elif s*n_grid > n_grid - (m+1)/2:
        return(n_grid-m)
    else:
        return(np.ceil(s*n_grid)- (m+1)/2)
    
    

for i in range(n_obs):
        
    left_lim = ell(loc_obs[i,0],n_grid,m)
    xNei = np.arange(left_lim,left_lim+m+1) 
    
    down_lim = ell(loc_obs[i,1],n_grid,m)
    yNei = np.arange(down_lim,down_lim+m+1) 

    
    ogNei[i] = [ii*(n_grid+1)+jj for ii in yNei for jj in xNei]
    
### showcase obs-grid neighbors

# for i in range(n_obs):

#     fig, ax = plt.subplots()
    
#     ax.set_aspect(1)
    
    
#     plt.scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#     plt.scatter(loc_grid[ogNei[i],0],loc_grid[ogNei[i],1],c="tab:green")
#     plt.scatter(loc_obs[i,0],loc_obs[i,1],c="tab:orange")
    
#     # plt.title(str(i))
    
#     plt.show()


### corresponding single index in 1:npat to vertical and horizontal indices i,j



    

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


dist_nei_ogrid = distance_matrix(loc_grid[ogNei[0]],loc_grid[ogNei[0]])
dist_pnei_ogrid = np.zeros(((n_obs,(m+1)**2)))


for i in range(n_obs):
    
    dist_pnei_ogrid[i] = distance_matrix([loc_obs[i]],loc_grid[ogNei[i]])[0]



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
# V_current = random.normal(size=(p,n_obs))*1
# mu_current = np.zeros(p)
# V_grid_current = random.normal(size=(p,(n_grid+1)**2))*1
# taus_current = np.ones(p)
# A_current = random.normal(size=(p,p))

### likelihood quantities

gbs = np.zeros((p,npat),dtype=object)
grs = np.zeros((p,npat))

for i in range(npat):
    for j in range(p):
        
        R_j_Ni_inv = np.linalg.inv(np.exp(-dist_nei_grid[i]*phis_current[j]))
        r_j_Nii = np.exp(-dist_pnei_grid[i]*phis_current[j])
        
        gb = R_j_Ni_inv@r_j_Nii
        
        gbs[j,i] = gb
        grs[j,i] = 1 - np.inner(r_j_Nii,gb)
    

ogbs = np.zeros((p,n_obs,(m+1)**2))
ogrs = np.zeros((p,n_obs))


for j in range(p):
    
    R_j_N_inv = np.linalg.inv(np.exp(-dist_nei_ogrid*phis_current[j]))
    
    
    for i in range(n_obs):
    
        r_j_Nii = np.exp(-dist_pnei_ogrid[i]*phis_current[j])
    
        ogb = R_j_N_inv@r_j_Nii
        
        ogbs[j,i] = ogb
        ogrs[j,i] = 1 - np.inner(r_j_Nii,ogb)
        




## 1 index correspondance

def kay1c(j,i,m,n_grid):
    
    
    if (i > m) & (j > m):
        return(m*(n_grid+1)+m)
    elif (i > m) & (j <= m):
        return(j*(n_grid+1)+m)
    elif (i <= m) & (j > m):
        return(m*(n_grid+1)+i)
    else:
        return(j*(n_grid+1)+i)



# ### showcase kay correspondance function

# for j in range(n_grid+1):
#     for i in range(n_grid+1):
        
#         ind = j*(n_grid+1)+i
        
#         indc = kay1c(j,i,m,n_grid)

#         fig, ax = plt.subplots(1,2)
        
#         ax[0].set_aspect(1)
        
        
#         ax[0].scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#         ax[0].scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],c="tab:green")
#         ax[0].scatter(loc_grid[ind,0],loc_grid[ind,1],c="tab:orange")
        
#         ax[1].set_aspect(1)
        
        
#         ax[1].scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#         ax[1].scatter(loc_grid[gNei[indc],0],loc_grid[gNei[indc],1],c="tab:green")
#         ax[1].scatter(loc_grid[indc,0],loc_grid[indc,1],c="tab:orange")
        
#         plt.show()
        
        
        
### unchanged quantities from exact


VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])


Vmmu1_current = V_current-np.outer(mu_current,np.ones(n_obs))


A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current


Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs        






















