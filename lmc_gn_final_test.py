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

import time

from base import makeGrid, vec_inv
from noisyLMC_generation import rNLMC_mu



from gn_func import phis_move, A_move_slice, mu_move, V_move_conj_scale, V_grid_move_scale, taus_move




cols = ["Blues","Oranges","Greens","Reds","Purples"]

# random.seed(0)

### number of points 
n_obs=2000
n_grid=20
# n_grid=int(np.sqrt(n_obs)-1)

### number of dimensions
p = 2

### number of neighbors

m = 3

### markov chain + tail length
N = 1000
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

### neighbors

gNei = np.zeros((n_grid+1)**2,dtype=object)
ogNei = np.zeros((n_obs,(m+1)**2),dtype=int)

### anti-neighbors

agNei = np.zeros((n_grid+1)**2,dtype=object)
agInd = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    agNei[i] = []
    agInd[i] = []

aogNei = np.zeros((n_grid+1)**2,dtype=object)
aogInd = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    aogNei[i] = []
    aogInd[i] = []

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

# for j in range(n_grid+1):
#     for i in range(n_grid+1):
        
        
#         xaNei = np.arange(i,np.min([n_grid+1,i+m+1]))
#         yaNei = np.arange(j,np.min([n_grid+1,j+m+1]))
    
#         agNei[j*(n_grid+1)+i] = np.array([jj*(n_grid+1)+ii for jj in yaNei for ii in xaNei if (ii != i) | (jj != j )],dtype=int)
         
 
## showcase grid neighbors

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

# showcase grid anti neighbors

# for j in range(n_grid+1):
#     for i in range(n_grid+1):
        
#         ind = j*(n_grid+1)+i

#         fig, ax = plt.subplots()
        
#         ax.set_aspect(1)
        
        
#         plt.scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#         plt.scatter(loc_grid[agNei[ind],0],loc_grid[agNei[ind],1],c="tab:green")
#         plt.scatter(loc_grid[ind,0],loc_grid[ind,1],c="tab:orange")
        
#         # print("i = ", i, "j = ",j)
        
#         plt.show()

### verify grid indices

# val = True

# for i in range((n_grid+1)**2):
#     for j in range(len(agInd[i])):
#         # print(gNei[agNei[i][j]][agInd[i][j]] == i)
#         val *= (gNei[agNei[i][j]][agInd[i][j]] == i)
    
# print(val)   


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

    
    ogNei[i] = np.array([ii*(n_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
    
    ind = 0 
    for j in ogNei[i]:
        aogNei[j].append(i)
        aogInd[j].append(ind)
        ind += 1
    


    
### showcase obs-grid neighbors

# for i in range(n_obs):

#     fig, ax = plt.subplots()
    
#     ax.set_aspect(1)
    
    
#     plt.scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#     plt.scatter(loc_grid[ogNei[i],0],loc_grid[ogNei[i],1],c="tab:green")
#     plt.scatter(loc_obs[i,0],loc_obs[i,1],c="tab:orange")
    
#     # plt.title(str(i))
    
#     plt.show()

## showcase obs-grid anti neighbors

# for i in range((n_grid+1)**2):

#     fig, ax = plt.subplots()
    
#     ax.set_aspect(1)
    
    
#     plt.scatter(loc_grid[:,0],loc_grid[:,1],c="black")
#     plt.scatter(loc_grid[i,0],loc_grid[i,1],c="tab:orange")
    
#     plt.scatter(loc_obs[:,0],loc_obs[:,1],c="grey")
#     plt.scatter(loc_obs[aogNei[i],0],loc_obs[aogNei[i],1],c="tab:green")

    
#     # plt.title(str(i))
    
#     plt.show()

### verify obs grid indices

# val = True

# for i in range((n_grid+1)**2):
#     for j in range(len(aogInd[i])):
        
#         val *= (ogNei[aogNei[i][j]][aogInd[i][j]] == i)
    
# print(val)  


### sizes of anti neighbor sets

# sizes = np.zeros((n_grid+1)**2)

# for i in range((n_grid+1)**2):
#     sizes[i] = len(aogInd[i])

# plt.hist(sizes)
# plt.show()

# print("max of antineighbors sets:", np.max(sizes))
# print("median of antineighbors sets:", np.median(sizes))
# print("sum of antineighbors sets:", np.sum(sizes))
    

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
        







        
        
### unchanged quantities from exact


VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])


Vmmu1_current = V_current-np.outer(mu_current,np.ones(n_obs))
V_gridmmu1_current = V_grid_current-np.outer(mu_current,np.ones((n_grid+1)**2))


A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current
A_invV_gridmmu1_current = A_inv_current @ V_gridmmu1_current


Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs        




st = time.time()

for i in range(N):
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj_scale(ogbs, ogrs, ogNei, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y_obs, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current)
    
    V_grid_current, V_gridmmu1_current, A_invV_gridmmu1_current = V_grid_move_scale(gbs, ogbs, grs, ogrs, gNei, ogNei, agNei, agInd, aogNei, aogInd, A_inv_current, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current)
    
    mu_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current = mu_move(A_inv_current,gNei,ogNei,gbs,grs,ogbs,ogrs,V_current,V_grid_current,sigma_mu,mu_mu)

    
    A_current,A_inv_current,A_invVmmu1_current,A_invV_gridmmu1_current = A_move_slice(A_current, A_invVmmu1_current, A_invV_gridmmu1_current, Vmmu1_current, V_gridmmu1_current, gNei, ogNei, gbs, grs, ogbs, ogrs, sigma_A, mu_A, sigma_slice)

    phis_current,gbs,grs,ogbs,ogrs,acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,A_invVmmu1_current,A_invV_gridmmu1_current,gNei,ogNei,dist_nei_grid,dist_pnei_grid,dist_nei_ogrid,dist_pnei_ogrid,gbs,grs,ogbs,ogrs)
    
    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

    
 


    mu_run[i] = mu_current
    V_run[i] = V_current
    taus_run[i] = taus_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    V_grid_run[i] = V_grid_current 

    
    if i % 1 == 0:
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
    plt.plot(taus_run[tail:,i])
plt.show()

print("True taus ",taus)
print("Post Mean taus ",np.mean(taus_run[tail:],axis=0))


for i in range(p):
    plt.plot(phis_run[tail:,i])
plt.show()


for i in range(p):
    for j in range(p):
        plt.plot(A_run[tail:,i,j])
plt.show()


# covariance

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




### mean processes

V_grid_mean = np.mean(V_grid_run[tail:],axis=0)


for i in range(p):
    
    


    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_true_grid[i],n_grid+1), cmap = cols[i])
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()

    xv, yv = np.meshgrid(marg_grid, marg_grid)
    
    
    
    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    
    
    c = ax.pcolormesh(xv, yv, vec_inv(V_grid_mean[i],n_grid+1), cmap = cols[i])
    plt.colorbar(c)
    # plt.savefig("aaaaa.pdf", bbox_inches='tight')
    plt.show()

















