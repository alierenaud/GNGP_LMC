#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:01:33 2023

@author: homeboy
"""

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from base import matern_kernel, fct2, makeGrid, vec_inv

random.seed(0)

n_obs=2000
m=3

n_grid = 11

xlim=10

marg_grid = np.linspace(0,1,n_grid+1)
grid_locs = makeGrid(marg_grid,marg_grid)

locs = random.uniform(size = (n_obs,2))

### distance function


### parameters

mu = np.zeros(n_obs)
mu_grid = np.zeros((n_grid+1)**2)
a = 1
phi_current = 1
tau = 10


### priors

alpha_phi = 100
beta_phi = 100



### proposals

# alpha_prop = 100
sigma_prop = 0.02

### compute grid neighbors

gNei = np.zeros((n_grid+1)**2,dtype=object)






for i in np.arange((n_grid+1)**2):
    
    row = i//(n_grid+1)
    col = i%(n_grid+1)
    
    
    # print(i)
    gNei[i] = np.array([],dtype=int)
    
    

    for j in np.arange(np.max([row-m,0]),row):
        gNei[i] = np.append(gNei[i], np.arange(np.max([(n_grid+1)*j+col-m,(n_grid+1)*j]) ,(n_grid+1)*j+col+1,dtype=int))
        
      
    gNei[i] = np.append(gNei[i],np.arange(np.max([i-m,i-col]),i,dtype=int)) 


### showcase neighboring structure



fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])



aw = 0.02
al = 0.03



for i in range((n_grid+1)**2):

    # if (grid_locs[i,0] < 1) & (grid_locs[i,1] < 1) :
    #     plt.arrow(grid_locs[i,0], grid_locs[i,1], 1/n_grid, 1/n_grid,length_includes_head=True,head_width=aw,head_length=al,color="tab:blue")
    if (grid_locs[i,1] < 1) :
        plt.arrow(grid_locs[i,0], grid_locs[i,1], 0, 1/n_grid,length_includes_head=True,head_width=aw,head_length=al,color="tab:blue")
    if (grid_locs[i,0] < 1) :
        plt.arrow(grid_locs[i,0], grid_locs[i,1], 1/n_grid, 0,length_includes_head=True,head_width=aw,head_length=al,color="tab:blue")

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")

# plt.show()
plt.savefig("grid_points.pdf", bbox_inches='tight')


### order

fig, ax = plt.subplots()

ax.set_aspect(1)


plt.scatter(grid_locs[:,0],grid_locs[:,1],c="white")

for i in range((n_grid+1)**2):
    plt.text(grid_locs[i,0],grid_locs[i,1], str(i+1),color="black", fontsize=8, ha="center", va = "center")


# plt.show()
plt.savefig("grid_order.pdf", bbox_inches='tight')



i = 27

fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
plt.savefig("NeiA.pdf", bbox_inches='tight')
# plt.show()

i = 35

fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
plt.savefig("NeiB.pdf", bbox_inches='tight')
# plt.show()

i = 102

fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
plt.savefig("NeiC.pdf", bbox_inches='tight')
# plt.show()

i = 134

fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
plt.savefig("NeiD.pdf", bbox_inches='tight')
# plt.show()


### neighboring patterns 


fig, ax = plt.subplots()

ax.set_aspect(1)

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")

topRight = (grid_locs[:,0] > 0.2) & (grid_locs[:,1] > 0.2)

plt.scatter(grid_locs[topRight,0],grid_locs[topRight,1],c="tab:orange")

plt.savefig("topRight.pdf", bbox_inches='tight')
# plt.show()



fig, ax = plt.subplots()

ax.set_aspect(1)

plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")



plt.scatter(grid_locs[0,0],grid_locs[0,1],c="tab:orange")
plt.scatter(grid_locs[1,0],grid_locs[1,1],c="tab:orange")
plt.scatter(grid_locs[2,0],grid_locs[2,1],c="tab:orange")
plt.scatter(grid_locs[3,0],grid_locs[3,1],c="tab:orange")
plt.scatter(grid_locs[13,0],grid_locs[13,1],c="tab:orange")
plt.scatter(grid_locs[14,0],grid_locs[14,1],c="tab:orange")
plt.scatter(grid_locs[15,0],grid_locs[15,1],c="tab:orange")
plt.scatter(grid_locs[26,0],grid_locs[26,1],c="tab:orange")
plt.scatter(grid_locs[27,0],grid_locs[27,1],c="tab:orange")
plt.scatter(grid_locs[39,0],grid_locs[39,1],c="tab:orange")

plt.savefig("triangle.pdf", bbox_inches='tight')
plt.show()


# for i in range((n_grid+1)**2):


#     fig, ax = plt.subplots()
    
#     ax.set_aspect(1)
    
#     # ax.set_xticks([])
#     # ax.set_yticks([])
    
#     plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
#     plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
#     plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
#     plt.show()


#### NODE COLOURING

fig, ax = plt.subplots()

ax.set_aspect(1)

# ax.set_xticks([])
# ax.set_yticks([])

for i in range((n_grid+1)**2):
    
    if (grid_locs[i,0]*(n_grid) % ((m+1)*2) < (m+1)) & (grid_locs[i,1]*(n_grid) % ((m+1)*2) < (m+1)):
        plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:blue")
    if (grid_locs[i,0]*(n_grid) % ((m+1)*2) < (m+1)) & (grid_locs[i,1]*(n_grid) % ((m+1)*2) >= (m+1)):
        plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
    if (grid_locs[i,0]*(n_grid) % ((m+1)*2) >= (m+1)) & (grid_locs[i,1]*(n_grid) % ((m+1)*2) < (m+1)):
        plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:green")
    if (grid_locs[i,0]*(n_grid) % ((m+1)*2) >= (m+1)) & (grid_locs[i,1]*(n_grid) % ((m+1)*2) >= (m+1)):
        plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:purple")        

plt.show()


agNei = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    agNei[i] = np.array([],dtype = int)
    for j in range((n_grid+1)**2):
        if i in gNei[j]:
            agNei[i] = np.append(agNei[i],j)




### compute B,r,dists

Distg = distance_matrix(grid_locs, grid_locs)

DistgMats = np.zeros((n_grid+1)**2,dtype=object)
Bg_current = np.zeros(((n_grid+1)**2,(n_grid+1)**2))
rg_current= np.zeros((n_grid+1)**2)

for i in range((n_grid+1)**2):

    
    DistgMat_temp = Distg[np.append(gNei[i], i)][:,np.append(gNei[i], i)]
    
    DistgMats[i] =  DistgMat_temp
    
    cov_mat_temp = matern_kernel(DistgMat_temp,phi_current)
    
    ngNei_temp = gNei[i].shape[0]
    
    R_inv_temp = np.linalg.inv(cov_mat_temp[:ngNei_temp,:ngNei_temp])
    r_temp = cov_mat_temp[ngNei_temp,:ngNei_temp]

    
    b_temp = r_temp@R_inv_temp
    

    Bg_current[i][gNei[i]] = b_temp
    
    rg_current[i] = 1-np.inner(b_temp,r_temp)



### check on structure

# C = Bg_current+np.identity((n_grid+1)**2)
# D = np.transpose(C)@C                    

# print((D!=0)*1)

### compute obs neighbors on grid

ogNei = np.zeros(n_obs,dtype=object)
aogNei = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    aogNei[i] = np.empty(0,dtype=int)

Distog = distance_matrix(locs, grid_locs)




for i in range(n_obs):
    
    
    left_col =  int(np.floor(locs[i,0]  * n_grid))
    down_row =  int(np.floor(locs[i,1]  * n_grid))
    
    
    
    
    off_left = -min(left_col - m//2 + 1,0)
    off_right = max(left_col + m//2,n_grid) - n_grid
    
    leftMostNei = left_col-m//2+1+off_left-off_right
    rightMostNei = left_col + 1 + m//2 + off_left - off_right
    
    hor_ind = np.arange(leftMostNei,rightMostNei)
    
    
    off_down = -min(down_row - m//2 + 1,0)
    off_up = max(down_row + m//2,n_grid) - n_grid
    
    downMostNei = down_row-m//2+1+off_down-off_up
    upMostNei = down_row + m//2 + 1 + off_down - off_up
    
    ver_ind = np.arange(downMostNei,upMostNei)
    
    
    ogNei[i] = np.array([],dtype=int)
    
    for v in ver_ind:
        ogNei[i] = np.append(ogNei[i],v*(n_grid+1) + hor_ind)
        
    
    
    
    
    # fig, ax = plt.subplots()
    
    # ax.set_aspect(1)
    
    # # ax.set_xticks([])
    # # ax.set_yticks([])
    
    # plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
    # plt.scatter(locs[i,0],locs[i,1],c="tab:orange")
    # # plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
    # plt.scatter(grid_locs[ogNei[i],0],grid_locs[ogNei[i],1],c="tab:green")
    # plt.show()
    
    
    
    
    for j in ogNei[i]:
        aogNei[j] = np.append(aogNei[j],i)
    

    
DistggMats = np.zeros(n_obs,dtype=object)
Bog_current = np.zeros((n_obs,(n_grid+1)**2))
rog_current = np.zeros(n_obs)


for i in range(n_obs):

    
    DistMatog_temp = Distg[ogNei[i]][:,ogNei[i]]
    DistggMats[i]  = DistMatog_temp
    CovMatgg_temp = matern_kernel(DistMatog_temp,phi_current)
    
    R_inv_temp = np.linalg.inv(CovMatgg_temp)
    
    DistMatog_temp = Distog[i][ogNei[i]]
    r_temp = matern_kernel(DistMatog_temp,phi_current)
    

    
    b_temp = r_temp@R_inv_temp
    

    Bog_current[i][ogNei[i]] = b_temp
    
    rog_current[i] = 1-np.inner(b_temp,r_temp)


### simulate an example y


w_true = fct2(locs)
w_grid_true = fct2(grid_locs)
y = w_true + 1/np.sqrt(tau)*random.normal(size = n_obs)
### showcase data

xv, yv = np.meshgrid(marg_grid, marg_grid)



plt.figure(figsize=(4,4))
fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(w_grid_true,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.title("True")
# plt.savefig("True_1000.pdf", bbox_inches='tight')
plt.show()


# w_current = w_true
# w_current = np.load("w_current.npy")
# w_current = random.normal(size=n_obs)
w_current = np.zeros(shape=n_obs)

# w_grid = np.copy(w_grid_true)
# w_grid = np.load("w_grid.npy")
# w_grid = random.normal(size=(n_grid+1)**2)
w_grid = np.zeros(shape=(n_grid+1)**2)




### algorithm


N = 2000

w_grid_run = np.zeros((N,(n_grid+1)**2))
w_current_run = np.zeros((N,n_obs))
phi_run = np.zeros(N)

acc_phi = np.zeros(N)


### containers

Bg_new = np.zeros(((n_grid+1)**2,(n_grid+1)**2))
rg_new = np.zeros((n_grid+1)**2)

Bog_new = np.zeros((n_obs,(n_grid+1)**2))
rog_new = np.zeros(n_obs)

import time

st = time.time()

for i in range(N):
    
    
    
    if i % 100 ==0:
        fig, ax = plt.subplots()
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        ax.set_box_aspect(1)



        c = ax.pcolormesh(xv, yv, vec_inv(w_grid,n_grid+1), cmap = "Blues")
        plt.colorbar(c)
        plt.show()
        print(i)
        print(phi_current)
    
    # w_grid update
    
    for ii in random.permutation(range((n_grid+1)**2)):
    # for ii in range(n_grid+1):
        
        
        A_temp = a/rg_current[ii] + np.sum([a/rg_current[jj]*Bg_current[jj,ii]**2 for jj in agNei[ii]]) + np.sum([a/rog_current[jj]*Bog_current[jj,ii]**2 for jj in aogNei[ii]])
        
        B_temp = a/rg_current[ii]*(mu_grid[ii] + np.inner(Bg_current[ii,gNei[ii]],w_grid[gNei[ii]]-mu_grid[gNei[ii]]) ) + np.sum([a*Bg_current[jj,ii]/rg_current[jj]*(w_grid[jj] - mu_grid[jj] - np.inner(Bg_current[jj,gNei[jj]],w_grid[gNei[jj]] - mu_grid[gNei[jj]]) + Bg_current[jj,ii]*w_grid[ii] ) for jj in agNei[ii]])  +  np.sum([a*Bog_current[jj,ii]/rog_current[jj]*(w_current[jj] - mu[jj] - np.inner(Bog_current[jj,ogNei[jj]],w_grid[ogNei[jj]] - mu_grid[ogNei[jj]]) + Bog_current[jj,ii]*w_grid[ii] ) for jj in aogNei[ii]])             
        
        w_grid[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp

    
    # w_obs update


    for ii in range(n_obs):
        
        a_temp = a/rog_current[ii] + tau 
        b_temp = a/rog_current[ii]*(mu[ii] + np.inner(Bog_current[ii,ogNei[ii]],w_grid[ogNei[ii]] - mu_grid[ogNei[ii]])) + tau*y[ii]
        
        w_current[ii] = 1/np.sqrt(a_temp)*random.normal() + b_temp/a_temp
    
    w_grid_run[i] = w_grid
    w_current_run[i] = w_current
    
    ### phi update
    
    # phi_new = random.gamma(alpha_prop,1/alpha_prop) * phi_current
    
    while True:
        phi_new = sigma_prop*random.normal() + phi_current
        if phi_new > 0:
            break
    
    
    Bg_new = np.zeros(((n_grid+1)**2,(n_grid+1)**2))
    rg_new= np.zeros((n_grid+1)**2)
    
    for ii in range((n_grid+1)**2):

        
        DistgMat_temp = DistgMats[ii]
        
        cov_mat_temp = matern_kernel(DistgMat_temp,phi_new)
        
        ngNei_temp = gNei[ii].shape[0]
        
        R_inv_temp = np.linalg.inv(cov_mat_temp[:ngNei_temp,:ngNei_temp])
        r_temp = cov_mat_temp[ngNei_temp,:ngNei_temp]

        
        b_temp = r_temp@R_inv_temp
        

        Bg_new[ii][gNei[ii]] = b_temp
        
        rg_new[ii] = 1-np.inner(b_temp,r_temp)
    
    Bog_new = np.zeros((n_obs,(n_grid+1)**2))
    rog_new = np.zeros(n_obs)
    
    for ii in range(n_obs):

        
        DistMatog_temp = DistggMats[ii]
        CovMatgg_temp = matern_kernel(DistMatog_temp,phi_new)
        
        R_inv_temp = np.linalg.inv(CovMatgg_temp)
        
        DistMatog_temp = Distog[ii][ogNei[ii]]
        r_temp = matern_kernel(DistMatog_temp,phi_new)
        

        
        b_temp = r_temp@R_inv_temp
        

        Bog_new[ii][ogNei[ii]] = b_temp
        
        rog_new[ii] = 1-np.inner(b_temp,r_temp)
    
    sus_grid = - a/2* np.sum([ (w_grid[ii] - mu_grid[ii] - np.inner(Bg_new[ii,gNei[ii]],w_grid[gNei[ii]]-mu_grid[gNei[ii]]))**2/rg_new[ii] - (w_grid[ii] - mu_grid[ii] - np.inner(Bg_current[ii,gNei[ii]],w_grid[gNei[ii]]-mu_grid[gNei[ii]]))**2/rg_current[ii] for ii in range((n_grid+1)**2)])
    
    # print("sus grid",sus_grid)
    
    sus_obs = - a/2* np.sum([ (w_current[ii] - mu[ii] - np.inner(Bog_new[ii,ogNei[ii]],w_grid[ogNei[ii]]-mu_grid[ogNei[ii]]))**2/rog_new[ii] - (w_current[ii] - mu[ii] - np.inner(Bog_current[ii,ogNei[ii]],w_grid[ogNei[ii]]-mu_grid[ogNei[ii]]))**2/rog_current[ii] for ii in range(n_obs)])
    
    # print("sus obs",sus_obs)
    
    pect_grid = np.sum([(1/2)*np.log(rg_current[ii]/rg_new[ii]) for ii in range((n_grid+1)**2)])
    
    # print("pect grid",pect_grid)
    
    pect_obs = np.sum([(1/2)*np.log(rog_current[ii]/rog_new[ii]) for ii in range(n_obs)])
    
    # print("pect obs",pect_obs)
    
    prior = (alpha_phi-1)*np.log(phi_new/phi_current) -beta_phi*(phi_new-phi_current)
        
    # print("prior",prior)
    
    # trans = (phi_current/phi_new)**(alpha_prop-1) * np.exp(-alpha_prop*(phi_current/phi_new - phi_new/phi_current))
    
    # print("trans",trans)

    
    ratio =  np.exp(sus_grid + pect_grid + sus_obs + pect_obs + prior )
    
    if random.uniform() < ratio:
        phi_current = phi_new
        Bg_current = np.copy(Bg_new)
        rg_current = np.copy(rg_new)
        Bog_current = np.copy(Bog_new)
        rog_current = np.copy(rog_new)
        acc_phi[i] = 1
   
    phi_run[i] = phi_current 
    
et = time.time()

print("Total Time:", (et-st)/60, "minutes")

tail = 1000

print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

plt.boxplot(phi_run[tail:])
plt.show()

w_grid_mean = np.mean(w_grid_run[tail:], axis=0)
w_grid_025 = np.quantile(w_grid_run[tail:], 0.025, axis=0)
w_grid_975 = np.quantile(w_grid_run[tail:], 0.975, axis=0)

plt.figure(figsize=(4,4))
fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(w_grid_mean,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.title("GNGP")
# plt.savefig("mean_GNGP_2000.pdf", bbox_inches='tight')
plt.show()
 
            






print("MSE:", np.mean((w_grid_true - w_grid_mean)**2))
print("TMSE:", np.mean((w_grid_run[tail:] - w_grid_true)**2))


# np.save("phi_run_2000_GNGP",phi_run)

# #### compare both distributions

tail=1000

phi_run_2000_GNGP = np.load("phi_run_2000_GNGP.npy")
phi_run_2000_GP = np.load("phi_run_2000_GP.npy")

my_dict = {'GP': phi_run_2000_GP[tail:], 'GNGP': phi_run_2000_GNGP[tail:]}
plt.boxplot(my_dict.values(), labels=my_dict.keys())
plt.title("Posterior Distribution of Phi")
# plt.savefig("post_lin_2000.pdf", bbox_inches='tight')
plt.show()





