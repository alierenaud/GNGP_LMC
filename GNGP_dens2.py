#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:22:00 2023

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix
from scipy.stats import truncnorm


import matplotlib.pyplot as plt


from base import matern_kernel, fct2, makeGrid, vec_inv

from scipy.stats import norm

random.seed(0)

n_grid = 20
m = 4


marg_grid = np.linspace(0,1,n_grid+1)



grid_locs = makeGrid(marg_grid,marg_grid)
f_grid = fct2(grid_locs)

### showcase data

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(f_grid,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.show()


n_1 = 2000


### generate data

x_true = np.zeros((0,2))
g_true = np.array([])
y_true = np.array([])

r=0
while r < n_1:
    x = random.uniform(size=(1,2))
    g = fct2(x)
    y = fct2(x) + random.normal()
    
    x_true = np.append(x_true,x,axis=0)
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




fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, norm.cdf(vec_inv(f_grid,n_grid+1)), cmap = "Blues")
# ax.scatter(x_1[:,0],x_1[:,1],c="black")
plt.colorbar(c)
plt.show()





### priors

alpha_phi = 100
beta_phi = 100



### proposals

# alpha_prop = 100
sigma_prop = 0.05

### algorithm

mu = 0

mu_grid = np.ones((n_grid+1)**2)*mu
a = 0.1
tau = 1.0

phi_current = 1


### initiiate point process

# n_0_current = n_0_true
# x_0_current = x_0_true
# g_0_current = g_0_true
# y_0_current = y_0_true



n_0_current = 0
x_0_current = np.zeros(shape=(0,2))
g_0_current = np.zeros(shape=(0,2)) + mu


# n_0_current = n_1
# x_0_current = random.uniform(size=(n_1,2))
# g_0_current = random.normal(size=n_0_current) + mu
y_0_current = -np.ones(n_0_current)



# g_1_current = g_1_true
# y_1_current = y_1_true


g_1_current = np.zeros(n_1) + mu
# g_1_current = random.normal(size=n_1) + mu
y_1_current = np.ones(n_1)


n_current = n_1 + n_0_current 
x_current = np.append(x_1,x_0_current,axis=0)
g_current = np.append(g_1_current,g_0_current)
y_current = np.append(y_1_current,y_0_current)

mu_obs = np.ones(n_current)*mu

g_grid_current = np.zeros((n_grid+1)**2)
# g_grid_current = random.normal(size=(n_grid+1)**2)
# g_grid_current = np.copy(f_grid)


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



# for i in range((n_grid+1)**2):


#     fig, ax = plt.subplots()
    
#     ax.set_aspect(1)
    
#     # ax.set_xticks([])
#     # ax.set_yticks([])
    
#     plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
#     plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
#     plt.scatter(grid_locs[gNei[i],0],grid_locs[gNei[i],1],c="tab:green")
#     plt.show()






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

ogNei = np.zeros((n_current,m**2),dtype=int)
aogNei = np.zeros((n_grid+1)**2,dtype=object)

for i in range((n_grid+1)**2):
    aogNei[i] = np.empty(0,dtype=int)

Distogs = np.zeros((n_current,m**2))




for i in range(n_current):
    
    
    left_col =  int(np.floor(x_current[i,0]  * n_grid))
    down_row =  int(np.floor(x_current[i,1]  * n_grid))
    
    
    
    
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
    
    
    
    
    cs = 0
    for v in ver_ind:
        ogNei[i,np.arange(cs*m,cs*m+m)] = v*(n_grid+1) + hor_ind
        cs+=1
        
    
    Distogs[i] = distance_matrix([x_current[i]], grid_locs[ogNei[i]])
    
    
    # fig, ax = plt.subplots()
    
    # ax.set_aspect(1)
    
    # # ax.set_xticks([])
    # # ax.set_yticks([])
    
    # plt.scatter(grid_locs[:,0],grid_locs[:,1],c="black")
    
    # # plt.scatter(grid_locs[i,0],grid_locs[i,1],c="tab:orange")
    # plt.scatter(grid_locs[ogNei[i],0],grid_locs[ogNei[i],1],c="tab:green")
    # plt.scatter(x_current[i,0],x_current[i,1],c="tab:orange")
    # plt.show()
    
    
    
    
    for j in ogNei[i]:
        aogNei[j] = np.append(aogNei[j],i)
    

    
DistggMat = Distg[ogNei[0]][:,ogNei[0]]
Bog_current = np.zeros((n_current,(n_grid+1)**2))
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

N = 2000

g_grid_run = np.zeros((N,(n_grid+1)**2))
phi_run = np.zeros(N)
n_0_run = np.zeros(N)

acc_phi = np.zeros(N)


from time import time

st = time()

for i in range(N):

    if i%100==0:

        fig, ax = plt.subplots()
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        ax.set_box_aspect(1)



        c = ax.pcolormesh(xv, yv, norm.cdf(vec_inv(g_grid_current,n_grid+1)), cmap = "Blues")
        ax.scatter(x_0_current[:,0],x_0_current[:,1],c="black")
        plt.colorbar(c)
        plt.show()

        print(i)

        
        print(phi_current)
        
        
    # ## point process (update x_0,g_0,y_0)    
        
    # count = 0
    # num = 0
    
    # n_0_current = 0
    # x_0_current = np.array([])
    # g_0_current = np.array([])
    # y_0_current = np.array([])
    
    # ogNei_0_current = np.zeros((0,m),dtype=int)
    
    # Distogs_0_current = np.zeros((0,m))
    
    # Bog_0_current = np.zeros(shape=(0,n_grid+1))
    # rog_0_current = np.array([])
    
    
    # #### remove neighbor rejections
    
    # for ii in range(n_grid+1):
        
    #     aogNei[ii] = aogNei[ii][aogNei[ii]<n_1]
    
    
    # while count < n_1:
        
    
        
        
    #     x_new = random.uniform() * 20 - 10
        
        
    #     ## compute neighbors on grid
        
    #     leftNei = int(np.floor( (x_new + xlim)/(2*xlim) * n_grid))
        
    #     off_left = -min(leftNei - m//2 + 1,0)
    #     off_right = max(leftNei + m//2,n_grid) - n_grid
        
    #     leftMostNei = leftNei-m//2+1+off_left-off_right
    #     rightMostNei = leftNei + m//2 + 1 + off_left - off_right
        
    #     ogNei_new = np.arange(leftMostNei,rightMostNei)
        
        
    #     Distogs_new = distance_matrix([[x_new]], np.transpose([grid_locs[ogNei_new]]))
        
        
        
    #     CovMatgg_temp = matern_kernel(DistggMat,phi_current)
        
    #     R_inv_temp = np.linalg.inv(CovMatgg_temp)
        
        
    #     r_temp = matern_kernel(Distogs_new,phi_current)
        
    
        
    #     b_temp = r_temp@R_inv_temp
        
        
    #     rog_temp = 1-np.inner(b_temp,r_temp)
        
    #     g_new = np.sqrt(rog_temp/a)*random.normal() + np.inner(b_temp,g_grid_current[ogNei_new] - mu_grid[ogNei_new]) + mu
    #     y_new = g_new + random.normal()
        
    #     if y_new < 0:
            
    #         #### update 
            
    #         n_0_current += 1
    #         x_0_current = np.append(x_0_current, x_new)
    #         g_0_current = np.append(g_0_current, g_new)
    #         y_0_current = np.append(y_0_current, y_new)
            
            
    #         ogNei_0_current = np.append(ogNei_0_current,[ogNei_new],axis=0)
            
    #         Distogs_0_current = np.append(Distogs_0_current,Distogs_new,axis=0)
            
            
    #         Bog_new = np.zeros(n_grid+1)
    #         Bog_new[ogNei_new] = b_temp
            
    #         Bog_0_current = np.append(Bog_0_current,[Bog_new],axis=0)
    #         rog_0_current = np.append(rog_0_current,rog_temp)
            
    #         for nei in ogNei_new:
    #             aogNei[nei] = np.append(aogNei[nei],num+n_1)
         
    #         num += 1
            
    #     else:
    #         count += 1
    
    
    
    
    # #### append to current quantities for obs
    
    # n_current = n_1 + n_0_current
    # x_current = np.append(x_1,x_0_current)
    # g_current = np.append(g_1_current,g_0_current)
    # y_current = np.append(y_1_current,y_0_current)
    
    # ogNei = np.append(ogNei[:n_1], ogNei_0_current,axis=0)
    
    # Distogs = np.append(Distogs[:n_1],Distogs_0_current,axis=0)
    
    # Bog_current = np.append(Bog_current[:n_1],Bog_0_current,axis=0)
    # rog_current = np.append(rog_current[:n_1],rog_0_current)
    
    # mu_obs = mu*np.ones(n_current)

    ## point process (update x_0,g_0,y_0)
    
    count = 0
    x_cont= np.zeros(shape=(0,2))
    g_cont= []
    y_cont= []
    
    ogNei_cont = np.zeros(shape=(0,m**2),dtype=int)
    
    
    
    
    Distogs_cont = np.zeros(shape=(0,m**2))
    
    Bog_cont = np.zeros(shape=(0,(n_grid+1)**2))
    rog_cont = []
    
    while True:
        
        
        ### simulate n_1 new variables 
        x_new = random.uniform(size=(3*n_1,2))
        n_new = x_new.shape[0]
        
        ### compute obs neighbors on grid

        ogNei_new = np.zeros(shape=(n_new,m**2),dtype=int)
        

        # Distog = distance_matrix(np.transpose([x_current]), np.transpose([grid_locs]))

        Distogs_new = np.zeros(shape=(n_new,m**2))


        for ii in range(n_new):
            
            
            
            
            
            
            
            left_col =  int(np.floor(x_new[ii,0]  * n_grid))
            down_row =  int(np.floor(x_new[ii,1]  * n_grid))
            
            
            
            
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
            
            
            
            
            cs = 0
            for v in ver_ind:
                ogNei_new[ii,np.arange(cs*m,cs*m+m)] = v*(n_grid+1) + hor_ind
                cs+=1
                
            
            Distogs_new[ii] = distance_matrix([x_new[ii]], grid_locs[ogNei_new[ii]])

            
        
        Bog_new = np.zeros((n_new,(n_grid+1)**2))
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
            g_new[ii] = np.sqrt(rog_new[ii]/a)*random.normal() + np.inner(Bog_new[ii,ogNei_new[ii]],g_grid_current[ogNei_new[ii]] - mu_grid[ogNei_new[ii]]) + mu
    
        y_new = g_new + random.normal(size=n_new)
        

        x_cont = np.append(x_cont, x_new, axis=0)  
        g_cont = np.append(g_cont, g_new) 
        y_cont = np.append(y_cont, y_new) 
        
        ogNei_cont = np.append(ogNei_cont, ogNei_new, axis=0)
        
        Distogs_cont = np.append(Distogs_cont, Distogs_new, axis=0)
        
        Bog_cont = np.append(Bog_cont, Bog_new, axis=0)
        rog_cont = np.append(rog_cont, rog_new)
                
        count += np.sum(y_new>0)
        
        if count >= n_1:
            
            n_tail = np.where(y_cont > 0)[0][n_1-1]
            
            x_cont = x_cont[:n_tail+1]
            g_cont = g_cont[:n_tail+1]
            y_cont = y_cont[:n_tail+1]
            
            ogNei_cont = ogNei_cont[:n_tail+1]
            
            
            Distogs_cont = Distogs_cont[:n_tail+1]
            
            Bog_cont = Bog_cont[:n_tail+1]
            rog_cont = rog_cont[:n_tail+1]
            
            thinned = y_cont < 0 
            
            
            n_0_current = np.sum(thinned)
            
            n_0_run[i] = n_0_current
            
            x_0_current = x_cont[thinned]
            g_0_current = g_cont[thinned]
            y_0_current = y_cont[thinned]
            
            ogNei = np.append(ogNei[:n_1],ogNei_cont[thinned],axis=0)
            Distogs = np.append(Distogs[:n_1],Distogs_cont[thinned], axis=0)
            
            
            
            Bog_current = np.append(Bog_current[:n_1],Bog_cont[thinned],axis=0)
            rog_current = np.append(rog_current[:n_1],rog_cont[thinned])
            
          
            n_current = n_1 + n_0_current 
            x_current = np.append(x_1,x_0_current)
            g_current = np.append(g_1_current,g_0_current)
            y_current = np.append(y_1_current,y_0_current)
            
            mu_obs = mu*np.ones(n_current)
            
            
            for ii in range((n_grid+1)**2):
                aogNei[ii] = np.empty(0,dtype=int)
            
            for ii in range(n_current):
                for jj in ogNei[ii]:
                    aogNei[jj] = np.append(aogNei[jj],ii)
                
            
            
            
            
            
            
            break
        
        else:
            print("not enough points")
            


    # w_grid update
    
    for ii in random.permutation(range((n_grid+1)**2)):
    # for ii in range(n_grid+1):
        
        
        A_temp = a/rg_current[ii] + np.sum([a/rg_current[jj]*Bg_current[jj,ii]**2 for jj in agNei[ii]]) + np.sum([a/rog_current[jj]*Bog_current[jj,ii]**2 for jj in aogNei[ii]])
        
        B_temp = a/rg_current[ii]*(mu_grid[ii] + np.inner(Bg_current[ii,gNei[ii]],g_grid_current[gNei[ii]]-mu_grid[gNei[ii]]) ) + np.sum([a*Bg_current[jj,ii]/rg_current[jj]*(g_grid_current[jj] - mu_grid[jj] - np.inner(Bg_current[jj,gNei[jj]],g_grid_current[gNei[jj]] - mu_grid[gNei[jj]]) + Bg_current[jj,ii]*g_grid_current[ii] ) for jj in agNei[ii]])  +  np.sum([a*Bog_current[jj,ii]/rog_current[jj]*(g_current[jj] - mu_obs[jj] - np.inner(Bog_current[jj,ogNei[jj]],g_grid_current[ogNei[jj]] - mu_grid[ogNei[jj]]) + Bog_current[jj,ii]*g_grid_current[ii] ) for jj in aogNei[ii]])             
        
        g_grid_current[ii] = 1/np.sqrt(A_temp)*random.normal() + B_temp/A_temp

    
    # w_obs update


    for ii in range(n_current):
        
        a_temp = a/rog_current[ii] + tau 
        b_temp = a/rog_current[ii]*(mu_obs[ii] + np.inner(Bog_current[ii,ogNei[ii]],g_grid_current[ogNei[ii]] - mu_grid[ogNei[ii]])) + tau*y_current[ii]
        
        g_current[ii] = 1/np.sqrt(a_temp)*random.normal() + b_temp/a_temp
    
    g_0_current = g_current[n_1:]
    g_1_current = g_current[:n_1]
    
    
    g_grid_run[i] = g_grid_current
    # g_current_run[i] = g_current
    
    ## y_update
    
    for ii in range(n_1):
        
        y_current[ii] = truncnorm.rvs(a=-g_current[ii],b=np.inf,loc=g_current[ii])
        
    for ii in range(n_1,n_current):
        
        y_current[ii] = truncnorm.rvs(a=-np.inf,b=-g_current[ii],loc=g_current[ii])
    
    y_0_current = y_current[n_1:]
    y_1_current = y_current[:n_1]
    
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
    
    Bog_new = np.zeros((n_current,(n_grid+1)**2))
    rog_new = np.zeros(n_current)
    
    for ii in range(n_current):

        
        
        CovMatgg_temp = matern_kernel(DistggMat,phi_new)
        
        R_inv_temp = np.linalg.inv(CovMatgg_temp)
        
        DistMatog_temp = Distogs[ii]
        r_temp = matern_kernel(DistMatog_temp,phi_new)
        

        
        b_temp = r_temp@R_inv_temp
        

        Bog_new[ii][ogNei[ii]] = b_temp
        
        rog_new[ii] = 1-np.inner(b_temp,r_temp)
    
    sus_grid = np.exp(- a/2* np.sum([ (g_grid_current[ii] - mu_grid[ii] - np.inner(Bg_new[ii,gNei[ii]],g_grid_current[gNei[ii]]-mu_grid[gNei[ii]]))**2/rg_new[ii] - (g_grid_current[ii] - mu_grid[ii] - np.inner(Bg_current[ii,gNei[ii]],g_grid_current[gNei[ii]]-mu_grid[gNei[ii]]))**2/rg_current[ii] for ii in range(n_grid+1)]))
    
    # print("sus grid",sus_grid)
    
    sus_obs = np.exp(- a/2* np.sum([ (g_current[ii] - mu_obs[ii] - np.inner(Bog_new[ii,ogNei[ii]],g_grid_current[ogNei[ii]]-mu_grid[ogNei[ii]]))**2/rog_new[ii] - (g_current[ii] - mu_obs[ii] - np.inner(Bog_current[ii,ogNei[ii]],g_grid_current[ogNei[ii]]-mu_grid[ogNei[ii]]))**2/rog_current[ii] for ii in range(n_current)]))
    
    # print("sus obs",sus_obs)
    
    pect_grid = np.prod([(rg_current[ii]/rg_new[ii])**(1/2) for ii in range(n_grid+1)])
    
    # print("pect grid",pect_grid)
    
    pect_obs = np.prod([(rog_current[ii]/rog_new[ii])**(1/2) for ii in range(n_current)])
    
    # print("pect obs",pect_obs)
    
    prior = (phi_new/phi_current)**(alpha_phi-1) * np.exp(-beta_phi*(phi_new-phi_current))
        
    # print("prior",prior)
    
    # trans = (phi_current/phi_new)**(alpha_prop-1) * np.exp(-alpha_prop*(phi_current/phi_new - phi_new/phi_current))
    
    # print("trans",trans)

    
    ratio =  sus_grid * pect_grid * sus_obs * pect_obs * prior
    
    if random.uniform() < ratio:
        phi_current = phi_new
        Bg_current = np.copy(Bg_new)
        rg_current = np.copy(rg_new)
        Bog_current = np.copy(Bog_new)
        rog_current = np.copy(rog_new)
        acc_phi[i] = 1
   
    phi_run[i] = phi_current 
    


et = time()
print("Time:",(et-st)/60,"minutes")

tail = 1000

Phi_g_grid_run = norm.cdf(g_grid_run)
maxes = np.max(Phi_g_grid_run,axis=1)


for i in range(N):
    Phi_g_grid_run[i]=Phi_g_grid_run[i]/maxes[i]

Phi_g_grid_mean = np.mean(Phi_g_grid_run[tail:], axis=0)
Phi_g_grid_025 = np.quantile(Phi_g_grid_run[tail:], 0.05, axis=0)
Phi_g_grid_975 = np.quantile(Phi_g_grid_run[tail:], 0.95, axis=0)


fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, vec_inv(Phi_g_grid_mean,n_grid+1), cmap = "Blues")
plt.colorbar(c)
plt.show()


print("Accept rate phi:",np.mean(acc_phi))
### trace plots

plt.plot(phi_run[tail:])
plt.show()

plt.boxplot(phi_run[tail:])
plt.show()

plt.plot(n_0_run[tail:])
plt.show()

plt.boxplot(n_0_run[tail:])
plt.show()




print("MSE:", np.mean((norm.cdf(f_grid) - Phi_g_grid_mean)**2))
print("TMSE:", np.mean((Phi_g_grid_run[tail:] - norm.cdf(f_grid))**2))



