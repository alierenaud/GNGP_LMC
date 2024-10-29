#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:19:28 2024

@author: homeboy
"""


import numpy as np
from numpy import random

from base import makeGrid

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

def ell(s,n_grid,m):
    
    if s*n_grid < (m+1)/2:
        return(0)
    elif s*n_grid > n_grid - (m+1)/2:
        return(n_grid-m)
    else:
        return(np.ceil(s*n_grid)- (m+1)/2)

random.seed(0)

### obs locations
n_obs = 1000
loc_obs = random.uniform(size=(n_obs,2))

### reorder
loc_obs = loc_obs[np.argsort(loc_obs[:,0]+loc_obs[:,1])]


### grid locations
n_marg_grid = 17
margGrid = np.linspace(0, 1, n_marg_grid+1)
loc_grid = makeGrid(margGrid,margGrid)
n_grid_tot = loc_grid.shape[0]


### number neighbors

m_obs = 16
m_grid = 3


### distance matrix

D_obs = distance_matrix(loc_obs,loc_obs)
D_grid_obs = distance_matrix(loc_grid,loc_obs)

# ### obs on grid

# fig, ax = plt.subplots()
# ax.set_box_aspect(1)

# ax.scatter(loc_grid[:,0],loc_grid[:,1],color="lightgrey",s=40,marker="s")
# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black",s=40,marker="o")
# # plt.savefig("w_obs_grid.pdf", format="pdf", bbox_inches="tight")
# plt.show()


# for i in range(n_obs):
    
#     ### compute neighbors 
    
#     nei = np.argsort(D_obs[i,:i])[:np.min([m_obs,i])]
    
#     fig, ax = plt.subplots()
#     ax.set_box_aspect(1)

#     ax.scatter(loc_grid[:,0],loc_grid[:,1],color="lightgrey",s=40,marker="s")
#     ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black",s=40,marker="o")
#     ax.scatter(loc_obs[i,0],loc_obs[i,1],color="tab:orange",s=40,marker="o")
#     ax.scatter(loc_obs[nei ,0],loc_obs[nei ,1],color="tab:green",s=40,marker="o")
#     # plt.title(i)
#     # plt.savefig("w_obs_obs"+str(i)+".pdf", format="pdf", bbox_inches="tight")
#     plt.show()
    
    
# for i in range(n_grid_tot):
    
#     ### compute neighbors 
    
#     nei = np.argsort(D_grid_obs[i])[:m_obs]
    
#     fig, ax = plt.subplots()
#     ax.set_box_aspect(1)

#     ax.scatter(loc_grid[:,0],loc_grid[:,1],color="lightgrey",s=40,marker="s")
#     ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black",s=40,marker="o")
#     ax.scatter(loc_grid[i,0],loc_grid[i,1],color="tab:orange",s=40,marker="s")
#     ax.scatter(loc_obs[nei ,0],loc_obs[nei ,1],color="tab:green",s=40,marker="o")
#     # plt.title(i)
#     # plt.savefig("w_grid_obs"+str(i)+".pdf", format="pdf", bbox_inches="tight")
#     plt.show()   


### obs on grid

# fig, ax = plt.subplots()
# ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
# ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
# # plt.savefig("w_grid_obs.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# for j in range(n_marg_grid+1):
#     for i in range(n_marg_grid+1):
        
#         ind = j*(n_marg_grid+1)+i
        
#         xNei = np.arange(np.max([0,i-m_grid]),i+1)
#         yNei = np.arange(np.max([0,j-m_grid]),j+1)

#         gNei= np.array([jj*(n_marg_grid+1)+ii for jj in yNei for ii in xNei if (ii != i) | (jj != j )],dtype=int)
        
#         fig, ax = plt.subplots()
#         ax.set_box_aspect(1)
        
#         ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
#         ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
#         ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
#         ax.scatter(loc_grid[gNei,0],loc_grid[gNei,1],color="tab:green",s=40,marker="s")
#         # plt.title(str(i)+"-"+str(j))
#         # plt.savefig("w_grid_grid"+str(i)+"-"+str(j)+".pdf", format="pdf", bbox_inches="tight")
#         plt.show()


# for i in range(n_obs):

#     left_lim = ell(loc_obs[i,0],n_marg_grid,m_grid)
#     xNei = np.arange(left_lim,left_lim+m_grid+1) 
    
#     down_lim = ell(loc_obs[i,1],n_marg_grid,m_grid)
#     yNei = np.arange(down_lim,down_lim+m_grid+1) 

    
#     ogNei = np.array([ii*(n_marg_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
    
#     fig, ax = plt.subplots()
#     ax.set_box_aspect(1)
    
#     ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
#     ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
#     ax.scatter(loc_obs[i,0],loc_obs[i,1],color="tab:orange",s=40,marker="o")
#     ax.scatter(loc_grid[ogNei,0],loc_grid[ogNei,1],color="tab:green",s=40,marker="s")
#     # plt.title(i)
#     # plt.savefig("w_obs_grid"+str(i)+".pdf", format="pdf", bbox_inches="tight")
#     plt.show()



### denser grid + colouring


### compute all neighbors

### grid neighbors

gNei = np.zeros((n_marg_grid+1)**2,dtype=object)


### anti-neighbors

agNei = np.zeros((n_marg_grid+1)**2,dtype=object)
agInd = np.zeros((n_marg_grid+1)**2,dtype=object)

for i in range((n_marg_grid+1)**2):
    agNei[i] = []
    agInd[i] = []
    
    
### neighbors grid

for j in range(n_marg_grid+1):
    for i in range(n_marg_grid+1):
        
        xNei = np.arange(np.max([0,i-m_grid]),i+1)
        yNei = np.arange(np.max([0,j-m_grid]),j+1)
    
        gNei[j*(n_marg_grid+1)+i] = np.array([jj*(n_marg_grid+1)+ii for jj in yNei for ii in xNei if (ii != i) | (jj != j )],dtype=int)

        ind = 0 
        for jj in gNei[j*(n_marg_grid+1)+i]:
            agNei[jj].append(j*(n_marg_grid+1)+i)
            agInd[jj].append(ind)
            ind += 1
            
### neighbors observations

ogNei = np.zeros((n_obs,(m_grid+1)**2),dtype=int)

aogNei = np.zeros((n_marg_grid+1)**2,dtype=object)
aogInd = np.zeros((n_marg_grid+1)**2,dtype=object)

for i in range((n_marg_grid+1)**2):
    aogNei[i] = []
    aogInd[i] = []
    
### neihgbors obs-grid


    

for i in range(n_obs):
        
    left_lim = ell(loc_obs[i,0],n_marg_grid,m_grid)
    xNei = np.arange(left_lim,left_lim+m_grid+1) 
    
    down_lim = ell(loc_obs[i,1],n_marg_grid,m_grid)
    yNei = np.arange(down_lim,down_lim+m_grid+1) 

    
    ogNei[i] = np.array([ii*(n_marg_grid+1)+jj for ii in yNei for jj in xNei],dtype=int)
    
    ind = 0 
    for j in ogNei[i]:
        aogNei[j].append(i)
        aogInd[j].append(ind)
        ind += 1

### point

i = n_marg_grid//2
j = n_marg_grid//2

ind = j*(n_marg_grid+1) + i

fig, ax = plt.subplots()
ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
ax.scatter(loc_grid[j*(n_marg_grid+1) + i,0],loc_grid[j*(n_marg_grid+1) + i,1],color="tab:orange",s=40,marker="s")
plt.savefig("xpoint.pdf", format="pdf", bbox_inches="tight")
plt.show()

### point + nei
     

fig, ax = plt.subplots()
ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
ax.scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],color="tab:green",s=40,marker="s")
plt.savefig("xpointnei.pdf", format="pdf", bbox_inches="tight")
plt.show()


### point + nei + anei
     

fig, ax = plt.subplots()
ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
ax.scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],color="tab:green",s=40,marker="s")
ax.scatter(loc_grid[agNei[ind],0],loc_grid[agNei[ind],1],color="tab:green",s=40,marker="s")

ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
plt.savefig("xpointneianei.pdf", format="pdf", bbox_inches="tight")
plt.show()


### point + nei + anei + aneinei

fig, ax = plt.subplots()
ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
ax.scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],color="tab:green",s=40,marker="s")
ax.scatter(loc_grid[agNei[ind],0],loc_grid[agNei[ind],1],color="tab:green",s=40,marker="s")
for ii in agNei[ind]:
    ax.scatter(loc_grid[gNei[ii],0],loc_grid[gNei[ii],1],color="tab:green",s=40,marker="s")

ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
plt.savefig("xpointneianeianeinei.pdf", format="pdf", bbox_inches="tight")
plt.show()

### point + nei + anei + aneinei + aonei

fig, ax = plt.subplots()
ax.set_box_aspect(1)

# ax.scatter(loc_obs[:,0],loc_obs[:,1],color="lightgrey",s=40,marker="o")
ax.scatter(loc_grid[:,0],loc_grid[:,1],color="black",s=40,marker="s")
ax.scatter(loc_grid[gNei[ind],0],loc_grid[gNei[ind],1],color="tab:green",s=40,marker="s")
ax.scatter(loc_grid[agNei[ind],0],loc_grid[agNei[ind],1],color="tab:green",s=40,marker="s")
for ii in agNei[ind]:
    ax.scatter(loc_grid[gNei[ii],0],loc_grid[gNei[ii],1],color="tab:green",s=40,marker="s")


ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
ax.scatter(loc_obs[aogNei[ind],0],loc_obs[aogNei[ind],1],color="lightgrey",s=40,marker="o")
plt.savefig("xpointneianeianeineiaonei.pdf", format="pdf", bbox_inches="tight")
plt.show()


### colouring

fig, ax = plt.subplots()
ax.set_box_aspect(1)

for i in range(n_marg_grid+1):
    for j in range(n_marg_grid+1):
        
        ind = j*(n_marg_grid+1) + i
        
        if (i % (2*m_grid) < m_grid) & (j % (2*m_grid) < m_grid):
            ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:blue",s=40,marker="s")
        elif (i % (2*m_grid) >= m_grid) & (j % (2*m_grid) < m_grid):
             ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:orange",s=40,marker="s")
        elif (i % (2*m_grid) < m_grid) & (j % (2*m_grid) >= m_grid):
             ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:red",s=40,marker="s")
        elif (i % (2*m_grid) >= m_grid) & (j % (2*m_grid) >= m_grid):
             ax.scatter(loc_grid[ind,0],loc_grid[ind,1],color="tab:purple",s=40,marker="s")
         


plt.savefig("xcol.pdf", format="pdf", bbox_inches="tight")
plt.show()



