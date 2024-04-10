# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:28:47 2024

@author: alier
"""


import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

def structShow(mat):

    np.flip(mat,axis=0)
    n = mat.shape[0]
    marg_grid = np.linspace(0,1,n)

    xv, yv = np.meshgrid(marg_grid, marg_grid)



    fig, ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    ax.set_box_aspect(1)
    
    # plt.tick_params(left = False, right = False , labelleft = False , 
                    # labelbottom = False, bottom = False)


    # c = ax.pcolormesh(xv, yv, np.flip(mat,axis=0), cmap = "Blues", edgecolors= "black", alpha=0.8)
    c = ax.pcolormesh(xv, yv, np.flip(mat,axis=0), cmap = "Blues", alpha=0.8)
    # plt.savefig("mdiagStruc.pdf", bbox_inches='tight')
    plt.show()


# random.seed(0)

n=100
m=8
locs = random.uniform(size=(n,2))



# locs = locs[np.argsort(locs[:,0])]
# locs = locs[np.argsort(locs[:,1])]
locs = locs[np.argsort(locs[:,0]+locs[:,1])]


Dist = distance_matrix(locs, locs)



Nei = np.zeros(n,dtype=object)



for i in range(m):
    Nei[i] = np.arange(i)
    
    
    
for j in range(m,n):
    Nei[j] = np.argsort(Dist[j,0:j])[:m]





# for i in range(n):
    
#     fig, ax = plt.subplots()

#     ax.set_aspect(1)
#     plt.scatter(locs[:,0],locs[:,1], c="black")
#     plt.scatter(locs[i,0],locs[i,1], c="tab:orange")
#     plt.scatter(locs[Nei[i],0],locs[Nei[i],1], c="tab:green")
#     plt.show()


### check on sparsity structure

A = np.identity(n)

for i in range(n):
    for j in range(n):
        
        
        if np.isin(i, Nei[j]):
            A[j,i] = 1


eps = 0.000001


structShow(A)

S = A@np.transpose(A)
structS = (S**2>eps)*1
structShow(structS)

SIG = S + np.identity(n)
structSIG = (SIG**2>eps)*1
structShow(structSIG)


# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import reverse_cuthill_mckee

# graph = csr_matrix(SIG)
# print(graph)

# order = reverse_cuthill_mckee(graph)

# SIG = SIG[order][:,order]

ASIG = np.linalg.cholesky(SIG)
structASIG = (ASIG**2>eps)*1
structShow(structASIG)

MAX = np.max(np.sum(structASIG,axis=1))
SUM = np.sum(np.sum(structASIG,axis=1))

print("MAX is ", MAX)
print("SUM is ", SUM)




# aNei = np.zeros(n,dtype=object)

# for i in range(n):
#     aNei[i] = np.array([],dtype = int)

# for i in range(n):
#     for j in Nei[i]:
#             aNei[j] = np.append(aNei[j],i)
        
    
    
# for i in range(n):
    
#     fig, ax = plt.subplots()

#     ax.set_aspect(1)
#     plt.scatter(locs[:,0],locs[:,1], c="black")
#     plt.scatter(locs[i,0],locs[i,1], c="tab:orange")
#     plt.scatter(locs[aNei[i],0],locs[aNei[i],1], c="tab:green")
#     plt.show()
    

# n_aNei = np.zeros(n)


# for i in range(n):
    
#     n_aNei[i] = aNei[i].shape[0]
    
    
    
# plt.hist(n_aNei)
# plt.show()
   
# print(np.max(n_aNei))
    


# ### Alternate ordering scheme 


# n=10
# m=2

# random.seed(0)
# locs = random.uniform(size=(n**2,2))


# locs = locs[np.argsort(locs[:,1])]



# for j in range(n):
    
#     locs[j*n:j*n+n] = locs[j*n:j*n+n][np.argsort(locs[j*n:j*n+n,0])]
    
    

# Nei = np.zeros(n**2,dtype=object)




    
    
    
# for i in np.arange(n**2):
    
#     row = i//n
#     col = i%n
    
    
#     # print(i)
#     Nei[i] = np.array([],dtype=int)
    
    

#     for j in np.arange(np.max([row-m,0]),row):
#         Nei[i] = np.append(Nei[i], np.arange(np.max([n*j+col-m,n*j]) ,n*j+col+1,dtype=int))
        
      
#     Nei[i] = np.append(Nei[i],np.arange(np.max([i-m,i-col]),i,dtype=int)) 
    


# for i in range(n**2):
    
#     fig, ax = plt.subplots()

#     ax.set_aspect(1)
#     plt.scatter(locs[:,0],locs[:,1], c="black")
#     plt.scatter(locs[i,0],locs[i,1], c="tab:orange")
#     plt.scatter(locs[Nei[i],0],locs[Nei[i],1], c="tab:green")
#     plt.show()






#### new ordering heuristic


# locs = random.uniform(size=(n,2))


# Dist = distance_matrix(locs, locs)

# order = np.zeros(n,dtype=int)
# ind = np.arange(n)

# order[0] = random.randint(0,n)
# ind = np.delete(ind, order[0])
# Dist = np.delete(Dist, order[0], axis=1)

# for i in range(1,n):
#     newi = np.argmax(Dist[order[i-1]])
#     order[i] = ind[newi]
#     ind = np.delete(ind, newi)
#     Dist = np.delete(Dist, newi, axis=1)
#     if i % 100 == 0:
#         print(i)


# locs = locs[order]

# Nei = np.zeros(n,dtype=object)



# for i in range(m):
#     Nei[i] = np.arange(i)
    
    
# for i in range(m,n):
#     Nei[i] = np.arange(i-m,i)
    

# for i in range(n):
    
    
#     if i % 100 == 0:
#         fig, ax = plt.subplots()
    
#         ax.set_aspect(1)
#         plt.ylim(0,1)
#         plt.xlim(0,1)
#         plt.scatter(locs[:,0],locs[:,1], c="black")
#         plt.scatter(locs[i,0],locs[i,1], c="tab:orange")
#         plt.scatter(locs[Nei[i],0],locs[Nei[i],1], c="tab:green")
#         plt.title(str(i))
#         plt.show()