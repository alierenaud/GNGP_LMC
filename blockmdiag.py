# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:10:04 2024

@author: alier
"""




import numpy as np
import scipy as sp
from numpy import random

import matplotlib.pyplot as plt

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

n = 10
m = 2

a = np.zeros((n,n))



for j in range(m+1):
    
    
    for i in range(n-j):
        
        
        a[i+j,i] =  1

print(a)
print(a@np.transpose(a))

A = np.zeros((n**2,n**2))



for j in range(m+1):
    
    
    for i in range(n-j):
        
        
        A[((i+j)*n):((i+j+1)*n)][:,(i*n):((i+1)*n)] =  a

print(A)  
S =  A@np.transpose(A)  
print(S)
print(np.linalg.cholesky(S))
SIG =  S + np.identity(n**2)  
print(SIG)


### alternate ordering

# order = random.permutation(n**2)


# order_m = np.zeros((m+1)**2,dtype=int)


# for i in range(m+1):
#     for j in range(m+1):
#         order_m[i*(m+1)+j] = i*n+j
        
# order_m_n = np.array([i for i in range(n**2) if i not in order_m])

# order = np.concatenate((order_m,order_m_n))

# order = np.flip(order)




# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import reverse_cuthill_mckee

# graph = csr_matrix(SIG)
# print(graph)

# order = reverse_cuthill_mckee(graph)

# SIG = SIG[order][:,order]



eps = 0.000001
structSIG = (SIG**2>eps)*1

print(structSIG)
np.savetxt('structSIG.txt', structSIG, delimiter=',')

structShow(structSIG)

    
    

ASIG = np.linalg.cholesky(SIG)
structASIG = (ASIG**2>eps)*1

print(structASIG)
np.savetxt('structASIG.txt', structASIG, delimiter=',')

structShow(structASIG)

MAX = np.max(np.sum(structASIG,axis=1))
SUM = np.sum(np.sum(structASIG,axis=1))

print("MAX is ", MAX)
print("SUM is ", SUM)



#### line of sight neighbor structure

n = 5
m = 2

Nei = np.zeros(n**2,dtype=object)

for i in range(n):
    for j in range(n):
        
        left = np.arange(i*n+np.max((0,j-m)),i*n+j)
        bottom = np.arange(np.max((0,i-m))*n+j,i*n+j,n)
        
        Nei[i*n+j] = np.concatenate((left,bottom))


from base import makeGrid

marg_grid = np.linspace(0,1,n)
grid_locs = makeGrid(marg_grid,marg_grid)

# for i in range(n**2):
    
#     fig, ax = plt.subplots()

#     ax.set_aspect(1)
#     plt.scatter(grid_locs[:,0],grid_locs[:,1], c="black")
#     plt.scatter(grid_locs[i,0],grid_locs[i,1], c="tab:orange")
#     plt.scatter(grid_locs[Nei[i],0],grid_locs[Nei[i],1], c="tab:green")
#     plt.show()


A = np.identity(n**2)


for i in range(n**2):
    for j in range(i):
        
        
        if np.isin(j, Nei[i]):
            A[i,j] = 1

eps = 0.000001


print(A)  
structShow(A)


S =  A@np.transpose(A)  
print(S)
structS = (S**2>eps)*1
structShow(structS)

AS = np.linalg.cholesky(S)
structShow(AS)


SIG =  S + np.identity(n**2)  
structSIG = (SIG**2>eps)*1
structShow(structSIG)


ASIG = np.linalg.cholesky(SIG)
structASIG = (ASIG**2>eps)*1
structShow(structASIG)










