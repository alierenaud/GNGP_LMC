#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:22:36 2023

@author: homeboy
"""





import numpy as np

import matplotlib.pyplot as plt
from numpy import random

from scipy.spatial import distance_matrix

random.seed(0)

n=10
m=4

u = random.uniform(size=n)
u = np.sort(u)




for i in range(n):
    
    plt.figure(figsize=(8,1))
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    plt.scatter(u,np.zeros(n),c="black")
    plt.scatter(u[np.max([0,i-m]):i],np.zeros(np.min([i,m])),c="tab:green")
    plt.scatter(u[i],0,c="tab:orange")
    # plt.savefig("1Dnei"+str(i)+".pdf", bbox_inches='tight')
    # print(i)
    plt.show()
    
    


mat = np.zeros((n,n))



for i in range(n):   
    
    mat[i,np.max([0,i-m]):i+1] = 1
    
    
    
    
mat = (mat@np.transpose(mat)>0)*1
np.flip(mat,axis=0)



marg_grid = np.linspace(0,1,n)

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)

plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)


c = ax.pcolormesh(xv, yv, np.flip(mat,axis=0), cmap = "Blues")
# plt.savefig("mdiagStruc.pdf", bbox_inches='tight')
plt.show()



n2 = 100
u2 = random.uniform(size=(n2,2))
u2 = u2[np.argsort(u2[:,0] + u2[:,1])]



D = distance_matrix(u2, u2)

nei = np.zeros(n2,dtype=object)

m=6

for i in range(20):

    fig, ax = plt.subplots()

    ax.set_box_aspect(1)

    nei[i]= np.argsort(D[i,:i])[:np.min([m,i])]
    
    plt.figure(figsize=(4,4))
    plt.scatter(u2[:,0],u2[:,1],c="black")
    plt.scatter(u2[i,0],u2[i,1],c="tab:orange")
    plt.scatter(u2[nei[i],0],u2[nei[i],1],c="tab:green")
    plt.savefig("2Dnn"+str(i)+".pdf", bbox_inches='tight')
    # plt.show()





    