# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:05:29 2024

@author: alier
"""




import numpy as np
from numpy import random

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

# def matern_kernel(r, phi = 1):
    
#     return (1+np.sqrt(3)*r/phi)*np.exp(-(np.sqrt(3)*r/phi))

# def matern_kernel(r, phi = 1):
    
#     return np.exp(-r**2/2/phi**2)

def matern_kernel(r, phi = 1):
    
    return np.exp(-r/phi)

n = 1000
m = 20

locs = np.linspace(0,1,n+1)


### compute grid neighbors

Nei = np.zeros(n+1,dtype=object)

for i in range(m):
    Nei[i] = np.arange(i)
    
    
    
for j in range(m,n+1):
    Nei[j] = np.arange(j-m,j)


aNei = np.zeros(n,dtype=object)

for i in range(n):
    aNei[i] = np.array([],dtype = int)

for i in range(n):
    for j in Nei[i]:
            aNei[j] = np.append(aNei[j],i)



locst = np.transpose([locs])
Dists = distance_matrix(locst, locst)

bs = np.zeros(n+1,dtype=object)
rs = np.zeros(n+1)

w = np.ones(n+1)
# w = random.normal(size=n+1)

N = 1000



import time

#### naive implementation

# random.seed(0)

likes_n = np.zeros(N)

bs = np.zeros(n+1,dtype=object)
rs = np.zeros(n+1)



st = time.time()

for j in range(N):

    phi = (10+random.normal())*1
    
    for i in range(n+1):
    
        Cnei_inv = np.linalg.inv(matern_kernel(Dists[Nei[i]][:,Nei[i]],phi))
        
        Cnei_i = matern_kernel(Dists[Nei[i],i],phi)
        
        b = Cnei_inv @ Cnei_i
        
        bs[i] = b
        
        rs[i] = 1 - np.transpose(Cnei_i)@b
    
    
    
    

    likes_n[j] = - 1/2* np.sum([ (w[ii]  - np.inner(bs[ii],w[Nei[ii]]))**2/rs[ii] + np.log(rs[ii])  for ii in range(n+1)]) 




et = time.time()

print("Total Time:", (et-st)/60, "minutes")





#### smart implementation

# random.seed(0)

likes_s = np.zeros(N)

bs = np.zeros(m+1,dtype=object)
rs = np.zeros(m+1)



st = time.time()

for j in range(N):

    phi = (10+random.normal())*0.01
    
    for i in range(m+1):
    
        Cnei_inv = np.linalg.inv(matern_kernel(Dists[Nei[i]][:,Nei[i]],phi))
        
        Cnei_i = matern_kernel(Dists[Nei[i],i],phi)
        
        b = Cnei_inv @ Cnei_i
        
        bs[i] = b
        
        rs[i] = 1 - np.transpose(Cnei_i)@b
    
    
    
    

    likes_s[j] = - 1/2* np.sum([ (w[ii]  - np.inner(bs[ii],w[Nei[ii]]))**2/rs[ii] + np.log(rs[ii])  for ii in range(m+1)])  - 1/2* np.sum([ (w[ii]  - np.inner(bs[m],w[Nei[ii]]))**2/rs[m] + np.log(rs[m])  for ii in range(m+1,n+1)])




et = time.time()

print("Total Time:", (et-st)/60, "minutes")



#### WITH GRID LOCATIONS???


