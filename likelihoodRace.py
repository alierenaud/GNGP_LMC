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

n = 10
m = 4

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

phi = (10+random.normal(size=N))*1

st = time.time()

for j in range(N):

    
    
    for i in range(n+1):
    
        Cnei_inv = np.linalg.inv(matern_kernel(Dists[Nei[i]][:,Nei[i]],phi[j]))
        
        Cnei_i = matern_kernel(Dists[Nei[i],i],phi[j])
        
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

    
    
    for i in range(m+1):
    
        Cnei_inv = np.linalg.inv(matern_kernel(Dists[Nei[i]][:,Nei[i]],phi[j]))
        
        Cnei_i = matern_kernel(Dists[Nei[i],i],phi[j])
        
        b = Cnei_inv @ Cnei_i
        
        bs[i] = b
        
        rs[i] = 1 - np.transpose(Cnei_i)@b
    
    
    
    

    likes_s[j] = - 1/2* np.sum([ (w[ii]  - np.inner(bs[ii],w[Nei[ii]]))**2/rs[ii] + np.log(rs[ii])  for ii in range(m+1)])  - 1/2* np.sum([ (w[ii]  - np.inner(bs[m],w[Nei[ii]]))**2/rs[m] + np.log(rs[m])  for ii in range(m+1,n+1)])




et = time.time()

print("Total Time:", (et-st)/60, "minutes")



#### WITH GRID LOCATIONS


n_obs = 10
loc_obs = random.uniform(size=n_obs)

### compute grid neighbors

gNei = np.zeros((n_obs,m),dtype=int)

for i in range(n_obs):
    
    left_nei = int(np.floor(loc_obs[i]*n))
    
    left_lim = left_nei-m/2+1
    right_lim = left_nei+m/2+1
    
    if left_lim < 0:
        gNei[i] = np.arange(0,m)
    elif right_lim > n+1:
        gNei[i] = np.arange(n+1-m,n+1)
    else:
        gNei[i] = np.arange(left_lim,right_lim)
        
        
    
for i in range(n_obs):
    
    fig, ax = plt.subplots()

    ax.set_aspect(1)
    plt.scatter(locs[:],np.ones(n+1)*0.5, c="black")
    plt.scatter(loc_obs[i],0.5, c="tab:orange")
    plt.scatter(locs[gNei[i]],np.ones(m)*0.5, c="tab:green")
    plt.show()    
    



Distm = distance_matrix(locst[0:m,:], locst[0:m,:])

gDists = np.zeros((n_obs,m))

for i in range(n_obs):
    gDists[i] = distance_matrix([[loc_obs[i]]],locst[gNei[i]])[0]

glikes_n = np.zeros(N)

gbs = np.zeros((n_obs,m),dtype=object)
grs = np.zeros(n_obs)

w_obs = np.ones(n_obs)    


st = time.time()

for j in range(N):
    
    

    for i in range(n_obs):

        Cnei_inv = np.linalg.inv(matern_kernel(Distm,phi[j]))
        
        Cnei_i = matern_kernel(gDists[i],phi[j])
        
        b = Cnei_inv @ Cnei_i
        
        gbs[i] = b
        
        grs[i] = 1 - np.transpose(Cnei_i)@b



    glikes_n[j] = - 1/2* np.sum([ (w_obs[ii]  - np.inner(gbs[ii],w[gNei[ii]]))**2/grs[ii] + np.log(grs[ii])  for ii in range(n_obs)]) 

et = time.time()

print("Total Time:", (et-st)/60, "minutes")


### smart implementation

glikes_s = np.zeros(N)

gbs = np.zeros((n_obs,m),dtype=object)
grs = np.zeros(n_obs)



st = time.time()

for j in range(N):
    
    
    
    Cnei_inv = np.linalg.inv(matern_kernel(Distm,phi[j]))

    for i in range(n_obs):

        
        
        Cnei_i = matern_kernel(gDists[i],phi[j])
        
        b = Cnei_inv @ Cnei_i
        
        gbs[i] = b
        
        grs[i] = 1 - np.transpose(Cnei_i)@b



    glikes_s[j] = - 1/2* np.sum([ (w_obs[ii]  - np.inner(gbs[ii],w[gNei[ii]]))**2/grs[ii] + np.log(grs[ii])  for ii in range(n_obs)]) 

et = time.time()

print("Total Time:", (et-st)/60, "minutes")
