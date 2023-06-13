#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:54:16 2023

@author: homeboy
"""





import numpy as np
from numpy import random

from LMC_generation import rLMC

import matplotlib.pyplot as plt
from noisyLMC_interweaved import vec

from scipy.spatial import distance_matrix

import time



def likeVec(v,A,phis,D):
    
    p = A.shape[0]
    
    
    Rs = np.array([ np.exp(-D*phis[j]) for j in range(p) ])
    
    
    Sigma = np.sum([np.kron(Rs[j],np.outer(A[:,j],A[:,j])) for j in range(p) ],axis=0)
    
    num = np.exp(-1/2*np.inner(np.linalg.inv(Sigma)@v,v))
    
    den = np.linalg.det(Sigma)**(1/2)
    
    # like = num/den
    
    return(num,den)
    
def likeMat(V,A,phis,D):
    
    p = A.shape[0]
    n = D.shape[0]
    
    Rs = np.array([ np.exp(-D*phis[j]) for j in range(p) ])


    Rs_inv = np.array([ np.linalg.inv(Rs[j]) for j in range(p) ])
    
    
    A_inv = np.linalg.inv(A)
    
    A_invV = A_inv@V

    num = np.exp( -1/2*np.sum( [ np.inner(A_invV[j]@Rs_inv[j],A_invV[j]) for j in range(p)]  ) )
    
    den = np.abs(np.linalg.det(A))**n * np.prod([np.linalg.det(Rs[j]) for j in range(p)])**(1/2) 

    # like = num/den                               

    return(num,den)

stg = time.time()

random.seed(0)


### global parameters
ns = np.array([100,200,300,400])
ps = np.array([2,4,6,8,10])

### number of likelihood evaluations
N = 1000


### results container

times_vec = np.zeros(shape=(ns.shape[0],ps.shape[0]))
times_mat = np.zeros(shape=(ns.shape[0],ps.shape[0]))


for k in range(ns.shape[0]):
    for l in range(ps.shape[0]):

        locs = random.uniform(0,1,(ns[k],2))
        D = distance_matrix(locs,locs)
        
        A = random.normal(size=(ps[l],ps[l]))
        phis = random.exponential(size=ps[l])


        ### generate LMC V
        
        V = rLMC(A, phis, locs)
        v = vec(V)







        print(likeVec(v,A,phis,D))
        print(likeMat(V,A,phis,D))



        
        
        
        stvec = time.time()
        
        for i in range(N):
            likeVec(v,A,phis,D)
            
        etvec = time.time()
        
        
        
        stmat = time.time()
        
        for i in range(N):
            likeMat(V,A,phis,D)
        
        etmat = time.time()
        
        times_vec[k,l] = etvec-stvec
        times_mat[k,l] = etmat-stmat
        
        print("n =",ns[k])
        print("p =",ps[l])
        print('vec time:', (etvec-stvec)/60, 'minutes')
        print('mat time:', (etmat-stmat)/60, 'minutes')
        print('ratio vec/mat', (etvec-stvec)/(etmat-stmat))
        
np.savetxt("res_vec.csv", times_vec, delimiter=",")
np.savetxt("res_mat.csv", times_mat, delimiter=",")

etg = time.time()
print('Global time:', (etg-stg)/60, 'minutes')