# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:11:25 2024

@author: alier
"""

import numpy as np
from numpy import random

random.seed(0)

n = 10000
m = 101

A1 = np.zeros((n,n))


for j in range(m):
    
    
    for i in range(n-j):
        
        
        # A1[i+j,i] =  1
        A1[i+j,i] =  random.normal()


# print(A1)
# print(np.linalg.cholesky(A1@np.transpose(A1)))
# print(np.linalg.cholesky(A1@np.transpose(A1) + np.identity(n)))






from scipy.linalg import solve_banded



A2 = np.zeros((m,n))


for j in range(m):
    
    
    for i in range(n-j):
        
        
        # A2[j,i] =  1
        A2[j,i] =  random.normal()


b = np.ones(n)
# b = random.normal(size=n)

import time

REP = 10


st = time.time()
for i in range(REP):
    x = np.linalg.solve(A1, b)
et = time.time()
print("Total Time:", (et-st))

st = time.time()
for i in range(REP):
    x = solve_banded((m-1, 0), A2, b)
et = time.time()
print("Total Time:", (et-st))





B1 = A1@np.transpose(A1)

B2 = np.concatenate((np.flip(A2[1:],axis=1),A2),axis=0)


st = time.time()
for i in range(REP):
    x = np.linalg.solve(B1, b)
et = time.time()
print("Total Time:", (et-st))

st = time.time()
for i in range(REP):
    x = solve_banded((m-1, m-1), B2, b)
et = time.time()
print("Total Time:", (et-st))



