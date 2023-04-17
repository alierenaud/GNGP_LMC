# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:35:11 2023

@author: alier
"""


import numpy as np
from numpy import random


from LMC_generation import rLMC

import matplotlib.pyplot as plt

random.seed(3258)
# random.seed(1684)

### global parameters
n = 1000
p = 2


### generate random example

locs = np.linspace(0, 1, n)



A = np.array([[-1.5,1.1],
              [1,2]])

# A = np.array([[-1,1],
#               [1,1]])

phis = np.array([5.,20.])




V_true = rLMC(A,phis,np.transpose(np.array([locs])))


### showcase V and Y

plot_size_scale = 1.5

plt.figure(figsize=(3*plot_size_scale,2*plot_size_scale))

plt.plot(locs,V_true[0], label = "j=1")
plt.plot(locs,V_true[1], label = "j=2")

plt.xlabel('s') 
plt.ylabel('v_j(s)') 
plt.title("LMC Realization")
plt.legend(loc="upper right")

plt.savefig('trajec.pdf', bbox_inches="tight") 
# plt.savefig('trajec2.pdf', bbox_inches="tight")   
plt.show()



### showcase crosscovariance

plt.figure(figsize=(3*plot_size_scale,2*plot_size_scale))

max_d = 1
res = 100

ds = np.linspace(0,max_d,res)

def crossCov(d,A,phis,i,j):
    return(np.sum(A[i] * A[j] * np.exp(-d*phis)))
    
cc = np.zeros(res)

i=0
j=0

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
plt.plot(ds,cc, c="tab:blue", label="i=1, j=1")

i=1
j=1

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
plt.plot(ds,cc, c="tab:orange", label="i=2, j=2")
# plt.plot(ds,cc, c="tab:orange", linestyle='dashed', label="i=2, j=2")

i=0
j=1

for r in range(res):
    cc[r] = crossCov(ds[r],A,phis,i,j)
    
plt.plot(ds,cc, c="black", label="i=1, j=2")

plt.xlabel('d') 
plt.ylabel('C_ij(d)') 
plt.title("Cross-Covariance Function")
plt.legend(loc="upper right")

plt.savefig('crosscov.pdf', bbox_inches="tight")  
# plt.savefig('crosscov2.pdf', bbox_inches="tight")  
plt.show()


cor_0_theo = crossCov(0,A,phis,0,1)/np.sqrt(crossCov(0,A,phis,1,1)*crossCov(0,A,phis,1,1))
cor_0p1_theo = crossCov(0.1,A,phis,0,1)/np.sqrt(crossCov(0,A,phis,1,1)*crossCov(0,A,phis,1,1))

# N = 10000

# results = np.zeros(N)

# import time
# st = time.time()

# for i in range(N):

#     random.seed(i)
        
#     V_true = rLMC(A,phis,np.transpose(np.array([locs])))

#     v_0 = np.sum(V_true[0]*V_true[0])/n
#     v_1 = np.sum(V_true[1]*V_true[1])/n
#     v_01 = np.sum(V_true[0]*V_true[1])/n
#     v_01_lag = (np.sum(V_true[0,100:]*V_true[1,:900]) + np.sum(V_true[0,:900]*V_true[1,100:]))/(2*(n-100))
    
#     cor_0_samp = v_01/np.sqrt(v_1*v_0)
#     cor_0p1_samp = v_01_lag/np.sqrt(v_1*v_0)
    
#     # cor_0_samp = np.corrcoef(V_true)[0,1]
#     # cor_0p1_samp = np.corrcoef(V_true[0,100:],V_true[1,:900])[0,1]
    
#     results[i] = (cor_0_samp - cor_0_theo)**2 + (cor_0p1_samp - cor_0p1_theo)**2
    
#     if i % 100 == 0:
#         print(i)

# et = time.time()
# print('Execution time:', (et-st)/60, 'minutes')

# print(np.argsort(results)[0:10])


