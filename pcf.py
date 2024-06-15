#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:31:24 2023

@author: homeboy
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from LMC_generation import rLMC
from scipy.spatial import distance_matrix


### computing pcf

Ns = 10000
nds = 50
lim = 0.5

d = 0.1

p = 5

A = np.ones((p,p))*np.sqrt(1/p)
fac = np.ones((p,p))
for i in range(p):
    for j in range(i+1,p):
        fac[i,j] = -1 
A *= fac

Sigma = A@np.transpose(A)
    
phis = np.exp(np.linspace(np.log(5), np.log(25),p))
mu = np.ones(p)*-1



def pcf_val(d,A,phis,mu,Ns):


    
    locs = np.array([[0],[d]])
    
    p = A.shape[0]
    
    D = distance_matrix(locs,locs)
    
    Rs = np.array([ np.exp(-D*phis[j]) for j in range(p) ])
    Cs = np.array([np.linalg.cholesky(Rs[j]) for j in range(p)])
    Zs = np.array([np.matmul( Cs[j], random.normal(size=(2,Ns)) ) for j in range(p)])
    Vs = np.array([np.matmul( A, Zs[:,j] ) + np.outer(mu,np.ones(Ns)) for j in range(2)])
    Ys = Vs + random.normal(size=(2,p,Ns))
    return(Ys)




N = 10000
tail = 2000
jumps = 80

ds = (np.arange(nds)+1)/nds*lim

mms = np.zeros(((N-tail)//jumps,nds,2,Ns))


import time
st = time.time()

for i in range(tail,N,jumps):
    
    # print(i)
    for j in range(nds):
        y = pcf_val(ds[j],A,phis,mu,Ns)
        mms[(i-tail)//jumps,j] = (np.argmax(y,axis=1) + 1) * (np.max(y,axis=1) > 0)
          
        
        
et = time.time()
print((et-st)/60,"minutes")



### plotting

def pairs(p):
    
    prs = np.zeros((p*(p+1)//2,2))
    ind = 0
    
    for i in range(p):
        for j in range(i,p):
            prs[ind] = [i+1,j+1] 
            
            ind+=1
            
    return(prs)


prs_p = pairs(p)


pcfs = np.zeros((p*(p+1)//2,(N-tail)//jumps,nds))

st = time.time()

ind=0
for prs in prs_p:
    
    for i in range(tail,N,jumps):
        for j in range(nds):
            
            ik = mms[(i-tail)//jumps,j] == prs[0]
            il = mms[(i-tail)//jumps,j] == prs[1]
            
            ikl = np.array([ik[0] & il[1],ik[1] & il[0]])
            
            pcfs[ind,(i-tail)//jumps,j] = np.mean(ikl)/np.mean(ik)/np.mean(il)
            
    ind+=1

    
et = time.time()
print((et-st)/60,"minutes") 


for pn in range(p*(p+1)//2):

    plt.plot(ds,pcfs[pn,0])
    plt.show()



# mm = (np.argmax(y,axis=1) + 1) * (np.max(y,axis=1) > 0)


# k = 0
# l = 1

# ik1 =   mm == k+1 
# il1 =   mm == l+1
# ikl1 = np.array([ik1[0] & il1[1],ik1[1] & il1[0]])

# np.mean(ikl1)/np.mean(ik1)/np.mean(il1)

# ik = (np.argmax(y,axis=1) == k) & (y[:,k] > 0)
# il = (np.argmax(y,axis=1) == l) & (y[:,l] > 0)
# ikl = np.array([ik[0] & il[1],ik[1] & il[0]])

# np.mean(ikl)/np.mean(ik)/np.mean(il)

# def pcf(k,l,d,A,phis,mu,Ns):
    
#     pcf_k = np.zeros(Ns)
#     pcf_l = np.zeros(Ns)
    
#     pcf_kl = np.zeros(Ns)
#     pcf_kk = np.zeros(Ns)
#     pcf_ll = np.zeros(Ns)
    

    
#     for ii in range(Ns):
    
#         y = rLMC(A, phis, np.array([[0],[d]])) + np.outer(mu,np.ones(2)) + random.normal(size=(2,2))
        
        
        
#         pcf_k[ii] = (int((np.argmax(y[:,0]) == k-1) & (y[k-1,0] > 0)) )
#         pcf_l[ii] = (int((np.argmax(y[:,1]) == l-1) & (y[l-1,1] > 0) ))
        
#         # y = rLMC(A, phis, np.array([[0],[d]])) + np.outer(mu,np.ones(2)) + random.normal(size=(2,2))
        
        
#         pcf_kl[ii] = int((np.argmax(y[:,0]) == k-1) & (y[k-1,0] > 0) & (np.argmax(y[:,1]) == l-1) & (y[l-1,1] > 0) )
#         pcf_kk[ii] = int((np.argmax(y[:,0]) == k-1) & (y[k-1,0] > 0) & (np.argmax(y[:,1]) == k-1) & (y[k-1,1] > 0) )
#         pcf_ll[ii] = int((np.argmax(y[:,0]) == l-1) & (y[l-1,0] > 0) & (np.argmax(y[:,1]) == l-1) & (y[l-1,1] > 0) )
        
#     return(np.mean(pcf_kk)/np.mean(pcf_k)**2,np.mean(pcf_ll)/np.mean(pcf_l)**2,np.mean(pcf_kl)/np.mean(pcf_k)/np.mean(pcf_l))



# # mu = np.array([-1.,-1.])
# # A = np.array([[1.,1.],
# #               [1.,0.]])
# # phis = np.array([5.,25.])



# # want 100
# nds = 50
# lim = 0.5

# ds = (np.arange(nds)+1)/nds*lim

# # Ns = 100000
# # truepcf = np.zeros(nds)

# # for j in range(nds):
# #     print(j)
    
# #     truepcf[j] = pcf(2,1,ds[j],A,phis,mu,Ns)







# # plt.plot(ds,truepcf)
# # plt.show()


# A_run = np.load("A_run.npy")
# phis_run = np.load("phis_run.npy")
# mu_run = np.load("mu_run.npy")

# N = A_run.shape[0]
# tail = 2000
# # want 100000
# Ns = 100000
# # want 40
# jumps = 80

# pcfs = np.zeros(((N-tail)//jumps,nds,3))

# # pcfs[i,j] = pcf(2,1,ds[j],A_run[i],phis_run[i],mu_run[i],Ns)

# import time
# st = time.time()


# for i in range(tail,N,jumps):
#     if i % 1 == 0:
#         print(i)
#     for j in range(nds):
        
#         pcfs[(i-tail)//jumps,j] = pcf(1,2,ds[j],A_run[i],phis_run[i],mu_run[i],Ns)

# et = time.time()
# print((et-st)/60,"minutes")

# step=1
# lim=50
# mean_pcfs = np.mean(pcfs,axis=0)
# q05_pcfs = np.quantile(pcfs,0.05,axis=0)
# q95_pcfs = np.quantile(pcfs,0.95,axis=0)
# plt.plot(ds[:lim:step],mean_pcfs[:lim:step,0])
# plt.plot(ds[:lim:step],mean_pcfs[:lim:step,1])
# plt.plot(ds[:lim:step],mean_pcfs[:lim:step,2],c="grey")
# plt.fill_between(ds[:lim:step], q05_pcfs[:lim:step,0], q95_pcfs[:lim:step,0], alpha=0.5)
# plt.fill_between(ds[:lim:step], q05_pcfs[:lim:step,1], q95_pcfs[:lim:step,1], alpha=0.5)
# plt.fill_between(ds[:lim:step], q05_pcfs[:lim:step,2], q95_pcfs[:lim:step,2],color="grey", alpha=0.5)
# # plt.plot(ds,truepcf,c="grey")
# plt.title("Pair Correlation Function")
# plt.legend(["maple", "hickory","cross"], loc ="upper right") 
# plt.show()
# # plt.savefig('pcfhickmap2.pdf', bbox_inches='tight')