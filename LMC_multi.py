

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:12:40 2023

@author: alier
"""

import numpy as np
from numpy import random

from multiLMC_generation import rmultiLMC
from multiLMC_generation import mult

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.stats import truncnorm

from noisyLMC_interweaved import A_move_slice
from LMC_inference import phis_move
from LMC_mean import mu_move
from noisyLMC_inference import V_move_conj



def Z_move(V_current,Z_current,Y):
    
    p = V_current.shape[0]
    n = V_current.shape[1]
    
    for ii in range(n):
        
            if Y[ii] == 0:
                for jj in range(p):
                    Z_current[jj,ii] = truncnorm.rvs(a=-np.inf,b=-V_current[jj,ii],loc=V_current[jj,ii])
                    
            else:
                ### compute max Z_current[:,i]
                
                mini = np.max(np.delete(Z_current[:,ii], Y[ii]-1))
                
                if mini < 0:
                    mini = 0
                
                for jj in range(p):
                    if jj == Y[ii]-1:
                        Z_current[jj,ii] = truncnorm.rvs(a=mini-V_current[jj,ii],b=np.inf,loc=V_current[jj,ii])
                    
                    else:
                        Z_current[jj,ii] = truncnorm.rvs(a=-np.inf,b=V_current[Y[ii]-1,ii]-V_current[jj,ii],loc=V_current[jj,ii])
    
    VmZ_current = V_current - Z_current
    VmZ_inner_rows_current = np.array([ np.inner(VmZ_current[j], VmZ_current[j]) for j in range(p) ])
    
    return(Z_current,VmZ_current,VmZ_inner_rows_current)


def mult_vec(Y,p):
    
    n = Y.shape[0]
    Y_vec = np.zeros((p,n))
    
    for i in range(n):
        if Y[i] != 0:
            Y_vec[Y[i]-1,i] = 1
    return(Y_vec)    


def probs(V_true,V_current,bigN,locs):
    
    order = np.argsort(locs[:,0])
    
    p = V_true.shape[0]
    n = V_true.shape[1]
    
    Z_exs_infV = random.normal(size=(bigN,p,n)) + V_current
    Z_exs_trueV = random.normal(size=(bigN,p,n)) + V_true
    
    
    
    
    Y_exs_infV = np.zeros((bigN,n),dtype=int)
    Y_exs_trueV = np.zeros((bigN,n),dtype=int)
    
    for ns in range(bigN):
        for i in range(n):
            Y_exs_infV[ns,i] = mult(Z_exs_infV[ns,:,i])
            Y_exs_trueV[ns,i] = mult(Z_exs_trueV[ns,:,i])
    
    
    
    Y_vec_exs_infV=np.array([mult_vec(Y_exs_infV[ns],p) for ns in range(bigN)])
    Y_vec_exs_trueV=np.array([mult_vec(Y_exs_trueV[ns],p) for ns in range(bigN)])
    
    probs_infV = np.mean(Y_vec_exs_infV,axis=0)
    probs_trueV = np.mean(Y_vec_exs_trueV,axis=0)
    
    plt.plot(locs[order,0],probs_infV[0,order],c="tab:blue")
    plt.plot(locs[order,0],probs_infV[1,order],c="tab:orange")
    plt.plot(locs[order,0],probs_trueV[0,order],c="tab:blue", alpha=0.5)
    plt.plot(locs[order,0],probs_trueV[1,order],c="tab:orange", alpha=0.5)
    plt.show()

random.seed(2)


### global parameters
n = 500
p = 2


### generate random example
# locs = random.uniform(0,1,(n,2))
locs = np.transpose(np.array([np.linspace(0,1,n)]))


mu = np.array([0,0])
A = np.array([[1.,0.5],
              [-1,0.5]])/np.sqrt(1.25)
phis = np.array([5.,25.])



Y, Z_true, V_true = rmultiLMC(A,phis,mu,locs, retZV=True) 


## showcase RFs

plt.plot(locs[:,0],V_true[0])
plt.plot(locs[:,0],V_true[1])
plt.show()


### priors
sigma_A = 1.
mu_A = np.zeros((p,p))


min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.ones(p)
betas = np.ones(p)

sigma_mu = 1.
mu_mu = np.zeros((p))




### useful quantities 

D = distance_matrix(locs,locs)


### init and current state

# Z_current = Z_true
# Z_current = (mult_vec(Y,p) - 0.5)*4
Z_current = (mult_vec(Y,p) - 0.5)*2

# V_current = V_true
# V_current = (mult_vec(Y,p) - 0.5)*2
V_current = random.normal(size=(p,n))
VmZ_current = V_current - Z_current
VmZ_inner_rows_current = np.array([ np.inner(VmZ_current[j], VmZ_current[j]) for j in range(p) ])

# mu_current = mu
mu_current = np.array([0.,0.])
Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))

# A_current = A
A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current

# phis_current = phis
phis_current = np.array([10.,10.])
Rs_current = np.array([ np.exp(-D*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])




taus = np.array([1.,1.])
Dm1_current = np.diag(taus)
# Dm1Z_current = Dm1_current @ Z_current




### proposals


phis_prop = np.ones(p)*0.5
sigma_slice = 4




### samples

N = 2000
tail = 1000

### global run containers
mu_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
phis_run = np.zeros((N,p))




### acc vector

acc_phis = np.zeros((p,N))


import time
st = time.time()


for i in range(N):
    
    
    
    V_current, Vmmu1_current, VmZ_current, VmZ_inner_rows_current, A_invVmmu1_current = V_move_conj(Rs_inv_current, A_inv_current, taus, Z_current, Z_current, V_current, Vmmu1_current, mu_current)
        
    
    
    
    mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    A_current, A_inv_current, A_invVmmu1_current = A_move_slice(A_current, A_invVmmu1_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,D,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    
    Z_current,VmZ_current,VmZ_inner_rows_current = Z_move(V_current,Z_current,Y)
    
    
    

    mu_run[i] = mu_current
    A_run[i] = A_current
    phis_run[i] =  phis_current
    
    
    
    if i % 100 == 0:
        print(i)
        ## showcase RFs

        # plt.plot(locs[:,0],V_current[0],c="tab:blue")
        # plt.plot(locs[:,0],V_current[1],c="tab:orange")
        # plt.plot(locs[:,0],V_true[0],c="tab:blue", alpha=0.5)
        # plt.plot(locs[:,0],V_true[1],c="tab:orange", alpha=0.5)
        # plt.show()
        
        # diagnostic using probabilities
        
        probs(V_true,V_current,1000,locs)
        
        
        

et = time.time()
print('Execution time:', (et-st)/60, 'minutes')



print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))


plt.plot(phis_run[:,0])
plt.plot(phis_run[:,1])
plt.show()


plt.plot(mu_run[:,0])
plt.plot(mu_run[:,1])
plt.show()


plt.plot(A_run[:,0,0])
plt.plot(A_run[:,0,1])
plt.plot(A_run[:,1,0])
plt.plot(A_run[:,1,1])
plt.show()



print("Posterior Marginal Variance", np.mean([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)],axis=0))
print("True Marginal Variance", A@np.transpose(A))



