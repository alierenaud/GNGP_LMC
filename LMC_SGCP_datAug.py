# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:36:37 2023

@author: alier
"""

import numpy as np
from numpy import random

from multiLMC_generation import rmultiLMC
from LMC_multi import mult_vec

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from noisyLMC_interweaved import A_move_slice
from LMC_inference import phis_move
from LMC_mean import mu_move
from noisyLMC_inference import V_move_conj
from LMC_multi import Z_move
from LMC_multi import probs

random.seed(2)

### global parameters
lam = 500
n = random.poisson(lam)
# n = 500
p = 2


### generate random example
locs = random.uniform(0,1,(n,2))
# locs = np.transpose(np.array([np.linspace(0,1,n)]))


mu = np.array([-1,1])
A = np.array([[1.,0.5],
              [-1,0.5]])/np.sqrt(1.25)
phis = np.array([5.,25.])




Y, Z_true, V_true = rmultiLMC(A,phis,mu,locs, retZV=True) 


n_0_true = np.sum(Y==0)
n_1 = np.sum(Y!=0)

Y_0_true = Y[Y==0]
Y_1 = Y[Y!=0]

X_0_true = locs[Y==0]
X_1 = locs[Y!=0]

Z_0_true = Z_true[:,Y==0]
Z_1_true = Z_true[:,Y!=0]

V_0_true = V_true[:,Y==0]
V_1_true = V_true[:,Y!=0]


V_true = np.concatenate((V_1_true,V_0_true),axis=1)


### showcase

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(X_1[Y_1==1,0],X_1[Y_1==1,1])
ax.scatter(X_1[Y_1==2,0],X_1[Y_1==2,1])
plt.show()

### priors
sigma_A = 1.
mu_A = np.zeros((p,p))

sigma_mu = 1.
mu_mu = np.zeros((p))


min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.ones(p)
betas = np.ones(p)

a = 5
b = 0.1




### useful quantities 

X_0_current = X_0_true
n_0_current = X_0_current.shape[0]
Y_0_current = np.zeros(n_0_current,dtype=int)

D_0_current = distance_matrix(X_0_current,X_0_current)
D_01_current = distance_matrix(X_0_current,X_1)
D_1 = distance_matrix(X_1,X_1)

D_current = np.block([[D_1,np.transpose(D_01_current)],[D_01_current,D_0_current]])

### init and current state

X_current = np.concatenate((X_1,X_0_current),axis=0)
n_current = n_1 + n_0_current
Y_current = np.concatenate((Y_1,Y_0_current),axis=0)

Z_0_current = Z_0_true
Z_1_current = Z_1_true

V_0_current = V_0_true
V_1_current = V_1_true

# Z_current = Z_true
# Z_current = (mult_vec(Y,p) - 0.5)*4
Z_current = (mult_vec(Y_current,p) - 0.5)*2
# Z_current = np.concatenate((Z_1_current,Z_0_current),axis=1)

# V_current = V_true
# V_current = (mult_vec(Y,p) - 0.5)*2
V_current = random.normal(size=(p,n))
# V_current = np.concatenate((V_1_current,V_0_current),axis=1)
VmZ_current = V_current - Z_current
VmZ_inner_rows_current = np.array([ np.inner(VmZ_current[j], VmZ_current[j]) for j in range(p) ])








# mu_current = mu
mu_current = np.array([0.,0.])
Vmmu_current = V_current - np.outer(mu_current,np.ones(n))

# A_current = A
A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)
A_invVmmu_current = A_inv_current @ Vmmu_current

# phis_current = phis
phis_current = np.array([10.,10.])
Rs_current = np.array([ np.exp(-D_current*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])




taus = np.array([1.,1.])
Dm1_current = np.diag(taus)
# Dm1Z_current = Dm1_current @ Z_current


# ### proposals


phis_prop = np.ones(p)*0.5
sigma_slice = 4




### samples

N = 4000
tail = 1000

### global run containers
mu_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
phis_run = np.zeros((N,p))
lam_run = np.zeros((N,p))



### acc vector

acc_phis = np.zeros((p,N))





import time
st = time.time()


for i in range(N):
    
    
    
    V_current, Vmmu_current, VmZ_current, VmZ_inner_rows_current, A_invVmmu_current = V_move_conj(Rs_inv_current, A_inv_current, taus, Z_current, Z_current, V_current, Vmmu_current, mu_current)
        
    
    
    
    mu_current, Vmmu_current, A_invVmmu_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    A_current, A_inv_current, A_invVmmu_current = A_move_slice(A_current, A_invVmmu_current, Rs_inv_current, Vmmu_current, sigma_A, mu_A, sigma_slice)
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,D_current,A_invVmmu_current,Rs_current,Rs_inv_current)
    
    
    Z_current,VmZ_current,VmZ_inner_rows_current = Z_move(V_current,Z_current,Y_current)
    
    
    

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
        
        # probs(V_true,V_current,1000,X_current)
        
        
        
        
        
        

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
