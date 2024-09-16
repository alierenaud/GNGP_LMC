# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:40:50 2023

@author: alier
"""




import numpy as np
from numpy import random


import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from LMC_pred_rjmcmc import A_move_slice_mask
from LMC_pred_rjmcmc import A_rjmcmc
from LMC_pred_rjmcmc import V_pred
from LMC_pred_rjmcmc import pairs


from noisyLMC_inference import V_move_conj_scale_mis, taus_move_mis

from LMC_inference import phis_move
from LMC_mean import mu_move

from numpy import genfromtxt

import time

random.seed(0)

# air_data = np.loadtxt("data/april2602xyDataMiss.txt",delimiter=",",skiprows=1,usecols = (3,4,5,6,9,10))

air_data = genfromtxt("data/april2602xyDataMiss.txt", delimiter=",",skip_header=1,usecols = (3,4,5,6,9,10))



loc_obs = air_data[:,4:6]
conc_obs = air_data[:,:4]

Mis_obs =  np.transpose(1 - np.isnan(conc_obs))
conc_obs[np.isnan(conc_obs)] = 1
Y_obs = np.transpose(np.log(conc_obs))

ns = np.sum(Mis_obs,axis=1)

###
np.mean(Y_obs,axis=1)
np.cov(Y_obs)

n_obs = Y_obs.shape[1]
p = Y_obs.shape[0]

#### centered


# Y_obs[0] = (Y_obs[0] - np.mean(Y_obs[0])) / np.std(Y_obs[0])
# Y_obs[1] = (Y_obs[1] - np.mean(Y_obs[1])) / np.std(Y_obs[1])
# Y_obs[2] = (Y_obs[2] - np.mean(Y_obs[2])) / np.std(Y_obs[2])
# Y_obs[3] = (Y_obs[3] - np.mean(Y_obs[3])) / np.std(Y_obs[3])





### markov chain + tail length
N = 200000
tail = 50000

### showcase locations

fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black")
plt.title("Monitoring Sites")
# plt.savefig("monSites.pdf", format="pdf", bbox_inches="tight")
plt.show()


### priors
sigma_A = 1.
mu_A = np.zeros((p,p))
# mu_A = np.array([[0.,0.,0.],
#                  [0.,0.,0.],
#                  [0.,0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi

alphas = np.ones(p)
betas = np.ones(p)

### RJMCMC

prob_one = 1/p

## tau

a = 1
b = 1

### mu 

mu_mu = np.zeros(p)
sigma_mu = 10

### proposals


phis_prop = np.ones(p)*3
sigma_slice = 10


def ins_prob(n_ones,p):
    
    if n_ones == p**2:
        return(0)
    elif n_ones == p:
        return(1)
    else:
        return(0.5)

n_jumps = p

### global run containers
mu_run = np.zeros((N,p))
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))

n_comps_run = np.zeros((N))


### acc vector
acc_phis = np.zeros((p,N))


### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)


### current values

n_ones_current = p**2
A_mask_current = np.ones((p,p))
        
A_ones_ind_current = pairs(p)
A_zeros_ind_current = []

### init and current state
phis_current = np.repeat(10.,p)
Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


V_current = random.normal(size=(p,n_obs))*1
VmY_current = V_current - Y_obs
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j]*Mis_obs[j], VmY_current[j]) for j in range(p) ])

mu_current = np.zeros(p)
Vmmu1_current = V_current




A_current = random.normal(size=(p,p))
A_inv_current = np.linalg.inv(A_current)
A_invVmmu1_current = A_inv_current @ Vmmu1_current



taus_current = np.ones(p)
Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y_obs

st = time.time()

for i in range(N):
    
    
    V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current = V_move_conj_scale_mis(Mis_obs,Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y_obs, V_current, Vmmu1_current, A_invVmmu1_current, mu_current)
        
    
    
    
    mu_current, Vmmu1_current, A_invVmmu1_current = mu_move(A_inv_current,Rs_inv_current,V_current,sigma_mu,mu_mu)

    
    
    A_current, A_inv_current, A_invVmmu1_current = A_move_slice_mask(A_current, A_invVmmu1_current, A_mask_current, Rs_inv_current, Vmmu1_current, sigma_A, mu_A, sigma_slice)
    
    
    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs,A_invVmmu1_current,Rs_current,Rs_inv_current)
    
    taus_current, Dm1_current, Dm1Y_current = taus_move_mis(taus_current,VmY_inner_rows_current,Y_obs,a,b,ns)

    
    
        
    # A_current, A_inv_current, A_invVmmu1_current, n_ones_current, A_mask_current, A_ones_ind_current, A_zeros_ind_current = A_rjmcmc(Rs_inv_current, Vmmu1_current, A_current, A_inv_current, A_invVmmu1_current, A_zeros_ind_current, A_ones_ind_current, A_mask_current, n_ones_current, prob_one, mu_A, sigma_A, n_jumps)
    

    mu_run[i] = mu_current
    V_run[i] = V_current
    taus_run[i] = taus_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    n_comps_run[i] = n_ones_current
    
    if i % 1000 == 0:
        print(i)

et = time.time()

print("Time Elapsed", (et-st)/60, "min")

print("Accept Rate for phis",np.mean(acc_phis,axis=1))


plt.plot(n_comps_run[tail:])
plt.show()

Sigmas0 = np.array([A_run[i]@np.transpose(A_run[i]) for i in range(tail,N)])
Corrs0 = np.array([np.diag(1/np.sqrt(np.diag(Sigmas0[i])))@Sigmas0[i]@np.diag(1/np.sqrt(np.diag(Sigmas0[i]))) for i in range(N-tail)])


Sigma_med = np.median(Sigmas0,axis=0)
Sigma_05 = np.quantile(Sigmas0,0.05,axis=0)
Sigma_95 = np.quantile(Sigmas0,0.95,axis=0)

Corr_med = np.median(Corrs0,axis=0)
Corr_05 = np.quantile(Corrs0,0.05,axis=0)
Corr_95 = np.quantile(Corrs0,0.95,axis=0)


### number of components



unique, counts = np.unique(n_comps_run[tail:N], return_counts=True)
plt.figure(figsize=(5,3))
plt.bar(unique,counts,alpha=0.8)
# plt.title("Number of Non-Zero Elements")
# plt.savefig("numComp.pdf", format="pdf", bbox_inches="tight")
plt.show()

mu_med = np.median(mu_run,axis=0)






prob_ind = np.mean(Sigmas0==0,axis=0)


likes = np.array([np.sqrt(np.diag(taus_run[j])/2/np.pi)@np.exp(-1/2*np.diag(taus_run[j])@(Y_obs-V_run[j])**2) for j in range(tail,N)])

waic = - np.mean(np.log(np.mean(likes,axis=0))) + np.mean(np.var(np.log(likes),axis=0))


# reso = 200
# dist_cov = np.linspace(0,0.4,reso)

# i = 0
# j = 0

# C_00 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 0
# j = 1

# C_01 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 0
# j = 2

# C_02 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 0
# j = 3

# C_03 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 1
# j = 1

# C_11 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 1
# j = 2

# C_12 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 1
# j = 3

# C_13 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 2
# j = 2

# C_22 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 2
# j = 3

# C_23 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)

# i = 3
# j = 3

# C_33 = np.array([[A_run[k,i]*A_run[k,j]*np.exp(-phis_run[k] * d) for k in range(tail,N)] for d in dist_cov])

# print(i,j)


# C_00 = np.sum(C_00,axis=2)
# C_01 = np.sum(C_01,axis=2)
# C_02 = np.sum(C_02,axis=2)
# C_03 = np.sum(C_03,axis=2)
# C_11 = np.sum(C_11,axis=2)
# C_12 = np.sum(C_12,axis=2)
# C_13 = np.sum(C_13,axis=2)
# C_22 = np.sum(C_22,axis=2)
# C_23 = np.sum(C_23,axis=2)
# C_33 = np.sum(C_33,axis=2)


# # np.save("C_00_air.npy",C_00)
# # np.save("C_01_air.npy",C_01)
# # np.save("C_02_air.npy",C_02)
# # np.save("C_03_air.npy",C_03)
# # np.save("C_11_air.npy",C_11)
# # np.save("C_12_air.npy",C_12)
# # np.save("C_13_air.npy",C_13)
# # np.save("C_22_air.npy",C_22)
# # np.save("C_23_air.npy",C_23)
# # np.save("C_33_air.npy",C_33)

# # C_00 = np.load("C_00_air.npy")
# # C_01 = np.load("C_01_air.npy")
# # C_02 = np.load("C_02_air.npy")
# # C_03 = np.load("C_03_air.npy")
# # C_11 = np.load("C_11_air.npy")
# # C_12 = np.load("C_12_air.npy")
# # C_13 = np.load("C_13_air.npy")
# # C_22 = np.load("C_22_air.npy")
# # C_23 = np.load("C_23_air.npy")
# # C_33 = np.load("C_33_air.npy")

# C_00_med = np.median(C_00,axis=1)
# C_00_05 = np.quantile(C_00,0.05,axis=1)
# C_00_95 = np.quantile(C_00,0.95,axis=1)

# C_01_med = np.median(C_01,axis=1)
# C_01_05 = np.quantile(C_01,0.05,axis=1)
# C_01_95 = np.quantile(C_01,0.95,axis=1)

# C_02_med = np.median(C_02,axis=1)
# C_02_05 = np.quantile(C_02,0.05,axis=1)
# C_02_95 = np.quantile(C_02,0.95,axis=1)

# C_03_med = np.median(C_03,axis=1)
# C_03_05 = np.quantile(C_03,0.05,axis=1)
# C_03_95 = np.quantile(C_03,0.95,axis=1)

# C_11_med = np.median(C_11,axis=1)
# C_11_05 = np.quantile(C_11,0.05,axis=1)
# C_11_95 = np.quantile(C_11,0.95,axis=1)

# C_12_med = np.median(C_12,axis=1)
# C_12_05 = np.quantile(C_12,0.05,axis=1)
# C_12_95 = np.quantile(C_12,0.95,axis=1)

# C_13_med = np.median(C_13,axis=1)
# C_13_05 = np.quantile(C_13,0.05,axis=1)
# C_13_95 = np.quantile(C_13,0.95,axis=1)

# C_22_med = np.median(C_22,axis=1)
# C_22_05 = np.quantile(C_22,0.05,axis=1)
# C_22_95 = np.quantile(C_22,0.95,axis=1)

# C_23_med = np.median(C_23,axis=1)
# C_23_05 = np.quantile(C_23,0.05,axis=1)
# C_23_95 = np.quantile(C_23,0.95,axis=1)

# C_33_med = np.median(C_33,axis=1)
# C_33_05 = np.quantile(C_33,0.05,axis=1)
# C_33_95 = np.quantile(C_33,0.95,axis=1)



# plt.plot(dist_cov,C_00_med)
# plt.fill_between(dist_cov,C_00_05,C_00_95,alpha=0.5)
# plt.show()

# plt.plot(dist_cov,C_11_med)
# plt.fill_between(dist_cov,C_11_05,C_11_95,alpha=0.5)
# plt.show()

# plt.plot(dist_cov,C_22_med)
# plt.fill_between(dist_cov,C_22_05,C_22_95,alpha=0.5)
# plt.show()

# plt.plot(dist_cov,C_33_med)
# plt.fill_between(dist_cov,C_33_05,C_33_95,alpha=0.5)
# plt.show()


# plt.plot(dist_cov,C_01_med,c="grey")
# plt.fill_between(dist_cov,C_01_05,C_01_95,alpha=0.5,color="grey")
# plt.show()

# plt.plot(dist_cov,C_02_med,c="grey")
# plt.fill_between(dist_cov,C_02_05,C_02_95,alpha=0.5,color="grey")
# plt.show()

# plt.plot(dist_cov,C_03_med,c="grey")
# plt.fill_between(dist_cov,C_03_05,C_03_95,alpha=0.5,color="grey")
# plt.show()

# plt.plot(dist_cov,C_12_med,c="grey")
# plt.fill_between(dist_cov,C_12_05,C_12_95,alpha=0.5,color="grey")
# plt.show()

# plt.plot(dist_cov,C_13_med,c="grey")
# plt.fill_between(dist_cov,C_13_05,C_13_95,alpha=0.5,color="grey")
# plt.show()

# plt.plot(dist_cov,C_23_med,c="grey")
# plt.fill_between(dist_cov,C_23_05,C_23_95,alpha=0.5,color="grey")
# plt.show()


# fig, axs = plt.subplots(3, 2, figsize=(9,11), layout='constrained')


# axs[0,0].plot(dist_cov,C_00_med,label="C_11")
# axs[0,0].plot(dist_cov,C_11_med,label="C_22")
# axs[0,0].plot(dist_cov,C_01_med,label="C_12")
# axs[0,0].set_title("CO (1) and NO (2)")
# # axs[0,0].set_xlabel("Distance")
# # axs[0,0].set_ylabel("Covariance")
# axs[0,0].legend()
# # plt.show()

# axs[0,1].plot(dist_cov,C_00_med,label="C_11")
# axs[0,1].plot(dist_cov,C_22_med,label="C_33")
# axs[0,1].plot(dist_cov,C_02_med,label="C_13")
# axs[0,1].set_title("CO (1) and NO_2 (3)")
# # axs[0,1].set_xlabel("Distance")
# # axs[0,1].set_ylabel("Covariance")
# axs[0,1].legend()
# # plt.show()


# axs[1,0].plot(dist_cov,C_00_med,label="C_11")
# axs[1,0].plot(dist_cov,C_33_med,label="C_44")
# axs[1,0].plot(dist_cov,C_03_med,label="C_14")
# axs[1,0].set_title("CO (1) and O_3 (4)")
# # axs[1,0].set_xlabel("Distance")
# # axs[1,0].set_ylabel("Covariance")
# axs[1,0].legend()
# # plt.show()

# axs[1,1].plot(dist_cov,C_11_med,label="C_22")
# axs[1,1].plot(dist_cov,C_22_med,label="C_33")
# axs[1,1].plot(dist_cov,C_12_med,label="C_23")
# axs[1,1].set_title("NO (2) and NO_2 (3)")
# # axs[1,1].set_xlabel("Distance")
# # axs[1,1].set_ylabel("Covariance")
# axs[1,1].legend()
# # plt.show()

# axs[2,0].plot(dist_cov,C_11_med,label="C_22")
# axs[2,0].plot(dist_cov,C_33_med,label="C_44")
# axs[2,0].plot(dist_cov,C_13_med,label="C_24")
# axs[2,0].set_title("NO (2) and O_3 (4)")
# # axs[2,0].set_xlabel("Distance")
# # axs[2,0].set_ylabel("Covariance")
# axs[2,0].legend()
# # plt.show()

# axs[2,1].plot(dist_cov,C_22_med,label="C_33")
# axs[2,1].plot(dist_cov,C_33_med,label="C_44")
# axs[2,1].plot(dist_cov,C_23_med,label="C_34")
# axs[2,1].set_title("NO_2 (3) and O_3 (4)")
# # axs[2,1].set_xlabel("Distance")
# # axs[2,1].set_ylabel("Covariance")
# axs[2,1].legend()



# fig.supxlabel('Distance')
# fig.supylabel('Covariance')

# # plt.savefig("airCovFunc.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# plt.plot(Sigmas0[:,0,0])
# plt.show()
# plt.plot(Sigmas0[:,0,1])
# plt.show()
# plt.plot(Sigmas0[:,0,2])
# plt.show()
# plt.plot(Sigmas0[:,0,3])
# plt.show()
# plt.plot(Sigmas0[:,1,1])
# plt.show()
# plt.plot(Sigmas0[:,1,2])
# plt.show()
# plt.plot(Sigmas0[:,1,3])
# plt.show()
# plt.plot(Sigmas0[:,2,2])
# plt.show()
# plt.plot(Sigmas0[:,2,3])
# plt.show()
# plt.plot(Sigmas0[:,3,3])
# plt.show()


# plt.plot(mu_run[tail:,0])
# plt.show()
# plt.plot(mu_run[tail:,1])
# plt.show()
# plt.plot(mu_run[tail:,2])
# plt.show()
# plt.plot(mu_run[tail:,3])
# plt.show()

# plt.plot(phis_run[tail:,0])
# plt.show()
# plt.plot(phis_run[tail:,1])
# plt.show()
# plt.plot(phis_run[tail:,2])
# plt.show()
# plt.plot(phis_run[tail:,3])
# plt.show()


# plt.plot(taus_run[tail:,0])
# plt.show()
# plt.plot(taus_run[tail:,1])
# plt.show()
# plt.plot(taus_run[tail:,2])
# plt.show()
# plt.plot(taus_run[tail:,3])
# plt.show()


