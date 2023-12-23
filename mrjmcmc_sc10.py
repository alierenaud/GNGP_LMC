# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:03:46 2023

@author: alier
"""


import numpy as np
from numpy import random
from scipy.stats import beta
from scipy.linalg import sqrtm

from noisyLMC_generation import rNLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

from LMC_pred_rjmcmc import A_move_slice_mask
from LMC_pred_rjmcmc import A_rjmcmc
from LMC_pred_rjmcmc import V_pred
from LMC_pred_rjmcmc import pairs


from noisyLMC_interweaved import A_move_slice
from noisyLMC_interweaved import makeGrid
from noisyLMC_interweaved import vec_inv

from noisyLMC_inference import V_move_conj, taus_move

from LMC_inference import phis_move

import time

def WassDist(A,B):
    return(np.trace( A + B - 2*sqrtm(sqrtm(A)@B@sqrtm(A))))

random.seed(0)

### number of points 
n_obs=100
n_grid=10  ### 2D Grid

### repetitions per category
reps = 100

### markov chain + tail length
N = 2000
tail = 1000

### generate uniform locations

conc = 2
# loc_obs = random.uniform(0,1,(n_obs,2))
loc_obs = beta.rvs(conc, conc, size=(n_obs,2))
### grid locations
loc_grid = makeGrid(n_grid)
### all locations
locs = np.concatenate((loc_obs,loc_grid), axis=0)

### showcase locations

fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

ax.scatter(loc_obs[:,0],loc_obs[:,1],color="black")
plt.show()



### 5D examples
p = 6



### Full

A2 = np.ones((p,p))*np.sqrt(1/p)
# A2 *= np.array([[1,-1,-1,-1,-1],
#                 [1,1,-1,-1,-1],
#                 [1,1,1,-1,-1],
#                 [1,1,1,1,-1],
#                 [1,1,1,1,1]])


fac = np.ones((p,p))



for i in range(p):
    for j in range(i+1,p):
        fac[i,j] = -1 

A2 *= fac


### Triangular

A3 = np.zeros((p,p))

for i in range(p):
    for j in range(i+1):
        A3[i,j] = (-1)**(i)*1/np.sqrt(i+1)


### Diagonal


A4 = np.identity(p)


phis = np.exp(np.linspace(np.log(5), np.log(25),p))
noise_sd = 0.5
taus_sqrt_inv = np.ones(p)*noise_sd



As = np.array([A2,A3,A4])
Sigmas = np.array([A2@np.transpose(A2),A3@np.transpose(A3),A4@np.transpose(A4)])
D0p1 = np.diag(np.exp(-phis*0.1))
Sigmas0p1 = np.array([A2@D0p1@np.transpose(A2),A3@D0p1@np.transpose(A3),A4@D0p1@np.transpose(A4)])
n_exes = As.shape[0]



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

### proposals


phis_prop = np.ones(p)*1
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
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
V_run = np.zeros((N,p,n_obs))
A_run = np.zeros((N,p,p))
V_grid_run = np.zeros((N,p,n_grid**2))
n_comps_run = np.zeros((N))


### acc vector
acc_phis = np.zeros((p,N))

### useful quantities 

### distances

Dists_obs = distance_matrix(loc_obs,loc_obs)
Dists_obs_grid = distance_matrix(loc_obs,loc_grid)
Dists_grid = distance_matrix(loc_grid,loc_grid)


#### container of pred errors

MSES = np.zeros((n_exes,2,reps,N-tail,p,n_grid**2))
n_comps = np.zeros((n_exes,reps,N-tail))
As_exp = np.zeros((n_exes,2,reps,N-tail,p,p))
Sigmas_exp = np.zeros((n_exes,2,reps,N-tail,p,p))
Sigmas0p1_exp = np.zeros((n_exes,2,reps,N-tail,p,p))
Wnorms = np.zeros((n_exes,2,reps,N-tail))
Wnorms0p1 = np.zeros((n_exes,2,reps,N-tail))
times_all = np.zeros((n_exes,2,reps))
waic = np.zeros((n_exes,2,reps))

STG = time.time()

for ex in range(n_exes):
    for rep in range(reps):
        
        

        ### generate rfs
        
        Y_true, V_true = rNLMC(As[ex],phis,taus_sqrt_inv,locs, retV=True)
        
        Y_obs = Y_true[:,:n_obs]
        V_obs = V_true[:,:n_obs]
        V_grid = V_true[:,n_obs:]
        
        
        ### current values
        
        n_ones_current = np.sum(As[ex]!=0)
        A_mask_current = (As[ex]!=0)*1
                
        
        
        A_ones_ind_current = [(np.where(As[ex]!=0)[0][j],np.where(As[ex]!=0)[1][j]) for j in range(np.where(As[ex]!=0)[0].shape[0])]
        A_zeros_ind_current = [(np.where(As[ex]==0)[0][j],np.where(As[ex]==0)[1][j]) for j in range(np.where(As[ex]==0)[0].shape[0])]
        
        ### init and current state
        phis_current = np.copy(phis)
        Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        
        V_current = np.copy(V_obs)
        VmY_current = V_current - Y_obs
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        mu_current = np.zeros(p)
        Vmmu1_current = V_current
        
        
        
        
        A_current = np.copy(As[ex])
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/taus_sqrt_inv**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y_obs
        
        st = time.time()
        
        for i in range(N):
            
            
            V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current, Vmmu1_current, mu_current)
          
            
                               
            
                
            A_current, A_inv_current, A_invV_current = A_move_slice_mask(A_current, A_invV_current, A_mask_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
            
                
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)
    
            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)
    
            
            
                
            A_current, A_inv_current, A_invV_current, n_ones_current, A_mask_current, A_ones_ind_current, A_zeros_ind_current = A_rjmcmc(Rs_inv_current, V_current, A_current, A_inv_current, A_invV_current, A_zeros_ind_current, A_ones_ind_current, A_mask_current, n_ones_current, prob_one, mu_A, sigma_A, n_jumps)
            
            ### make pred cond on current phis, A
    
            V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invV_current, mu_current, n_grid**2)
            
            ###
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            V_grid_run[i] = V_grid_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            n_comps_run[i] = n_ones_current
            
            if i % 100 == 0:
                print(i)

        et = time.time()

        print("Time Elapsed", (et-st)/60, "min")
        print("RJMCMC", ex, rep)
        
        print("Accept Rate for phis",np.mean(acc_phis,axis=1))
        
        
        
        
        times_all[ex,0,rep] = (et-st)
        MSES[ex,0,rep] = np.array([(V_grid_run[j] - V_grid)**2 for j in range(tail,N)])
        Wnorms[ex,0,rep] = np.array([ WassDist(A_run[j]@np.transpose(A_run[j]),Sigmas[ex]) for j in range(tail,N)])
        Wnorms0p1[ex,0,rep] = [ WassDist(A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]),Sigmas0p1[ex]) for j in range(tail,N)]
        # fnorms0p1[ex,0,rep] = [ np.sum((A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]) - Sigmas0p1[ex])**2) for j in range(tail,N)]
        n_comps[ex,rep] = n_comps_run[tail:N]
        As_exp[ex,0,rep] = A_run[tail:N]
        Sigmas_exp[ex,0,rep] = [A_run[j]@np.transpose(A_run[j]) for j in range(tail,N)]
        
        Sigmas0p1_exp[ex,0,rep] = [A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]) for j in range(tail,N)]
        
        likes = np.array([np.sqrt(np.diag(taus_run[j])/2/np.pi)@np.exp(-1/2*np.diag(taus_run[j])@(Y_obs-V_run[j])**2) for j in range(tail,N)])
        
        waic[ex,0,rep] = - np.mean(np.log(np.mean(likes,axis=0))) + np.mean(np.var(np.log(likes),axis=0))
        
        ### init and current state
        phis_current = np.copy(phis)
        Rs_current = np.array([ np.exp(-Dists_obs*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        
        V_current = np.copy(V_obs)
        VmY_current = V_current - Y_obs
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        mu_current = np.zeros(p)
        Vmmu1_current = V_current
        
        
        
        
        A_current = np.copy(As[ex])
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/taus_sqrt_inv**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y_obs
        
        st = time.time()

        for i in range(N):
            
            
            V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y_obs, V_current, Vmmu1_current, mu_current)
          
            
            A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A, sigma_slice)
            
                
                
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists_obs,A_invV_current,Rs_current,Rs_inv_current)

            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y_obs,a,b,n_obs)

            
            
            ### make pred cond on current phis, A

            V_grid_current = V_pred(Dists_grid, Dists_obs_grid, phis_current, Rs_inv_current, A_current, A_invV_current, mu_current, n_grid**2)
            
            ###
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            V_grid_run[i] = V_grid_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            
            if i % 100 == 0:
                print(i)

        et = time.time()
        
        print("Time Elapsed", (et-st)/60, "min")
        print("Standard", ex, rep)
        
        print("Accept Rate for phis",np.mean(acc_phis,axis=1))
        
        times_all[ex,1,rep] = (et-st)
        MSES[ex,1,rep] = [(V_grid_run[j] - V_grid)**2 for j in range(tail,N)]
        Wnorms[ex,1,rep] = [ WassDist(A_run[j]@np.transpose(A_run[j]),Sigmas[ex]) for j in range(tail,N)]
        Wnorms0p1[ex,1,rep] = [ WassDist(A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]),Sigmas0p1[ex]) for j in range(tail,N)]
        # fnorms0p1[ex,1,rep] = [ np.sum((A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]) - Sigmas0p1[ex])**2) for j in range(tail,N)]
        
        As_exp[ex,1,rep] = A_run[tail:N]
        Sigmas_exp[ex,1,rep] = [A_run[j]@np.transpose(A_run[j]) for j in range(tail,N)]
        
        Sigmas0p1_exp[ex,1,rep] = [A_run[j]@np.diag(np.exp(-phis_run[j]*0.1))@np.transpose(A_run[j]) for j in range(tail,N)]
        
        likes = np.array([np.sqrt(np.diag(taus_run[j])/2/np.pi)@np.exp(-1/2*np.diag(taus_run[j])@(Y_obs-V_run[j])**2) for j in range(tail,N)])
        
        waic[ex,1,rep] = - np.mean(np.log(np.mean(likes,axis=0))) + np.mean(np.var(np.log(likes),axis=0))
        
        
ETG = time.time()

print("GLOBAL TIME", (ETG-STG)/60, "min")


### differences in RMSE

dMSE = np.sqrt(np.mean(MSES,axis=(3,4,5)))[:,0,:] - np.sqrt(np.mean(MSES,axis=(3,4,5)))[:,1,:]
dWAIC = waic[:,0,:] - waic[:,1,:]
dWnorms = np.sqrt(np.mean(Wnorms,axis=3))[:,0,:] - np.sqrt(np.mean(Wnorms,axis=3))[:,1,:]
dWnorms0p1 = np.sqrt(np.mean(Wnorms0p1,axis=3))[:,0,:] - np.sqrt(np.mean(Wnorms0p1,axis=3))[:,1,:]

np.mean(dMSE<0,axis=1)

# dMSE = MSES[:,0,:] - MSES[:,1,:]
        
# np.savetxt("dMSE.csv", dMSE, delimiter=",")  
# np.savetxt("dWnorms.csv", dWnorms, delimiter=",")  
# np.save("n_comps.npy", n_comps)
# np.save("fnorms.npy", fnorms)
# np.save("fnorms0p1.npy", fnorms0p1)

np.save("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),MSES)
np.save("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),n_comps)
np.save("1mAs_exp"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),As_exp)
np.save("1mSigmas_exp"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),Sigmas_exp)
np.save("1mSigmas0p1_exp"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),Sigmas0p1_exp)
np.save("1mWnorms"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),Wnorms)
np.save("1mWnorms0p1"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),Wnorms0p1)
np.save("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),times_all)
np.save("1mwaic"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one),waic)




np.mean(n_comps,axis=2)
np.mean(n_comps,axis=(1,2))
np.mean(MSES,axis=(3,4,5))
np.mean(MSES,axis=(2,3,4,5))
np.mean(Wnorms,axis=3)
np.mean(Wnorms0p1,axis=3)
np.mean(Wnorms,axis=(2,3))
np.mean(Wnorms0p1,axis=(2,3))



np.round(Sigmas,2)[0]
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[0,0],axis=0)
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[0,1],axis=0)

np.round(Sigmas,2)[1]
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[1,0],axis=0)
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[1,1],axis=0)

np.round(Sigmas,2)[2]
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[2,0],axis=0)
np.median(np.round(np.median(Sigmas_exp,axis=3),2)[2,1],axis=0)




np.round(Sigmas0p1,2)[0]
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[0,0],axis=0)
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[0,1],axis=0)

np.round(Sigmas0p1,2)[1]
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[1,0],axis=0)
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[1,1],axis=0)

np.round(Sigmas0p1,2)[2]
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[2,0],axis=0)
np.median(np.round(np.median(Sigmas0p1_exp,axis=3),2)[2,1],axis=0)

### MSE plots

my_dict = {'Full': dMSE[0], 'Triangular': dMSE[1], 'Diagonal': dMSE[2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("p="+str(p)+", conc="+str(conc)+", noise_sd="+str(noise_sd)+", prob_one="+str(prob_one))
plt.show()


plt.boxplot(dMSE[0])
plt.show()
plt.boxplot(dMSE[1])
plt.show()
plt.boxplot(dMSE[2])
plt.show()

### Waic plots

my_dict = {'Full': dWAIC[0], 'Triangular': dWAIC[1], 'Diagonal': dWAIC[2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.show()


plt.boxplot(dWAIC[0])
plt.show()
plt.boxplot(dWAIC[1])
plt.show()
plt.boxplot(dWAIC[2])
plt.show()



### Wnorms plots

my_dict = {'Full': dWnorms[0], 'Triangular': dWnorms[1], 'Diagonal': dWnorms[2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.show()


plt.boxplot(dWnorms[0])
plt.show()
plt.boxplot(dWnorms[1])
plt.show()
plt.boxplot(dWnorms[2])
plt.show()

my_dict = {'Full': dWnorms0p1[0], 'Triangular': dWnorms0p1[1], 'Diagonal': dWnorms0p1[2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.show()


plt.boxplot(dWnorms0p1[0])
plt.show()
plt.boxplot(dWnorms0p1[1])
plt.show()
plt.boxplot(dWnorms0p1[2])
plt.show()



