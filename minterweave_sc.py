
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 12:47:17 2023

@author: alier
"""



import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

from LMC_inference import phis_move
from noisyLMC_inference import V_move_conj, taus_move

# import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix


random.seed(10)

def vec(A):
    
    p = A.shape[0]
    n = A.shape[1]
    
    return(np.reshape(A,n*p,order='F'))


def vec_inv(A, nrow):
    
    N = A.shape[0]
    ncol = N//nrow
    
    
    return(np.reshape(A,newshape=(nrow,ncol),order='F'))


def A_move(A_current,A_inv_current,A_invV_current,A_prop,sigma_A,mu_A,V,Rs_inv_current):

    
    p = V.shape[0] 
    n = V.shape[1]

    A_new = A_current + A_prop*random.normal(size=(p,p))
    
    # if np.prod(A_new[0]>0):
    A_inv_new = np.linalg.inv(A_new)
    
    A_invV_new = A_inv_new @ V
    
    rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n * np.exp(-1/2/sigma_A**2 * (np.sum((A_new-mu_A)**2) - np.sum((A_current-mu_A)**2)))
    
    if random.uniform() < rat:
        
        return(A_new,A_inv_new,A_invV_new,1)
    else:
        
        return(A_current,A_inv_current,A_invV_current,0)
    # else:
        
        # return(A_current,A_inv_current,A_invV_current,0)

def A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A):

    
    p = A_current.shape[0] 
    n = A_invV_current.shape[1]
    
    ### threshold
    z =  -1/2 * np.sum( [A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_current))) - 1/2/sigma_A**2 * np.sum((A_current-mu_A)**2) - random.exponential(1,1)
    
    L = A_current - random.uniform(0,sigma_slice,(p,p))
    # L[0] = np.maximum(L[0],0)
    
    U = L + sigma_slice
        
    while True:
    
        
        
        A_prop = random.uniform(L,U)
        A_inv_prop = np.linalg.inv(A_prop)
        A_invV_prop = A_inv_prop @ V_current
        
        acc = z < -1/2 * np.sum( [A_invV_prop[j] @ Rs_inv_current[j] @ A_invV_prop[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_prop))) - 1/2/sigma_A**2 * np.sum((A_prop-mu_A)**2) 
            
        if acc:
            return(A_prop,A_inv_prop,A_invV_prop)
        else:
            for ii in range(p):
                for jj in range(p):
                    if A_prop[ii,jj] < A_current[ii,jj]:
                        L[ii,jj] = A_prop[ii,jj]
                    else:
                        U[ii,jj] = A_prop[ii,jj]



def A_move_white(A_invV_current,Dm1_current,Dm1Y_current,sigma_A,mu_A):
    
    p = A_invV_current.shape[0]
    
    M = np.kron( A_invV_current @ np.transpose(A_invV_current), Dm1_current ) + np.identity(p**2)/sigma_A**2
    b1 = vec(Dm1Y_current @ np.transpose(A_invV_current)) + vec(mu_A)/sigma_A**2
    
    M_inv = np.linalg.inv(M)
    
    A_current = np.zeros(shape=(p,p))
    
    # while not np.prod(A_current[0]>0):
    A_current = vec_inv( np.linalg.cholesky(M_inv) @ random.normal(size=p**2) + M_inv @ b1, p)
    A_inv_current = np.linalg.inv(A_current)
    
    ### update V
    V_current = A_current @ A_invV_current
    
    return(A_current,A_inv_current,V_current)


def makeGrid(n):
    
    locs1D = (np.arange(n) + 0.5)/n
    xv, yv = np.meshgrid(locs1D, locs1D)
    locs = np.transpose(np.concatenate(([xv.flatten()],[yv.flatten()]),axis=0))
    
    return(locs)

def crossCov(d,A,phis,i,j):
    return(np.sum(A[i] * A[j] * np.exp(-d*phis)))    

def practRange(min_range, max_range,A,phis,i,prec=0.001):
    
    min_range = 0
    max_range = 1
    
    cov = crossCov(0,A,phis,i,i)
    
    cov_max = crossCov(max_range,A,phis,i,i)
    
    while cov_max/cov > 0.05:
        max_range *= 2
        cov_max = crossCov(max_range,A,phis,i,i)
    
    range_cand = (max_range - min_range)/2
    cov_cand = crossCov(range_cand,A,phis,i,i)
    
    while np.abs(cov_cand/cov - 0.05) > prec:
        
        if cov_cand/cov > 0.05:
            min_range = range_cand
        else: 
            max_range = range_cand
        
        range_cand = (max_range + min_range)/2
        cov_cand = crossCov(range_cand,A,phis,i,i)
    
    return(range_cand)


### TEST OF VEC/INV VEC FUNCTION

# A = np.array([[1,2,7],[3,4,8],[5,6,9],[10,11,12]])
# nrow = A.shape[0]
# print(A)
# print(vec(A))
# print(vec_inv(vec(A),nrow))



### global parameters
n = 200
# n = 20
p = 2
# p = 3


### generate random example
locs = random.uniform(0,1,(n,2))
# locs = np.linspace(0, 1, n)

# locs = makeGrid(n)
A_base = np.array([[-1.5,1.1],
                   [1,2]])

phis = np.array([5.,20.])
taus_sqrt_inv_base = np.array([1.,1.]) * 1


# A = np.array([[-1.,-1.5,2.],
#               [-2.,1.5,1.],
#               [1.5,1.,2.]])
# A = np.array([[-1.,0,1.2],
#               [1.,0,1.2],
#               [0.,np.sqrt(1.2**2 + 1.**2),0.]])
# phis = np.array([5.,10.,20.])
# taus_sqrt_inv = np.array([1.,1.,1.]) * 0.1


### showcase crosscovariance

# max_d = 1
# res = 100

# ds = np.linspace(0,max_d,res)


    
# cc = np.zeros(res)

# i=0
# j=0

# for r in range(res):
#     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="tab:blue")
# # plt.show()

# i=1
# j=1

# for r in range(res):
#     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="tab:orange")
# # plt.show()

# # i=2
# # j=2

# # for r in range(res):
# #     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="tab:green")
# # plt.show()

# i=0
# j=1

# for r in range(res):
#     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="black")
# # plt.show()

# # i=0
# # j=2

# # for r in range(res):
# #     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="black")
# # plt.show()



# # i=1
# # j=2

# # for r in range(res):
# #     cc[r] = crossCov(ds[r],A,phis,i,j)
    
# # plt.plot(ds,cc, c="black")
# # plt.show()



### priors
sigma_A = 2.
mu_A = np.array([[0.,0.],
                 [0.,0.]])
# mu_A = np.array([[0.,0.,0.],
#                  [0.,0.,0.],
#                  [0.,0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


# alphas = np.linspace(1, p, p)*10
# betas = np.linspace(p, 1, p)*10

# prior_means = alphas/(alphas+betas) * range_phi + min_phi

### UNIFORM PRIORS

alphas = np.ones(p)
betas = np.ones(p)

### showcase of priors for phis
# from scipy.stats import beta

# for i in range(p):
#     plt.plot(np.linspace(0, 1, 1001)*range_phi + min_phi,beta.pdf(np.linspace(0, 1, 1001), alphas[i], betas[i]))
# plt.show()



## tau prior var = a/b^2, mean = a/b

a = 0.1
b = 0.1


### proposals

# phis_prop = np.linspace(1/p, 1, p) * 2.
phis_prop = np.ones(p) * 1
# A_prop = 0.03
sigma_slice = 10
# V_prop = 0.005


### samples
N = 2000

### global run containers
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
V_run = np.zeros((N,p,n))

### acc vector

acc_phis = np.zeros((p,N))
acc_A = np.zeros(N)
# acc_V = np.zeros(N)

Dists = distance_matrix(locs,locs)
# Dists = distance_matrix(np.transpose(np.array([locs])),np.transpose(np.array([locs])))

tail = 1000
reps = 100
nquant = 4
n_meth = 3

ratios = np.array([np.sqrt(0.1),1,np.sqrt(10)])
n_rat = ratios.shape[0]




arr = np.zeros((n_meth,n_rat,nquant,N-tail,reps))
times = np.zeros((n_meth,n_rat,reps))

import time
stg = time.time() ### global start

### INTERWEAVE



for rat in range(n_rat):
    taus_sqrt_inv = taus_sqrt_inv_base/ratios[rat]
    for rep in range(reps):
        print("------------INTERWEAVE-----------")    
    
        Y, V_true = rNLMC(A_base,phis,taus_sqrt_inv,locs, retV=True)
        # Y, V_true = rNLMC(A,phis,taus_sqrt_inv,np.transpose(np.array([locs])), retV=True)
        
        
        ### showcase V and Y
        
        # plt.plot(locs,V_true[0])
        # plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        # plt.plot(locs,V_true[1])
        # plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        # plt.plot(locs,V_true[2])
        # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        # plt.show()
        
        ### showcase V
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[0],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[1],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[2],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        
        
        
        
        
        ### useful quantities 
        
    
        
        ### init and current state
        phis_current = np.array([5.,20.])
        # phis_current = np.array([5.,10.,20.])
        Rs_current = np.array([ np.exp(-Dists*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        V_current = V_true
        # V_current = Y + random.normal(size=(p,n))*0.1
        # V_current = random.normal(size=(p,n))*1
        VmY_current = V_current - Y
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        # A_current = np.array([[1.,1.],
        #               [0.,1.]])
        # A_current = np.array([[1.,1.,1.],
        #                       [0.,1.,1.],
        #                       [0.,0.,1.]])
        # A_current = np.identity(p)
        # A_current = random.normal(size=(p,p)) * 1
        A_current = np.copy(A_base)
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/(taus_sqrt_inv_base/ratios[rat])**2
        # taus_current = 1/(np.array([1.,1.,1.]) * 0.1)**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y
        
        
        
    
        
        
        
        
        st = time.time() ### local time
        
        
        for i in range(N):
            
            
            V_current, dumb, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current, V_current, np.zeros(p))
                
            
            
            #### Interweave method                    
            
            A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A)
            
            A_current, A_inv_current, V_current = A_move_white(A_invV_current,Dm1_current,Dm1Y_current,sigma_A,mu_A) 
            
            
            
            
        
            
            
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists,A_invV_current,Rs_current,Rs_inv_current)
        
            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            
            if i % 100 == 0:
                print(i)
        
        et = time.time()
        #### SAVE TIME
        times[0,rat,rep] = et-st
        print('Execution time:', (et-st)/60, 'minutes')
        
        
        
        # print("Prior Means for Ranges", alphas / (alphas + betas) * range_phi + min_phi)
        
        
        
        print('accept phi_1:',np.mean(acc_phis[0,tail:]))
        print('accept phi_2:',np.mean(acc_phis[1,tail:]))
        # print('accept phi_3:',np.mean(acc_phis[2,tail:]))
        
        # plt.plot(phis_run[:,0])
        # plt.plot(phis_run[:,1])
        # plt.plot(phis_run[:,2])
        # plt.show()
        
        # print('mean phi_1:',np.mean(phis_run[tail:,0]))
        # print('mean phi_2:',np.mean(phis_run[tail:,1]))
        # print('mean phi_3:',np.mean(phis_run[tail:,2]))
        
        # print('accept A:',np.mean(acc_A[tail:]))
        # print('mean A:',np.mean(A_run[tail:],axis=0))
        
        # plt.plot(A_run[:,0,0])
        # plt.plot(A_run[:,0,1])
        # plt.plot(A_run[:,0,2])
        # plt.plot(A_run[:,1,0])
        # plt.plot(A_run[:,1,1])
        # plt.plot(A_run[:,1,2])
        # plt.plot(A_run[:,2,0])
        # plt.plot(A_run[:,2,1])
        # plt.plot(A_run[:,2,2])
        # plt.show()
        
        
        # print('accept V:',np.mean(acc_V[tail:]))
        
        # print('mean tau_1:',np.mean(taus_run[tail:,0]))
        # print('mean tau_2:',np.mean(taus_run[tail:,1]))
        # print('mean tau_3:',np.mean(taus_run[tail:,2]))
        
        # print('real taus:',taus_sqrt_inv ** (-2))
        
        # print('mean inv sqrt tau_1:',np.mean(1/np.sqrt(taus_run[tail:,0])))
        # print('mean inv sqrt tau_2:',np.mean(1/np.sqrt(taus_run[tail:,1])))
        # print('mean inv sqrt tau_3:',np.mean(1/np.sqrt(taus_run[tail:,2])))
        
        # print('real sqrt inv taus:',taus_sqrt_inv)
        
        # plt.plot(taus_run[:,0])
        # plt.plot(taus_run[:,1])
        # plt.plot(taus_run[:,2])
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        
        # plt.plot(1/np.sqrt(taus_run[:,0]))
        # plt.plot(1/np.sqrt(taus_run[:,1]))
        # plt.plot(1/np.sqrt(taus_run[:,2]))
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        # for i in range(N):
        #     if i % 100 == 0:
        #         plt.plot(locs,V_run[i,0])
        #         plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        #         plt.plot(locs,V_run[i,1])
        #         plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        #         # plt.plot(locs,V_run[i,2])
        #         # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        #         plt.show()
        
        
        ### inference of cross covariance
        
        # max_d = 1
        # res = 100
        
        # ds = np.linspace(0,max_d,res)
            
        # cc = np.zeros((N-tail,res))
        # cc_true = np.zeros(res) 
        
        p_range = np.zeros(N-tail) ### practical range container
        
        i=0
        j=0
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
                
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
            
            
    
         
            
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show()   
        
        # arr[0,:,rep] = cc[:,0]
        arr[0,rat,0,:,rep] = p_range
        
        
        
        
        i=1
        j=1
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
    
        
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show() 
        
        # arr[2,:,rep] = cc[:,0]
        arr[0,rat,1,:,rep] = p_range
        
        
        
        
        
        
        i=0
        j=1
        
        cc_0 = np.zeros(N-tail)
        cc_0p1 = np.zeros(N-tail)
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ## cross covariance at 0,0.1
            
            cc_0[ns-tail] = crossCov(0,A_run[ns],phis_run[ns],i,j)
            cc_0p1[ns-tail] = crossCov(0.1,A_run[ns],phis_run[ns],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        arr[0,rat,2,:,rep] = cc_0
        arr[0,rat,3,:,rep] = cc_0p1
        
        
        
        # i=0
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)               
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")   
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # i=1
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)           
        
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # plt.plot(A_run[tail:,0,0])
        # plt.show()
        # plt.plot(A_run[tail:,0,1])
        # plt.show()
        # plt.plot(A_run[tail:,0,2])
        # plt.show()
        # plt.plot(A_run[tail:,1,0])
        # plt.show()
        # plt.plot(A_run[tail:,1,1])
        # plt.show()
        # plt.plot(A_run[tail:,1,2])
        # plt.show()
        # plt.plot(A_run[tail:,2,0])
        # plt.show()
        # plt.plot(A_run[tail:,2,1])
        # plt.show()
        # plt.plot(A_run[tail:,2,2])
        # plt.show()
        
        print("Rat " + str(rat))
        print("Rep " + str(rep))
    
    
    



# arr = np.zeros((N-tail,reps))

# arr[:,0] = A_run[tail:,0,0]
# arr[:,1] = A_run[tail:,0,1]
# arr[:,2] = A_run[tail:,1,0]
# arr[:,3] = A_run[tail:,1,1]

# np.savetxt('interweaveC00.csv', arr[0], delimiter=',')
# np.savetxt('interweavep00.csv', arr[1], delimiter=',')
# np.savetxt('interweaveC11.csv', arr[2], delimiter=',')
# np.savetxt('interweavep11.csv', arr[3], delimiter=',')
# np.savetxt('interweaveC01.csv', arr[4], delimiter=',')


### CENTERED



for rat in range(n_rat):
    taus_sqrt_inv = taus_sqrt_inv_base/ratios[rat]
    for rep in range(reps):
        
        print("------------CENTERED-----------")
    
        Y, V_true = rNLMC(A_base,phis,taus_sqrt_inv,locs, retV=True)
        # Y, V_true = rNLMC(A,phis,taus_sqrt_inv,np.transpose(np.array([locs])), retV=True)
        
        
        ### showcase V and Y
        
        # plt.plot(locs,V_true[0])
        # plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        # plt.plot(locs,V_true[1])
        # plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        # plt.plot(locs,V_true[2])
        # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        # plt.show()
        
        ### showcase V
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[0],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[1],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[2],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        
        
        
        
        
        ### useful quantities 
        
    
        
        ### init and current state
        phis_current = np.array([5.,20.])
        # phis_current = np.array([5.,10.,20.])
        Rs_current = np.array([ np.exp(-Dists*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        V_current = V_true
        # V_current = Y + random.normal(size=(p,n))*0.1
        # V_current = random.normal(size=(p,n))*1
        VmY_current = V_current - Y
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        # A_current = np.array([[1.,1.],
        #               [0.,1.]])
        # A_current = np.array([[1.,1.,1.],
        #                       [0.,1.,1.],
        #                       [0.,0.,1.]])
        # A_current = np.identity(p)
        A_current = np.copy(A_base)
        # A_current = random.normal(size=(p,p)) * 1
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/(taus_sqrt_inv_base/ratios[rat])**2
        # taus_current = 1/(np.array([1.,1.,1.]) * 0.1)**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y
        
        
        
    
        
        
        
        
        st = time.time() ### local time
        
        
        for i in range(N):
            
            
            V_current, dumb, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current, V_current, np.zeros(p))            
            
            
            #### Centered method                    
            
            A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A)
            
            # A_current, A_inv_current, V_current = A_move_white(A_invV_current,Dm1_current,Dm1Y_current,sigma_A,mu_A) 
            
            
            
            
        
            
            
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists,A_invV_current,Rs_current,Rs_inv_current)
    
            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
             
            V_run[i] = V_current
            taus_run[i] = taus_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            
            if i % 100 == 0:
                print(i)
        
        et = time.time()
        #### SAVE TIME
        times[1,rat,rep] = et-st
        print('Execution time:', (et-st)/60, 'minutes')
        
        
        
        # print("Prior Means for Ranges", alphas / (alphas + betas) * range_phi + min_phi)
        
        
        
        print('accept phi_1:',np.mean(acc_phis[0,tail:]))
        print('accept phi_2:',np.mean(acc_phis[1,tail:]))
        # print('accept phi_3:',np.mean(acc_phis[2,tail:]))
        
        # plt.plot(phis_run[:,0])
        # plt.plot(phis_run[:,1])
        # plt.plot(phis_run[:,2])
        # plt.show()
        
        # print('mean phi_1:',np.mean(phis_run[tail:,0]))
        # print('mean phi_2:',np.mean(phis_run[tail:,1]))
        # print('mean phi_3:',np.mean(phis_run[tail:,2]))
        
        # print('accept A:',np.mean(acc_A[tail:]))
        # print('mean A:',np.mean(A_run[tail:],axis=0))
        
        # plt.plot(A_run[:,0,0])
        # plt.plot(A_run[:,0,1])
        # plt.plot(A_run[:,0,2])
        # plt.plot(A_run[:,1,0])
        # plt.plot(A_run[:,1,1])
        # plt.plot(A_run[:,1,2])
        # plt.plot(A_run[:,2,0])
        # plt.plot(A_run[:,2,1])
        # plt.plot(A_run[:,2,2])
        # plt.show()
        
        
        # print('accept V:',np.mean(acc_V[tail:]))
        
        # print('mean tau_1:',np.mean(taus_run[tail:,0]))
        # print('mean tau_2:',np.mean(taus_run[tail:,1]))
        # print('mean tau_3:',np.mean(taus_run[tail:,2]))
        
        # print('real taus:',taus_sqrt_inv ** (-2))
        
        # print('mean inv sqrt tau_1:',np.mean(1/np.sqrt(taus_run[tail:,0])))
        # print('mean inv sqrt tau_2:',np.mean(1/np.sqrt(taus_run[tail:,1])))
        # print('mean inv sqrt tau_3:',np.mean(1/np.sqrt(taus_run[tail:,2])))
        
        # print('real sqrt inv taus:',taus_sqrt_inv)
        
        # plt.plot(taus_run[:,0])
        # plt.plot(taus_run[:,1])
        # plt.plot(taus_run[:,2])
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        
        # plt.plot(1/np.sqrt(taus_run[:,0]))
        # plt.plot(1/np.sqrt(taus_run[:,1]))
        # plt.plot(1/np.sqrt(taus_run[:,2]))
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        # for i in range(N):
        #     if i % 100 == 0:
        #         plt.plot(locs,V_run[i,0])
        #         plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        #         plt.plot(locs,V_run[i,1])
        #         plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        #         # plt.plot(locs,V_run[i,2])
        #         # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        #         plt.show()
        
        
        ### inference of cross covariance
        
        # max_d = 1
        # res = 100
        
        # ds = np.linspace(0,max_d,res)
            
        # cc = np.zeros((N-tail,res))
        # cc_true = np.zeros(res) 
        
        p_range = np.zeros(N-tail) ### practical range container
        
        i=0
        j=0
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
                
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
            
            
    
         
            
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show()   
        
        # arr[0,:,rep] = cc[:,0]
        arr[1,rat,0,:,rep] = p_range
        
        
        
        
        i=1
        j=1
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
    
        
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show() 
        
        # arr[2,:,rep] = cc[:,0]
        arr[1,rat,1,:,rep] = p_range
        
        
        
        
        
        
        i=0
        j=1
        
        cc_0 = np.zeros(N-tail)
        cc_0p1 = np.zeros(N-tail)
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ## cross covariance at 0,0.1
            
            cc_0[ns-tail] = crossCov(0,A_run[ns],phis_run[ns],i,j)
            cc_0p1[ns-tail] = crossCov(0.1,A_run[ns],phis_run[ns],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        arr[1,rat,2,:,rep] = cc_0
        arr[1,rat,3,:,rep] = cc_0p1
        
        
        
        # i=0
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)               
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")   
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # i=1
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)           
        
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # plt.plot(A_run[tail:,0,0])
        # plt.show()
        # plt.plot(A_run[tail:,0,1])
        # plt.show()
        # plt.plot(A_run[tail:,0,2])
        # plt.show()
        # plt.plot(A_run[tail:,1,0])
        # plt.show()
        # plt.plot(A_run[tail:,1,1])
        # plt.show()
        # plt.plot(A_run[tail:,1,2])
        # plt.show()
        # plt.plot(A_run[tail:,2,0])
        # plt.show()
        # plt.plot(A_run[tail:,2,1])
        # plt.show()
        # plt.plot(A_run[tail:,2,2])
        # plt.show()
        
        print("Rat " + str(rat))
        print("Rep " + str(rep))
    
    
    



# arr = np.zeros((N-tail,reps))

# arr[:,0] = A_run[tail:,0,0]
# arr[:,1] = A_run[tail:,0,1]
# arr[:,2] = A_run[tail:,1,0]
# arr[:,3] = A_run[tail:,1,1]

# np.savetxt('interweaveC00.csv', arr[0], delimiter=',')
# np.savetxt('interweavep00.csv', arr[1], delimiter=',')
# np.savetxt('interweaveC11.csv', arr[2], delimiter=',')
# np.savetxt('interweavep11.csv', arr[3], delimiter=',')
# np.savetxt('interweaveC01.csv', arr[4], delimiter=',')

### WHITE



for rat in range(n_rat):
    taus_sqrt_inv = taus_sqrt_inv_base/ratios[rat]
    for rep in range(reps):
        
        print("------------WHITE-----------")
    
        Y, V_true = rNLMC(A_base,phis,taus_sqrt_inv,locs, retV=True)
        # Y, V_true = rNLMC(A,phis,taus_sqrt_inv,np.transpose(np.array([locs])), retV=True)
        
        
        ### showcase V and Y
        
        # plt.plot(locs,V_true[0])
        # plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        # plt.plot(locs,V_true[1])
        # plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        # plt.plot(locs,V_true[2])
        # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        # plt.show()
        
        ### showcase V
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[0],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[1],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.pcolormesh(locs[:n,0], locs[:n,0], np.reshape(V_true[2],(n,n)))
        # ax.set_aspect(1)
        # ax.set(xlim=(0, 1), ylim=(0, 1))
        # plt.show()
        
        
        
        
        
        
        ### useful quantities 
        
    
        
        ### init and current state
        phis_current = np.array([5.,20.])
        # phis_current = np.array([5.,10.,20.])
        Rs_current = np.array([ np.exp(-Dists*phis_current[j]) for j in range(p) ])
        Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])
        
        V_current = V_true
        # V_current = Y + random.normal(size=(p,n))*0.1
        # V_current = random.normal(size=(p,n))*1
        VmY_current = V_current - Y
        VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
        
        # A_current = np.array([[1.,1.],
        #               [0.,1.]])
        # A_current = np.array([[1.,1.,1.],
        #                       [0.,1.,1.],
        #                       [0.,0.,1.]])
        # A_current = np.identity(p)
        A_current = np.copy(A_base)
        # A_current = random.normal(size=(p,p)) * 1
        A_inv_current = np.linalg.inv(A_current)
        
        A_invV_current = A_inv_current @ V_current
        
        taus_current = 1/(taus_sqrt_inv_base/ratios[rat])**2
        # taus_current = 1/(np.array([1.,1.,1.]) * 0.1)**2
        Dm1_current = np.diag(taus_current)
        Dm1Y_current = Dm1_current @ Y
        
        
        
    
        
        
        
        
        st = time.time() ### local time
        
        
        for i in range(N):
            
            
            V_current, dumb, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current, V_current, np.zeros(p))     
            
            
            #### White method                    
            
            # A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A)
            
            A_current, A_inv_current, V_current = A_move_white(A_invV_current,Dm1_current,Dm1Y_current,sigma_A,mu_A) 
            
            
            
            
        
            
            
            phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,Dists,A_invV_current,Rs_current,Rs_inv_current)
    
            taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
            
            V_run[i] = V_current
            taus_run[i] = taus_current
            phis_run[i] =  phis_current
            A_run[i] = A_current
            
            if i % 100 == 0:
                print(i)
        
        et = time.time()
        #### SAVE TIME
        times[2,rat,rep] = et-st
        print('Execution time:', (et-st)/60, 'minutes')
        
        
        
        # print("Prior Means for Ranges", alphas / (alphas + betas) * range_phi + min_phi)
        
        
        
        print('accept phi_1:',np.mean(acc_phis[0,tail:]))
        print('accept phi_2:',np.mean(acc_phis[1,tail:]))
        # print('accept phi_3:',np.mean(acc_phis[2,tail:]))
        
        # plt.plot(phis_run[:,0])
        # plt.plot(phis_run[:,1])
        # plt.plot(phis_run[:,2])
        # plt.show()
        
        # print('mean phi_1:',np.mean(phis_run[tail:,0]))
        # print('mean phi_2:',np.mean(phis_run[tail:,1]))
        # print('mean phi_3:',np.mean(phis_run[tail:,2]))
        
        # print('accept A:',np.mean(acc_A[tail:]))
        # print('mean A:',np.mean(A_run[tail:],axis=0))
        
        # plt.plot(A_run[:,0,0])
        # plt.plot(A_run[:,0,1])
        # plt.plot(A_run[:,0,2])
        # plt.plot(A_run[:,1,0])
        # plt.plot(A_run[:,1,1])
        # plt.plot(A_run[:,1,2])
        # plt.plot(A_run[:,2,0])
        # plt.plot(A_run[:,2,1])
        # plt.plot(A_run[:,2,2])
        # plt.show()
        
        
        # print('accept V:',np.mean(acc_V[tail:]))
        
        # print('mean tau_1:',np.mean(taus_run[tail:,0]))
        # print('mean tau_2:',np.mean(taus_run[tail:,1]))
        # print('mean tau_3:',np.mean(taus_run[tail:,2]))
        
        # print('real taus:',taus_sqrt_inv ** (-2))
        
        # print('mean inv sqrt tau_1:',np.mean(1/np.sqrt(taus_run[tail:,0])))
        # print('mean inv sqrt tau_2:',np.mean(1/np.sqrt(taus_run[tail:,1])))
        # print('mean inv sqrt tau_3:',np.mean(1/np.sqrt(taus_run[tail:,2])))
        
        # print('real sqrt inv taus:',taus_sqrt_inv)
        
        # plt.plot(taus_run[:,0])
        # plt.plot(taus_run[:,1])
        # plt.plot(taus_run[:,2])
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        
        # plt.plot(1/np.sqrt(taus_run[:,0]))
        # plt.plot(1/np.sqrt(taus_run[:,1]))
        # plt.plot(1/np.sqrt(taus_run[:,2]))
        # # plt.plot(1/np.sqrt(taus_run[:,2]))
        # plt.show()
        
        # for i in range(N):
        #     if i % 100 == 0:
        #         plt.plot(locs,V_run[i,0])
        #         plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        #         plt.plot(locs,V_run[i,1])
        #         plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)
        #         # plt.plot(locs,V_run[i,2])
        #         # plt.plot(locs,Y[2], '.', c="tab:green", alpha=0.5)
        
        #         plt.show()
        
        
        ### inference of cross covariance
        
        # max_d = 1
        # res = 100
        
        # ds = np.linspace(0,max_d,res)
            
        # cc = np.zeros((N-tail,res))
        # cc_true = np.zeros(res) 
        
        p_range = np.zeros(N-tail) ### practical range container
        
        i=0
        j=0
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
                
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
            
            
    
         
            
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show()   
        
        # arr[0,:,rep] = cc[:,0]
        arr[2,rat,0,:,rep] = p_range
        
        
        
        
        i=1
        j=1
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ### practical range
            
            min_range = 0
            max_range = 1
            
            p_range[ns-tail] = practRange(min_range,max_range,A_run[ns],phis_run[ns],i)
    
        
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        # plt.plot(p_range, c="tab:orange")
        # plt.show() 
        
        # arr[2,:,rep] = cc[:,0]
        arr[2,rat,1,:,rep] = p_range
        
        
        
        
        
        
        i=0
        j=1
        
        cc_0 = np.zeros(N-tail)
        cc_0p1 = np.zeros(N-tail)
        
        for ns in range(tail,N):
            # for r in range(res):
            #     cc[ns-tail,r] = crossCov(ds[r],A_run[ns],phis_run[ns],i,j)
            
            ## cross covariance at 0,0.1
            
            cc_0[ns-tail] = crossCov(0,A_run[ns],phis_run[ns],i,j)
            cc_0p1[ns-tail] = crossCov(0.1,A_run[ns],phis_run[ns],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)        
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        arr[2,rat,2,:,rep] = cc_0
        arr[2,rat,3,:,rep] = cc_0p1
        
        
        
        # i=0
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)               
            
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")   
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # i=1
        # j=2
        
        
        
        # for n in range(tail,N):
        #     for r in range(res):
        #         cc[n-tail,r] = crossCov(ds[r],A_run[n],phis_run[n],i,j)
        
        # for r in range(res):
        #     cc_true[r] = crossCov(ds[r],A,phis,i,j)           
        
        # plt.fill_between(ds, np.quantile(cc,0.05,axis=0), np.quantile(cc,0.95,axis=0), color="silver")    
        # plt.plot(ds,np.mean(cc,axis=0), c="black")
        # plt.plot(ds,cc_true)
        # plt.show()
        
        # plt.plot(cc[:,0])
        # plt.show()
        
        
        # plt.plot(A_run[tail:,0,0])
        # plt.show()
        # plt.plot(A_run[tail:,0,1])
        # plt.show()
        # plt.plot(A_run[tail:,0,2])
        # plt.show()
        # plt.plot(A_run[tail:,1,0])
        # plt.show()
        # plt.plot(A_run[tail:,1,1])
        # plt.show()
        # plt.plot(A_run[tail:,1,2])
        # plt.show()
        # plt.plot(A_run[tail:,2,0])
        # plt.show()
        # plt.plot(A_run[tail:,2,1])
        # plt.show()
        # plt.plot(A_run[tail:,2,2])
        # plt.show()
        
        print("Rat " + str(rat))
        print("Rep " + str(rep))
    
    
    



# arr = np.zeros((N-tail,reps))

# arr[:,0] = A_run[tail:,0,0]
# arr[:,1] = A_run[tail:,0,1]
# arr[:,2] = A_run[tail:,1,0]
# arr[:,3] = A_run[tail:,1,1]

# np.savetxt('interweaveC00.csv', arr[0], delimiter=',')
# np.savetxt('interweavep00.csv', arr[1], delimiter=',')
# np.savetxt('interweaveC11.csv', arr[2], delimiter=',')
# np.savetxt('interweavep11.csv', arr[3], delimiter=',')
# np.savetxt('interweaveC01.csv', arr[4], delimiter=',')


etg = time.time()
print('Global time:', (etg-stg)/60, 'minutes')
print('Mean times:', np.mean(times, axis=2))

np.save("mresults.npy", arr)
np.save("mtimes.npy", times)

