





import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

from LMC_inference import A_move, phis_move

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

def M_comp(Rs_inv_current, A_inv_current, taus):
    
    p = Rs_inv_current.shape[0]
    n = Rs_inv_current.shape[1]
                        
    
    M = np.sum([ np.kron( Rs_inv_current[j] , np.outer(A_inv_current[j],A_inv_current[j]) ) for j in range(p) ],axis=0) 
    
    M = M.reshape((p,n,p,n), order="F")
    
    return(M)



def V_move_conj_uni(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, mu_current):
    
    p = Rs_inv_current.shape[0]
    n = Rs_inv_current.shape[1]
    
    M = M_comp(Rs_inv_current, A_inv_current, taus_current)
    
    
    for i in random.permutation(range(n)):
        for j in random.permutation(range(p)):
            
            Vmmu1_current[j,i] = np.sqrt(1/(M[j,i,j,i]+taus_current[j])) * random.normal() + 1/(M[j,i,j,i]+taus_current[j]) * (Dm1Y_current[j,i] - np.sum(M[j,i]*Vmmu1_current) + M[j,i,j,i]*Vmmu1_current[j,i] - taus_current[j]*mu_current[j])
    
    V_current = Vmmu1_current + np.outer(mu_current,np.ones(n))   
    
    VmY_current = V_current - Y
    VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)


# def V_move_conj_scale(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, A_invVmmu1_current, mu_current):
    
#     p = Rs_inv_current.shape[0]
#     n = Rs_inv_current.shape[1]
    
#     outsies = np.array([np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)])
    
    
    
#     for i in random.permutation(range(n)):
#         delta_i = np.sum([Rs_inv_current[j,i,i]*outsies[j] for j in range(p)],axis=0)
#         M = delta_i + Dm1_current
#         Minv = np.linalg.inv(M)

#         gamma = np.array([np.sum(Rs_inv_current[j,i]*(V_current-np.outer(mu_current,np.ones(n))),axis=1) - Rs_inv_current[j,i,i]*(V_current[:,i]-mu_current) for j in range(p)])
        
#         b = Dm1Y_current[:,i] + delta_i@mu_current - np.sum([ outsies[j]@gamma[j] for j in range(p)],axis=0) 
        
#         V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b

#     Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))   
    
#     VmY_current = V_current - Y
#     VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    
#     A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
#     return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)

def V_move_conj_scale(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, A_invVmmu1_current, mu_current):
    
    p = Rs_inv_current.shape[0]
    n = Rs_inv_current.shape[1]
    
    outsies = np.array([np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)])
    
    
    
    for i in random.permutation(range(n)):
        delta_i = np.sum([Rs_inv_current[j,i,i]*outsies[j] for j in range(p)],axis=0)
        M = delta_i + Dm1_current
        Minv = np.linalg.inv(M)

        b = Dm1Y_current[:,i] + delta_i@mu_current - np.sum([(np.inner(Rs_inv_current[j,i],A_invVmmu1_current[j]) - Rs_inv_current[j,i,i]*A_invVmmu1_current[j,i])*A_inv_current[j] for j in range(p)],axis=0)
        
        V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b
        Vmmu1_current[:,i] = V_current[:,i] - mu_current
        A_invVmmu1_current[:,i] = A_inv_current@Vmmu1_current[:,i]

  
    
    VmY_current = V_current - Y
    VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    

    
    return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)


def V_move_conj_scale_mis(Mis_obs,Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, A_invVmmu1_current, mu_current):
    
    p = Rs_inv_current.shape[0]
    n = Rs_inv_current.shape[1]
    
    outsies = np.array([np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)])
    
    
    
    for i in random.permutation(range(n)):
        delta_i = np.sum([Rs_inv_current[j,i,i]*outsies[j] for j in range(p)],axis=0)
        M = delta_i + Dm1_current * np.diag(Mis_obs[:,i])
        Minv = np.linalg.inv(M)

        b = Dm1Y_current[:,i]*Mis_obs[:,i] + delta_i@mu_current - np.sum([(np.inner(Rs_inv_current[j,i],A_invVmmu1_current[j]) - Rs_inv_current[j,i,i]*A_invVmmu1_current[j,i])*A_inv_current[j] for j in range(p)],axis=0)
        
        V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b
        Vmmu1_current[:,i] = V_current[:,i] - mu_current
        A_invVmmu1_current[:,i] = A_inv_current@Vmmu1_current[:,i]

  
    
    VmY_current = V_current - Y
    VmY_inner_rows_current = np.array([ np.inner(VmY_current[j]*Mis_obs[j], VmY_current[j]) for j in range(p) ])
    

    
    return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)

def V_move_conj_kron(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, mu_current):
    
    p = Rs_inv_current.shape[0]
    n = Rs_inv_current.shape[1]
    
    delta = np.sum([ np.kron( Rs_inv_current[j] , np.outer(A_inv_current[j],A_inv_current[j]) ) for j in range(p) ],axis=0) 
    
    
    
    for i in random.permutation(range(n)):
        delta_i = delta[i*p:(i+1)*p][:,i*p:(i+1)*p]
        M = delta_i + Dm1_current
        Minv = np.linalg.inv(M)
        b = Dm1Y_current[:,i] + delta_i@mu_current - delta[i*p:(i+1)*p]@np.reshape(V_current-np.outer(mu_current,np.ones(n)),n*p,order="F") + delta[i*p:(i+1)*p][:,i*p:(i+1)*p]@(V_current[:,i]-mu_current)
       
        
        V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b

    Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))   
    
    VmY_current = V_current - Y
    VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)


# def V_move_conj_kron2(Rs_inv_current, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, Vmmu1_current, mu_current):
    
#     p = Rs_inv_current.shape[0]
#     n = Rs_inv_current.shape[1]
    
#     delta = np.sum([ np.kron( Rs_inv_current[j] , np.outer(A_inv_current[j],A_inv_current[j]) ) for j in range(p) ],axis=0) 
    
    
    
#     for i in random.permutation(range(n)):
#         delta_i = delta[i*p:(i+1)*p][:,i*p:(i+1)*p]
#         M = delta_i + Dm1_current
#         Minv = np.linalg.inv(M)
#         b = Dm1Y_current[:,i] + delta_i@mu_current - np.sum([delta[i*p:(i+1)*p][:,k*p:(k+1)*p]@(V_current[:,k]-mu_current) for k in range(n) if k != i],axis=0) 
        
#         V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b

#     Vmmu1_current = V_current - np.outer(mu_current,np.ones(n))   
    
#     VmY_current = V_current - Y
#     VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    
#     A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
#     return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)




# def V_move_conj(Rs_inv_current, A_inv_current, Dm1_current, Dm1Y_current, Y, V_current):
    
#     n = Rs_inv_current.shape[1]
#     p = Rs_inv_current.shape[0]
    
#     for i in range(n):
        
#         M = Dm1_current + np.sum([np.outer(Rs_inv_current[j,i,i] * A_inv_current[j],A_inv_current[j]) for j in range(p)],axis=0)
        
        
        
#         b = Dm1Y_current[:,i] - np.sum([np.inner(A_inv_current[j],V_current[:,k])*Rs_inv_current[j,k,i]*A_inv_current[j] for j in range(p) for k in range(n) if k != i],axis=0)
        
#         M_inv = np.linalg.inv(M)
        
#         V_current[:,i] = np.linalg.cholesky(M_inv)@random.normal(size=p) + M_inv@b
        
#     VmY_current = V_current - Y
#     VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    
#     A_invV_current = A_inv_current @ V_current        

#     return(V_current, VmY_current, VmY_inner_rows_current, A_invV_current)


# def V_move_mh(V_current, VmY_inner_rows_current, V_prop, A_invV_current, Rs_inv_current):

#     V_new = V_current + random.normal(size=(p,n)) * V_prop
#     A_invV_new = A_inv_current @ V_new
    
#     VmY_new = V_new - Y
#     VmY_inner_rows_new = np.array([ np.inner(VmY_new[j], VmY_new[j]) for j in range(p) ])
    
#     rat = np.exp(-1/2 * ( np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) + np.sum(taus_current * ( VmY_inner_rows_new - VmY_inner_rows_current ) ) ) )
    
#     if random.uniform() < rat:
        
#         return(V_new, VmY_new, VmY_inner_rows_new, A_invV_new, 1)
#     else:
        
#         return(V_current, VmY_current, VmY_inner_rows_current, A_invV_current, 0)


def taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n):
    
    p = VmY_inner_rows_current.shape[0]
    
    for j in range(p):
        
        taus_current[j] = random.gamma(a + n/2, 1/( b + VmY_inner_rows_current[j]/2), 1)
        
    Dm1_current = np.diag(taus_current)
    Dm1Y_current = Dm1_current @ Y
    
    return(taus_current, Dm1_current, Dm1Y_current)



def taus_move_mis(taus_current,VmY_inner_rows_current,Y,a,b,ns):
    
    p = VmY_inner_rows_current.shape[0]
    
    for j in range(p):
        
        taus_current[j] = random.gamma(a + ns[j]/2, 1/( b + VmY_inner_rows_current[j]/2), 1)
        
    Dm1_current = np.diag(taus_current)
    Dm1Y_current = Dm1_current @ Y
    
    return(taus_current, Dm1_current, Dm1Y_current)



# ### global parameters
# n = 1000
# p = 2


# ### generate random example
# # locs = random.uniform(0,1,(n,2))
# locs = np.linspace(0, 1, n)


# A = np.array([[-1.,1.],
#               [1.,1.]])
# phis = np.array([5.,20.])
# taus_sqrt_inv = np.array([1.,2.]) * 0.1


# # Y, V_true = rNLMC(A,phis,taus_sqrt_inv,locs, retV=True)
# Y, V_true = rNLMC(A,phis,taus_sqrt_inv,np.transpose(np.array([locs])), retV=True)

# ### showcase V and Y

# plt.plot(locs,V_true[0])
# plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
# plt.plot(locs,V_true[1])
# plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)

# plt.show()

# ### priors
# sigma_A = 1.

# min_phi = 3.
# max_phi = 30.
# range_phi = max_phi - min_phi


# alphas = np.linspace(1, p, p)*10
# betas = np.linspace(p, 1, p)*10

# # alphas = np.linspace(2, p+1, p)*5
# # betas = np.linspace(p+1, 2, p)*5

# ### showcase of priors for phis
# from scipy.stats import beta

# for i in range(p):
#     plt.plot(np.linspace(0, 1, 1001),beta.pdf(np.linspace(0, 1, 1001), alphas[i], betas[i]))
# plt.show()


# ## tau

# a = 50
# b = 1




# ### useful quantities 

# # Dists = distance_matrix(locs,locs)
# Dists = distance_matrix(np.transpose(np.array([locs])),np.transpose(np.array([locs])))

# ### init and current state
# phis_current = np.array([5.,20.])
# Rs_current = np.array([ np.exp(-Dists*phis_current[j]) for j in range(p) ])
# Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])

# # V_current = V_true
# # V_current = Y + random.normal(size=(p,n))*0.1
# V_current = random.normal(size=(p,n))*1
# VmY_current = V_current - Y
# VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])

# # A_current = np.array([[-1.,0.],
# #                       [1.,-1.]])
# A_current = np.identity(p)
# A_inv_current = np.linalg.inv(A_current)

# A_invV_current = A_inv_current @ V_current

# taus_current = np.array([1.,1.])
# Dm1_current = np.diag(taus_current)
# Dm1Y_current = Dm1_current @ Y



# ### proposals

# phis_prop = np.linspace(1/p, 1, p) * 2.
# A_prop = 0.02
# # V_prop = 0.005


# ### samples
# N = 4000

# ### global run containers
# phis_run = np.zeros((N,p))
# taus_run = np.zeros((N,p))
# A_run = np.zeros((N,p,p))
# V_run = np.zeros((N,p,n))

# ### acc vector

# acc_phis = np.zeros((p,N))
# acc_A = np.zeros(N)
# # acc_V = np.zeros(N)



# import time
# st = time.time()


# for i in range(N):
    
    
#     V_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current)
        
        
    
        
        
#     A_current, A_inv_current, A_invV_current, acc_A[i] = A_move(A_current,A_inv_current,A_invV_current,A_prop,sigma_A,V_current,Rs_inv_current)
    
#     phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_current,Dists,A_invV_current,Rs_current,Rs_inv_current)

#     taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
    
#     V_run[i] = V_current
#     taus_run[i] = taus_current
#     phis_run[i] =  phis_current
#     A_run[i] = A_current
    
#     if i % 100 == 0:
#         print(i)

# et = time.time()
# print('Execution time:', (et-st)/60, 'minutes')


# tail = 2000

# print('accept phi_1:',np.mean(acc_phis[0,tail:]))
# print('accept phi_2:',np.mean(acc_phis[1,tail:]))
# # print('accept phi_3:',np.mean(acc_phis[2,tail:]))

# plt.plot(phis_run[:,0])
# plt.plot(phis_run[:,1])
# # plt.plot(phis_run[:,2])
# plt.show()

# print('mean phi_1:',np.mean(phis_run[tail:,0]))
# print('mean phi_2:',np.mean(phis_run[tail:,1]))
# # print('mean phi_3:',np.mean(phis_run[tail:,2]))

# print('accept A:',np.mean(acc_A[tail:]))
# print('mean A:',np.mean(A_run[tail:],axis=0))

# plt.plot(A_run[:,0,0])
# plt.plot(A_run[:,0,1])
# plt.plot(A_run[:,1,0])
# plt.plot(A_run[:,1,1])
# plt.show()


# # print('accept V:',np.mean(acc_V[tail:]))

# print('mean tau_1:',np.mean(taus_run[tail:,0]))
# print('mean tau_2:',np.mean(taus_run[tail:,1]))

# print('real taus:',taus_sqrt_inv ** (-2))

# print('mean inv sqrt tau_1:',np.mean(1/np.sqrt(taus_run[tail:,0])))
# print('mean inv sqrt tau_2:',np.mean(1/np.sqrt(taus_run[tail:,1])))

# print('real sqrt inv taus:',taus_sqrt_inv)

# plt.plot(taus_run[:,0])
# plt.plot(taus_run[:,1])
# # plt.plot(1/np.sqrt(taus_run[:,2]))
# plt.show()


# plt.plot(1/np.sqrt(taus_run[:,0]))
# plt.plot(1/np.sqrt(taus_run[:,1]))
# # plt.plot(1/np.sqrt(taus_run[:,2]))
# plt.show()

# for i in range(N):
#     if i % 100 == 0:
#         plt.plot(locs,V_run[i,0])
#         plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
#         plt.plot(locs,V_run[i,1])
#         plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)

#         plt.show()








