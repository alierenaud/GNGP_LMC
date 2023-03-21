

import numpy as np
from numpy import random

from noisyLMC_generation import rNLMC

from LMC_inference import phis_move
from noisyLMC_inference import V_move_conj, taus_move

import matplotlib.pyplot as plt

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
    
    if np.prod(A_new[0]>0):
        A_inv_new = np.linalg.inv(A_new)
        
        A_invV_new = A_inv_new @ V
        
        rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n * np.exp(-1/2/sigma_A**2 * (np.sum((A_new-mu_A)**2) - np.sum((A_current-mu_A)**2)))
        
        if random.uniform() < rat:
            
            return(A_new,A_inv_new,A_invV_new,1)
        else:
            
            return(A_current,A_inv_current,A_invV_current,0)
    else:
        
        return(A_current,A_inv_current,A_invV_current,0)

def A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A):

    
    p = A_current.shape[0] 
    n = A_invV_current.shape[1]
    
    ### threshold
    z =  -1/2 * np.sum( [A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) - n * np.log( np.abs(np.linalg.det(A_current))) - 1/2/sigma_A**2 * np.sum((A_current-mu_A)**2) - random.exponential(1,1)
    
    L = A_current - random.uniform(0,sigma_slice,(p,p))
    L[0] = np.maximum(L[0],0)
    
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
    
    while not np.prod(A_current[0]>0):
        A_current = vec_inv( np.linalg.cholesky(M_inv) @ random.normal(size=p**2) + M_inv @ b1, p)
    A_inv_current = np.linalg.inv(A_current)
    
    ### update V
    V_current = A_current @ A_invV_current
    
    return(A_current,A_inv_current,V_current)

# A = np.array([[1,2,7],[3,4,8],[5,6,9],[10,11,12]])
# nrow = A.shape[0]
# print(A)
# print(vec(A))
# print(vec_inv(vec(A),nrow))



### global parameters
n = 1000
p = 2


### generate random example
# locs = random.uniform(0,1,(n,2))
locs = np.linspace(0, 1, n)


A = np.array([[-1.,1.],
              [1.,1.]])
phis = np.array([5.,20.])
taus_sqrt_inv = np.array([1.,1.]) * 1


# Y, V_true = rNLMC(A,phis,taus_sqrt_inv,locs, retV=True)
Y, V_true = rNLMC(A,phis,taus_sqrt_inv,np.transpose(np.array([locs])), retV=True)

### showcase V and Y

plt.plot(locs,V_true[0])
plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
plt.plot(locs,V_true[1])
plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)

plt.show()

### priors
sigma_A = 1.
mu_A = np.array([[sigma_A,sigma_A],
                 [0.,0.]])

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.linspace(1, p, p)*10
betas = np.linspace(p, 1, p)*10

prior_means = alphas/(alphas+betas) * range_phi + min_phi

# alphas = np.linspace(2, p+1, p)*5
# betas = np.linspace(p+1, 2, p)*5

### showcase of priors for phis
from scipy.stats import beta

for i in range(p):
    plt.plot(np.linspace(0, 1, 1001)*range_phi + min_phi,beta.pdf(np.linspace(0, 1, 1001), alphas[i], betas[i]))
plt.show()



## tau

a = 50
b = 1




### useful quantities 

# Dists = distance_matrix(locs,locs)
Dists = distance_matrix(np.transpose(np.array([locs])),np.transpose(np.array([locs])))

### init and current state
phis_current = np.array([5.,20.])
Rs_current = np.array([ np.exp(-Dists*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])

# V_current = V_true
# V_current = Y + random.normal(size=(p,n))*0.1
V_current = random.normal(size=(p,n))*1
VmY_current = V_current - Y
VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])

A_current = np.array([[1.,1.],
              [0.,1.]])
# A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)

A_invV_current = A_inv_current @ V_current

taus_current = 1/(np.array([1.,1.]) * 1)**2
Dm1_current = np.diag(taus_current)
Dm1Y_current = Dm1_current @ Y



### proposals

phis_prop = np.linspace(1/p, 1, p) * 2.
A_prop = 0.03
sigma_slice = 1
# V_prop = 0.005


### samples
N = 10000

### global run containers
phis_run = np.zeros((N,p))
taus_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))
V_run = np.zeros((N,p,n))

### acc vector

acc_phis = np.zeros((p,N))
acc_A = np.zeros(N)
# acc_V = np.zeros(N)



import time
st = time.time()


for i in range(N):
    
    
    V_current, VmY_current, VmY_inner_rows_current, A_invV_current = V_move_conj(Rs_inv_current, A_inv_current, taus_current, Dm1Y_current, Y, V_current)
        
    
    
                        
    
    A_current, A_inv_current, A_invV_current = A_move_slice(A_current, A_invV_current, Rs_inv_current, V_current, sigma_A, mu_A)
    
    # A_current, A_inv_current, V_current = A_move_white(A_invV_current,Dm1_current,Dm1Y_current,sigma_A,mu_A) 
    
    
    
    

    
    
    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V_current,Dists,A_invV_current,Rs_current,Rs_inv_current)

    taus_current, Dm1_current, Dm1Y_current = taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n)
    
    V_run[i] = V_current
    taus_run[i] = taus_current
    phis_run[i] =  phis_current
    A_run[i] = A_current
    
    if i % 100 == 0:
        print(i)

et = time.time()
print('Execution time:', (et-st)/60, 'minutes')

print("Prior Means for Ranges", alphas / (alphas + betas) * range_phi + min_phi)

tail = 2000

print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))
# print('accept phi_3:',np.mean(acc_phis[2,tail:]))

plt.plot(phis_run[:,0])
plt.plot(phis_run[:,1])
# plt.plot(phis_run[:,2])
plt.show()

print('mean phi_1:',np.mean(phis_run[tail:,0]))
print('mean phi_2:',np.mean(phis_run[tail:,1]))
# print('mean phi_3:',np.mean(phis_run[tail:,2]))

print('accept A:',np.mean(acc_A[tail:]))
print('mean A:',np.mean(A_run[tail:],axis=0))

plt.plot(A_run[:,0,0])
plt.plot(A_run[:,0,1])
plt.plot(A_run[:,1,0])
plt.plot(A_run[:,1,1])
plt.show()


# print('accept V:',np.mean(acc_V[tail:]))

print('mean tau_1:',np.mean(taus_run[tail:,0]))
print('mean tau_2:',np.mean(taus_run[tail:,1]))

print('real taus:',taus_sqrt_inv ** (-2))

print('mean inv sqrt tau_1:',np.mean(1/np.sqrt(taus_run[tail:,0])))
print('mean inv sqrt tau_2:',np.mean(1/np.sqrt(taus_run[tail:,1])))

print('real sqrt inv taus:',taus_sqrt_inv)

plt.plot(taus_run[:,0])
plt.plot(taus_run[:,1])
# plt.plot(1/np.sqrt(taus_run[:,2]))
plt.show()


plt.plot(1/np.sqrt(taus_run[:,0]))
plt.plot(1/np.sqrt(taus_run[:,1]))
# plt.plot(1/np.sqrt(taus_run[:,2]))
plt.show()

for i in range(N):
    if i % 1000 == 0:
        plt.plot(locs,V_run[i,0])
        plt.plot(locs,Y[0], '.', c="tab:blue", alpha=0.5)
        plt.plot(locs,V_run[i,1])
        plt.plot(locs,Y[1], '.', c="tab:orange", alpha=0.5)

        plt.show()

plt.plot(A_run[tail:,0,0])
plt.show()
plt.plot(A_run[tail:,0,1])
plt.show()
plt.plot(A_run[tail:,1,0])
plt.show()
plt.plot(A_run[tail:,1,1])
plt.show()


arr = np.zeros((N-tail,p**2))

arr[:,0] = A_run[tail:,0,0]
arr[:,1] = A_run[tail:,0,1]
arr[:,2] = A_run[tail:,1,0]
arr[:,3] = A_run[tail:,1,1]

np.savetxt('output.csv', arr, delimiter=',')
