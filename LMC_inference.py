




import numpy as np
from numpy import random

from LMC_generation import rLMC

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix


def A_move(A_current,A_inv_current,A_invV_current,A_prop,sigma_A,V,Rs_inv_current):

    
    p = V.shape[0] 
    n = V.shape[1]

    A_new = A_current + A_prop*random.normal(size=(p,p))
    
    A_inv_new = np.linalg.inv(A_new)
    
    A_invV_new = A_inv_new @ V
    
    rat = np.exp( -1/2 * np.sum( [ A_invV_new[j] @ Rs_inv_current[j] @ A_invV_new[j] - A_invV_current[j] @ Rs_inv_current[j] @ A_invV_current[j] for j in range(p) ] ) ) * np.abs(np.linalg.det(A_inv_new @ A_current))**n * np.exp(-1/2/sigma_A**2 * (np.sum(A_new**2) - np.sum(A_current**2)))
    
    if random.uniform() < rat:
        
        return(A_new,A_inv_new,A_invV_new,1)
    else:
        
        return(A_current,A_inv_current,A_invV_current,0)


def phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V,D,A_invV_current,Rs_current,Rs_inv_current):
    
    p = phis_current.shape[0]
    range_phi = max_phi - min_phi
    
    acc_phis = np.zeros(p)
    
    for j in range(p):
        
        phis_new = phis_current[j] + phis_prop[j]*random.normal()
        
        if (phis_new > min_phi)  &  (phis_new < max_phi):
            
            Rs_new = np.exp(-D*phis_new)
            Rs_inv_new = np.linalg.inv(Rs_new)
            
            phis_new_star_j = (phis_new - min_phi)/range_phi
            phis_current_star_j = (phis_current[j] - min_phi)/range_phi
            
            
            rat = np.exp( -1/2 * ( A_invV_current[j] @ ( Rs_inv_new - Rs_inv_current[j] ) @ A_invV_current[j] ) ) * np.linalg.det( Rs_inv_new @ Rs_current[j] ) **(1/2) * (phis_new_star_j/phis_current_star_j)**(alphas[j]-1) * ((1-phis_new_star_j)/(1-phis_current_star_j))**(betas[j]-1)                             
            
            
            if random.uniform() < rat:
                phis_current[j] = phis_new
                Rs_current[j] = Rs_new
                Rs_inv_current[j] = Rs_inv_new
                
                acc_phis[j] = 1
                
    return(phis_current,Rs_current,Rs_inv_current,acc_phis)
### global parameters
n = 1000
p = 2


### generate random example
locs = random.uniform(0,1,(n,2))


A = np.array([[-1.,0.],
              [1.,-1.]])
phis = np.array([5.,25.])


# ### 1D case
# locs = np.linspace(0, 1, 1001)

# A = np.array([[1.]])
# phis = np.array([5.])


V = rLMC(A,phis,locs)





### priors
sigma_A = 1.

min_phi = 3.
max_phi = 30.
range_phi = max_phi - min_phi


alphas = np.linspace(1, p, p)*10
betas = np.linspace(p, 1, p)*10

# alphas = np.linspace(2, p+1, p)*5
# betas = np.linspace(p+1, 2, p)*5

### showcase of priors for phis
from scipy.stats import beta

for i in range(p):
    plt.plot(np.linspace(0, 1, 1001),beta.pdf(np.linspace(0, 1, 1001), alphas[i], betas[i]))
plt.show()



### useful quantities 

D = distance_matrix(locs,locs)
# D = distance_matrix(np.transpose(np.array([locs])),np.transpose(np.array([locs])))

### init and current state
phis_current = np.array([5.,25.])
Rs_current = np.array([ np.exp(-D*phis_current[j]) for j in range(p) ])
Rs_inv_current = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


A_current = np.identity(p)
A_inv_current = np.linalg.inv(A_current)

A_invV_current = A_inv_current @ V

### new state containers
phis_new = np.array([5.,25.])
Rs_new = np.array([ np.exp(-D*phis_current[j]) for j in range(p) ])
Rs_inv_new = np.array([ np.linalg.inv(Rs_current[j]) for j in range(p) ])


A_new = np.identity(p)
A_inv_new = np.linalg.inv(A_current)

A_invV_new = A_inv_current @ V

### proposals

phis_prop = np.linspace(1/p, 1, p) * 2.
A_prop = 0.02


### samples
N = 2000

### global run containers
phis_run = np.zeros((N,p))
A_run = np.zeros((N,p,p))

### acc vector

acc_phis = np.zeros((p,N))
acc_A = np.zeros(N)



import time
st = time.time()


for i in range(N):
    
    
    

    A_current, A_inv_current, A_invV_current, acc_A[i] = A_move(A_current,A_inv_current,A_invV_current,A_prop,sigma_A,V,Rs_inv_current)
    
    

    phis_current, Rs_current, Rs_inv_current, acc_phis[:,i] = phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,V,D,A_invV_current,Rs_current,Rs_inv_current)

    phis_run[i] =  phis_current
    A_run[i] = A_current
    
    if i % 100 == 0:
        print(i)

et = time.time()
print('Execution time:', (et-st)/60, 'minutes')

tail = 1000

print('accept phi_1:',np.mean(acc_phis[0,tail:]))
print('accept phi_2:',np.mean(acc_phis[1,tail:]))
# print('accept phi_3:',np.mean(acc_phis[2,tail:]))

plt.plot(phis_run[:,0])
plt.plot(phis_run[:,1])
# plt.plot(phis_run[:,2])
plt.show

print('mean phi_1:',np.mean(phis_run[tail:,0]))
print('mean phi_2:',np.mean(phis_run[tail:,1]))
# print('mean phi_3:',np.mean(phis_run[tail:,2]))

print('accept A:',np.mean(acc_A[tail:]))
print('mean A:',np.mean(A_run[tail:],axis=0))





