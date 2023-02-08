import numpy as np
from numpy import random
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

### number of iterations
N = 10000

### number of points
n = 1000

### number of neighbors
m = 50

### container of results
maxes_rand = np.zeros(N)
maxes_axis = np.zeros(N)
maxes_sum = np.zeros(N)

# for x in np.arange(N):

#     ### generate points
#     locs = random.uniform(size=(n,2))
    
#     ### distance matrix
#     D = distance_matrix(locs,locs)
    
    
#     ### compute neighbors
#     neighbors = np.array( [ np.argsort(D[i])[1:(m+1)] for i in range(n) ] )
    
#     ### sum occurences and take max
#     maxes[x] = np.max(np.array( [ np.sum(neighbors == j) for j in range(n) ] ))


# ### showcase

# # Creating plot
# plt.boxplot(maxes)
 
# # show plot
# plt.show()

import time
st = time.time()

### with ordering

for x in np.arange(N):

    ### generate points
    locs = random.uniform(size=(n,2))
    
    ### distance matrix
    D = distance_matrix(locs,locs)
    
    
    ### compute neighbors
    neighbors = np.ones((n,m)) * n
    
    for i in np.arange(m):
    
        neighbors[i,:i] = np.argsort(D[i,:i])
    
    for i in np.arange(m,n):
    
        neighbors[i] = np.argsort(D[i,:i])[:m]


    ### sum occurences and take max
    maxes_rand[x] = np.max(np.array( [ np.sum(neighbors == j) for j in range(n) ] ))





for x in np.arange(N):

    ### generate points
    locs = random.uniform(size=(n,2))
    
    ### change order
    locs = locs[np.argsort(locs[:,0])]
    
    ### distance matrix
    D = distance_matrix(locs,locs)
    
    
    ### compute neighbors
    neighbors = np.ones((n,m)) * n
    
    for i in np.arange(m):
    
        neighbors[i,:i] = np.argsort(D[i,:i])
    
    for i in np.arange(m,n):
    
        neighbors[i] = np.argsort(D[i,:i])[:m]


    ### sum occurences and take max
    maxes_axis[x] = np.max(np.array( [ np.sum(neighbors == j) for j in range(n) ] ))




for x in np.arange(N):

    ### generate points
    locs = random.uniform(size=(n,2))
    
    ### change order
    locs = locs[np.argsort(locs[:,0]+locs[:,1])]
    
    ### distance matrix
    D = distance_matrix(locs,locs)
    
    
    ### compute neighbors
    neighbors = np.ones((n,m)) * n
    
    for i in np.arange(m):
    
        neighbors[i,:i] = np.argsort(D[i,:i])
    
    for i in np.arange(m,n):
    
        neighbors[i] = np.argsort(D[i,:i])[:m]


    ### sum occurences and take max
    maxes_sum[x] = np.max(np.array( [ np.sum(neighbors == j) for j in range(n) ] ))



et = time.time()
print('Execution time:', (et-st)/60, 'minutes')

### showcase

# Creating plot
plt.boxplot([maxes_rand,maxes_axis,maxes_sum])
 
# show plot
plt.show()