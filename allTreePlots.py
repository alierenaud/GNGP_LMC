# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:15:59 2024

@author: alier
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

random.seed(0)

tab_cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


maple = np.loadtxt("maple.csv", delimiter=",")
hickory = np.loadtxt("hickory.csv", delimiter=",")
whiteoak = np.loadtxt("whiteoak.csv", delimiter=",")
redoak = np.loadtxt("redoak.csv", delimiter=",")
blackoak = np.loadtxt("blackoak.csv", delimiter=",")

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]
n_whiteoak = whiteoak.shape[0]
n_redoak = redoak.shape[0]
n_blackoak = blackoak.shape[0]

# X_obs = np.concatenate((maple,hickory,whiteoak,redoak,blackoak))
# Y_obs = np.concatenate((np.ones(n_maple,dtype=int)*1,np.ones(n_hickory,dtype=int)*2,np.ones(n_whiteoak,dtype=int)*3,np.ones(n_redoak,dtype=int)*4,np.ones(n_blackoak,dtype=int)*5))

X_obs = np.concatenate((maple,hickory))
Y_obs = np.concatenate((np.ones(n_maple,dtype=int)*1,np.ones(n_hickory,dtype=int)*2))


fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)

s = 20

ax.scatter(maple[:,0],maple[:,1],s=s,c=tab_cols[0])
ax.scatter(hickory[:,0],hickory[:,1],s=s,c=tab_cols[1])
# ax.scatter(whiteoak[:,0],whiteoak[:,1],s=s,c=tab_cols[2])
# ax.scatter(redoak[:,0],redoak[:,1],s=s,c=tab_cols[3])
# ax.scatter(blackoak[:,0],blackoak[:,1],s=s,c=tab_cols[4])

for i in random.permutation(X_obs.shape[0]):

    ax.scatter(X_obs[i,0],X_obs[i,1],s=s,c=tab_cols[Y_obs[i]-1])


plt.legend(["maple", "hickory"], bbox_to_anchor=(1,0.8)) 
# plt.legend(["maple", "hickory","whitoak","redoak","blackoak"], bbox_to_anchor=(1,0.8)) 
# plt.savefig("Tree2.pdf", format="pdf", bbox_inches="tight")
plt.show()
    
# ax.scatter(maple[:,0],maple[:,1],s=s,c=tab_cols[0])
# ax.scatter(hickory[:,0],hickory[:,1],s=s,c=tab_cols[1])
# ax.scatter(whiteoak[:,0],whiteoak[:,1],s=s,c=tab_cols[2])
# ax.scatter(redoak[:,0],redoak[:,1],s=s,c=tab_cols[3])
# ax.scatter(blackoak[:,0],blackoak[:,1],s=s,c=tab_cols[4])
# plt.show()