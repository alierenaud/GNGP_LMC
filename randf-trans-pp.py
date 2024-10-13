#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:16:40 2024

@author: homeboy
"""

import numpy as np
from numpy import random

from scipy.spatial import distance_matrix
from base import makeGrid, vec_inv

from scipy.stats import norm

import matplotlib.pyplot as plt

random.seed(10)

lam = 1000
n_obs = random.poisson(lam)
locs = random.uniform(size=(n_obs,2))

n_grid=50
marg_grid = np.linspace(0,1,n_grid+1)
loc_grid = makeGrid(marg_grid, marg_grid)

locs_tot = np.concatenate((locs,loc_grid),axis=0)



D_tot = distance_matrix(locs_tot,locs_tot)

phi = 10
R_tot = np.exp(-D_tot*phi)
A_tot = np.linalg.cholesky(R_tot)


n_tot = A_tot.shape[0]

V = A_tot@random.normal(size=n_tot)

V_obs = V[:n_obs]
V_grid = V[n_obs:]


###### RF plots

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(V_grid,n_grid+1)), cmap = "Blues")
# ax.scatter(X_obs[Y_obs==i+1,0],X_obs[Y_obs==i+1,1],c="black",s=20)
plt.colorbar(c)
# plt.savefig("zrf.pdf", bbox_inches='tight')
plt.show()


### intensity plot

Lam_grid = lam*norm.cdf(V_grid)

xv, yv = np.meshgrid(marg_grid, marg_grid)



fig, ax = plt.subplots()
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
ax.set_box_aspect(1)



c = ax.pcolormesh(xv, yv, np.transpose(vec_inv(Lam_grid,n_grid+1)), cmap = "Blues")
# ax.scatter(X_obs[Y_obs==i+1,0],X_obs[Y_obs==i+1,1],c="black",s=20)
plt.colorbar(c)
# plt.savefig("zint.pdf", bbox_inches='tight')
plt.show()

### point process

Y_obs = V_obs+random.normal(size=n_obs)
X_obs = locs[Y_obs>0]



fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_box_aspect(1)
ax.scatter(X_obs[:,0],X_obs[:,1])
# plt.savefig("zpoints.pdf", bbox_inches='tight')
plt.show()



