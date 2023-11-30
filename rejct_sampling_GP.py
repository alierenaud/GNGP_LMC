#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:34:15 2023

@author: homeboy
"""


import numpy as np
from numpy import random
from scipy.spatial import distance_matrix

from base import matern_kernel, fct

import matplotlib.pyplot as plt

from scipy.stats import norm

random.seed(10)

n_1 = 100

x_current = random.uniform(size=n_1)*20 - 10


phi = 1
a = 1

D_current = distance_matrix(np.transpose([x_current]),np.transpose([x_current]))
R_current = matern_kernel(D_current,phi)

R_inv_current = np.linalg.inv(R_current)


g_current = np.linalg.cholesky(R_current/a) @ random.normal(size=n_1)
y_current = g_current + random.normal(size=n_1)




# ### showcase g

# print(g)


# locs = np.sort(x_1_current)
# plt.plot(locs,g)
# plt.show()

n_0_current = np.sum(y_current<0)
x_0_current = x_current[y_current<0]
g_0_current = g_current[y_current<0]
y_0_current = y_current[y_current<0]


n_1_current = np.sum(y_current>0)
x_1_current = x_current[y_current>0]
g_1_current = g_current[y_current>0]
y_1_current = y_current[y_current>0]


while n_1_current < n_1:
    
    
    
    x_new = random.uniform()*20 - 10
    
    D_new_current = distance_matrix(np.transpose([[x_new]]),np.transpose([x_current]))
    
    R_new_current = matern_kernel(D_new_current,phi)
    
    b_temp = R_new_current@R_inv_current
    
    r_temp = 1-b_temp@np.transpose(R_new_current)
    
    if r_temp < 0:
        print("wtf")
        break
    
    g_new = np.sqrt(r_temp/a)*random.normal() + np.inner(b_temp,g_current)
    
    y_new = g_new + random.normal()
    
    x_current = np.append(x_current,x_new)
    g_current = np.append(g_current,g_new)
    y_current = np.append(y_current,y_new)
    
    R_current = np.block([[R_current,np.transpose(R_new_current)],[R_new_current,1]])
    
    
    DD = 1/r_temp
    BB = -DD*b_temp
    AA = R_inv_current - np.outer(b_temp, BB)
    
    R_inv_current = np.block([[AA,np.transpose(BB)],[BB,DD]])
    
    # print(R_current@R_inv_current)
    ind = np.argsort(x_current)
    plt.plot(x_current[ind],g_current[ind])
    plt.show()
    
    if y_new > 0:
        n_1_current += 1



n_0_current = np.sum(y_current<0)
x_0_current = x_current[y_current<0]
g_0_current = g_current[y_current<0]
y_0_current = y_current[y_current<0]


n_1 = np.sum(y_current>0)
x_1 = x_current[y_current>0]
g_1_current = g_current[y_current>0]
y_1_current = y_current[y_current>0]


ind = np.argsort(x_current)
plt.plot(x_current[ind],norm.cdf(g_current[ind]))
plt.show()







