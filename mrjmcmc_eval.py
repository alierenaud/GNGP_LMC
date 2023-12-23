#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 20:07:13 2023

@author: homeboy
"""

import numpy as np
import matplotlib.pyplot as plt


dMSEs_glob = np.zeros((5,3,100))
n_comps_glob = np.zeros((5,3,100))

conc = 2
noise_sd = 0.5


p = 2
prob_one = 1/p


SEs = np.load("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


MSEs = np.mean(SEs, axis=(3,4,5))
dMSEs = np.sqrt(MSEs[:,0]) - np.sqrt(MSEs[:,1])

mn_comps = np.mean(n_comps,axis=2)

dMSEs_glob[0] = dMSEs
n_comps_glob[0] = mn_comps


p = 3
prob_one = 1/p


SEs = np.load("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


MSEs = np.mean(SEs, axis=(3,4,5))
dMSEs = np.sqrt(MSEs[:,0]) - np.sqrt(MSEs[:,1])

mn_comps = np.mean(n_comps,axis=2)

dMSEs_glob[1] = dMSEs
n_comps_glob[1] = mn_comps


p = 4
prob_one = 1/p


SEs = np.load("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


MSEs = np.mean(SEs, axis=(3,4,5))
dMSEs = np.sqrt(MSEs[:,0]) - np.sqrt(MSEs[:,1])

mn_comps = np.mean(n_comps,axis=2)

dMSEs_glob[2] = dMSEs
n_comps_glob[2] = mn_comps


p = 5
prob_one = 1/p


SEs = np.load("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


MSEs = np.mean(SEs, axis=(3,4,5))
dMSEs = np.sqrt(MSEs[:,0]) - np.sqrt(MSEs[:,1])

mn_comps = np.mean(n_comps,axis=2)

dMSEs_glob[3] = dMSEs
n_comps_glob[3] = mn_comps


p = 6
prob_one = 1/p


SEs = np.load("1mMSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1mn_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


MSEs = np.mean(SEs, axis=(3,4,5))
dMSEs = np.sqrt(MSEs[:,0]) - np.sqrt(MSEs[:,1])

mn_comps = np.mean(n_comps,axis=2)

dMSEs_glob[4] = dMSEs
n_comps_glob[4] = mn_comps




my_dict = {'2': dMSEs_glob[0,0], '3': dMSEs_glob[1,0], '4': dMSEs_glob[2,0], '5': dMSEs_glob[3,0], '6': dMSEs_glob[4,0]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("Difference in RMSE")
plt.savefig("full_diff.pdf", bbox_inches='tight')
# plt.show()

my_dict = {'2': n_comps_glob[0,0], '3': n_comps_glob[1,0], '4': n_comps_glob[2,0], '5': n_comps_glob[3,0], '6': n_comps_glob[4,0]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("Number of Non-Zero Components")
plt.savefig("full_ncomp.pdf", bbox_inches='tight')
# plt.show()


# my_dict = {'2': dMSEs_glob[0,1], '3': dMSEs_glob[1,1], '4': dMSEs_glob[2,1], '5': dMSEs_glob[3,1], '6': dMSEs_glob[4,1]}

# fig, ax = plt.subplots()
# ax.boxplot(my_dict.values())
# ax.set_xticklabels(my_dict.keys())
# plt.title("Difference in RMSE")
# plt.show()

# my_dict = {'2': n_comps_glob[0,1], '3': n_comps_glob[1,1], '4': n_comps_glob[2,1], '5': n_comps_glob[3,1], '6': n_comps_glob[4,1]}

# fig, ax = plt.subplots()
# ax.boxplot(my_dict.values())
# ax.set_xticklabels(my_dict.keys())
# plt.title("Number of Non-Zero Components")
# plt.show()


my_dict = {'2': dMSEs_glob[0,2], '3': dMSEs_glob[1,2], '4': dMSEs_glob[2,2], '5': dMSEs_glob[3,2], '6': dMSEs_glob[4,2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("Difference in RMSE")
plt.savefig("diag_diff.pdf", bbox_inches='tight')
# plt.show()

my_dict = {'2': n_comps_glob[0,2], '3': n_comps_glob[1,2], '4': n_comps_glob[2,2], '5': n_comps_glob[3,2], '6': n_comps_glob[4,2]}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.title("Number of Non-Zero Components")
plt.savefig("diag_ncomp.pdf", bbox_inches='tight')
# plt.show()


