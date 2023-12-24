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
plt.plot([0,6],[0,0], linestyle='dotted',c="black")
bplot = ax.boxplot(my_dict.values(),patch_artist=True)
for patch in bplot['boxes']:
        patch.set_facecolor("white")
ax.set_xticklabels(my_dict.keys())
plt.xlim([0.5, 5.5])
plt.title("Difference in RMSE")
plt.savefig("full_diff.pdf", bbox_inches='tight')
# plt.show()

d=0.1
n_comp_med = np.median(n_comps_glob,axis=2)[:,0]
n_comp_05 = np.quantile(n_comps_glob,0.05,axis=2)[:,0]
n_comp_95 = np.quantile(n_comps_glob,0.95,axis=2)[:,0]
xs = [2,3,4,5,6]
plt.scatter(xs,n_comp_med,c="black")
for i in range(5):
    plt.plot([xs[i],xs[i]],[n_comp_05[i],n_comp_95[i]],c="black")
    plt.plot([xs[i]-0.1,xs[i]+0.1],[n_comp_05[i],n_comp_05[i]],c="black")
    plt.plot([xs[i]-0.1,xs[i]+0.1],[n_comp_95[i],n_comp_95[i]],c="black")
plt.plot(xs,n_comp_med)
plt.xticks(xs)
plt.title("Number of Non-Zero Components")
plt.savefig("full_ncomp.pdf", bbox_inches='tight')
# plt.show()

# my_dict = {'2': n_comps_glob[0,0], '3': n_comps_glob[1,0], '4': n_comps_glob[2,0], '5': n_comps_glob[3,0], '6': n_comps_glob[4,0]}

# fig, ax = plt.subplots()
# ax.boxplot(my_dict.values())
# ax.set_xticklabels(my_dict.keys())
# plt.title("Number of Non-Zero Components")
# plt.savefig("full_ncomp.pdf", bbox_inches='tight')
# # plt.show()


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
plt.plot([0,6],[0,0], linestyle='dotted',c="black")
bplot = ax.boxplot(my_dict.values(),patch_artist=True)
for patch in bplot['boxes']:
        patch.set_facecolor("white")
ax.set_xticklabels(my_dict.keys())
plt.xlim([0.5, 5.5])
plt.title("Difference in RMSE")
plt.savefig("diag_diff.pdf", bbox_inches='tight')
# plt.show()

d=0.1
n_comp_med = np.median(n_comps_glob,axis=2)[:,2]
n_comp_05 = np.quantile(n_comps_glob,0.05,axis=2)[:,2]
n_comp_95 = np.quantile(n_comps_glob,0.95,axis=2)[:,2]
xs = [2,3,4,5,6]
plt.scatter(xs,n_comp_med,c="black")
for i in range(5):
    plt.plot([xs[i],xs[i]],[n_comp_05[i],n_comp_95[i]],c="black")
    plt.plot([xs[i]-0.1,xs[i]+0.1],[n_comp_05[i],n_comp_05[i]],c="black")
    plt.plot([xs[i]-0.1,xs[i]+0.1],[n_comp_95[i],n_comp_95[i]],c="black")
plt.plot(xs,n_comp_med)
plt.xticks(xs)
plt.title("Number of Non-Zero Components")
plt.savefig("diag_ncomp.pdf", bbox_inches='tight')
# plt.show()

# my_dict = {'2': n_comps_glob[0,2], '3': n_comps_glob[1,2], '4': n_comps_glob[2,2], '5': n_comps_glob[3,2], '6': n_comps_glob[4,2]}

# fig, ax = plt.subplots()
# ax.boxplot(my_dict.values())
# ax.set_xticklabels(my_dict.keys())
# plt.title("Number of Non-Zero Components")
# plt.savefig("diag_ncomp.pdf", bbox_inches='tight')
# # plt.show()


#### TIMES

conc = 2
noise_sd = 0.5

mean_times = np.zeros((5,3,2))

p = 2
prob_one = 1/p


TIMES = np.load("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
TIMES = np.median(TIMES,axis=2)

mean_times[0] = TIMES


p = 3
prob_one = 1/p


TIMES = np.load("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
TIMES = np.median(TIMES,axis=2)

mean_times[1] = TIMES


p = 4
prob_one = 1/p


TIMES = np.load("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
TIMES = np.median(TIMES,axis=2)

mean_times[2] = TIMES


p = 5
prob_one = 1/p


TIMES = np.load("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
TIMES = np.median(TIMES,axis=2)

mean_times[3] = TIMES


p = 6
prob_one = 1/p


TIMES = np.load("1mtimes_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
TIMES = np.median(TIMES,axis=2)

mean_times[4] = TIMES
# np.mean(mean_times,axis=1)



# plt.plot([2,3,4,5,6],mean_times[:,0,0])
# plt.plot([2,3,4,5,6],mean_times[:,0,1])
# plt.show()

# plt.plot([2,3,4,5,6],mean_times[:,2,0])
# plt.plot([2,3,4,5,6],mean_times[:,2,1])
# plt.show()






