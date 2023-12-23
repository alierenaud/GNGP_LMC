#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:03:27 2023

@author: homeboy
"""

import numpy as np
import matplotlib.pyplot as plt

### indices = quant,A,method,p
Results = np.zeros((4,3,2,4))
Results_05 = np.zeros((4,3,2,4))
Results_95 = np.zeros((4,3,2,4))


conc = 2
noise_sd = 0.5
p = 2
prob_one = 1/p

times = np.load("1times_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
wnorms = np.load("1Wnorms"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
MSEs = np.load("1MSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1n_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


times.shape
times_med = np.median(times,axis=2)
times_05 = np.quantile(times, 0.05, axis=2)
times_95 = np.quantile(times, 0.95, axis=2)

n_comps.shape
n_comps_mean = np.mean(n_comps,axis=2)
n_comps_med = np.median(n_comps_mean,axis=1)
n_comps_med = np.transpose(np.append([n_comps_med], [[p**2,p**2,p**2]],axis=0))
n_comps_05 = np.quantile(n_comps_mean, 0.05, axis=1)
n_comps_05 = np.transpose(np.append([n_comps_05], [[p**2,p**2,p**2]],axis=0))
n_comps_95 = np.quantile(n_comps_mean, 0.95, axis=1)
n_comps_95 = np.transpose(np.append([n_comps_95], [[p**2,p**2,p**2]],axis=0))


wnorms.shape
wnorms_mean = np.mean(wnorms,axis=(3))
wnorms_mean.shape
wnorms_med = np.median(wnorms_mean,axis=2)
wnorms_05 = np.quantile(wnorms_mean, 0.05, axis=2)
wnorms_95 = np.quantile(wnorms_mean, 0.95, axis=2)

MSEs.shape
MSEs_mean = np.mean(MSEs, axis=(3,4,5))
MSEs_mean.shape
MSEs_med = np.median(MSEs_mean, axis=2)
MSEs_05 = np.quantile(MSEs_mean, 0.05, axis=2)
MSEs_95 = np.quantile(MSEs_mean, 0.95, axis=2)

Results[:,:,:,0] = np.array([times_med,n_comps_med,wnorms_med,MSEs_med])
Results_05[:,:,:,0] = np.array([times_05,n_comps_05,wnorms_05,MSEs_05])
Results_95[:,:,:,0] = np.array([times_95,n_comps_95,wnorms_95,MSEs_95])


p = 3
prob_one = 1/p

times = np.load("1times_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
wnorms = np.load("1Wnorms"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
MSEs = np.load("1MSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1n_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


times.shape
times_med = np.median(times,axis=2)
times_05 = np.quantile(times, 0.05, axis=2)
times_95 = np.quantile(times, 0.95, axis=2)

n_comps.shape
n_comps_mean = np.mean(n_comps,axis=2)
n_comps_med = np.median(n_comps_mean,axis=1)
n_comps_med = np.transpose(np.append([n_comps_med], [[p**2,p**2,p**2]],axis=0))
n_comps_05 = np.quantile(n_comps_mean, 0.05, axis=1)
n_comps_05 = np.transpose(np.append([n_comps_05], [[p**2,p**2,p**2]],axis=0))
n_comps_95 = np.quantile(n_comps_mean, 0.95, axis=1)
n_comps_95 = np.transpose(np.append([n_comps_95], [[p**2,p**2,p**2]],axis=0))


wnorms.shape
wnorms_mean = np.mean(wnorms,axis=(3))
wnorms_mean.shape
wnorms_med = np.median(wnorms_mean,axis=2)
wnorms_05 = np.quantile(wnorms_mean, 0.05, axis=2)
wnorms_95 = np.quantile(wnorms_mean, 0.95, axis=2)

MSEs.shape
MSEs_mean = np.mean(MSEs, axis=(3,4,5))
MSEs_mean.shape
MSEs_med = np.median(MSEs_mean, axis=2)
MSEs_05 = np.quantile(MSEs_mean, 0.05, axis=2)
MSEs_95 = np.quantile(MSEs_mean, 0.95, axis=2)

Results[:,:,:,1] = np.array([times_med,n_comps_med,wnorms_med,MSEs_med])
Results_05[:,:,:,1] = np.array([times_05,n_comps_05,wnorms_05,MSEs_05])
Results_95[:,:,:,1] = np.array([times_95,n_comps_95,wnorms_95,MSEs_95])




p = 4
prob_one = 1/p

times = np.load("1times_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
wnorms = np.load("1Wnorms"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
MSEs = np.load("1MSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1n_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


times.shape
times_med = np.median(times,axis=2)
times_05 = np.quantile(times, 0.05, axis=2)
times_95 = np.quantile(times, 0.95, axis=2)

n_comps.shape
n_comps_mean = np.mean(n_comps,axis=2)
n_comps_med = np.median(n_comps_mean,axis=1)
n_comps_med = np.transpose(np.append([n_comps_med], [[p**2,p**2,p**2]],axis=0))
n_comps_05 = np.quantile(n_comps_mean, 0.05, axis=1)
n_comps_05 = np.transpose(np.append([n_comps_05], [[p**2,p**2,p**2]],axis=0))
n_comps_95 = np.quantile(n_comps_mean, 0.95, axis=1)
n_comps_95 = np.transpose(np.append([n_comps_95], [[p**2,p**2,p**2]],axis=0))


wnorms.shape
wnorms_mean = np.mean(wnorms,axis=(3))
wnorms_mean.shape
wnorms_med = np.median(wnorms_mean,axis=2)
wnorms_05 = np.quantile(wnorms_mean, 0.05, axis=2)
wnorms_95 = np.quantile(wnorms_mean, 0.95, axis=2)

MSEs.shape
MSEs_mean = np.mean(MSEs, axis=(3,4,5))
MSEs_mean.shape
MSEs_med = np.median(MSEs_mean, axis=2)
MSEs_05 = np.quantile(MSEs_mean, 0.05, axis=2)
MSEs_95 = np.quantile(MSEs_mean, 0.95, axis=2)

Results[:,:,:,2] = np.array([times_med,n_comps_med,wnorms_med,MSEs_med])
Results_05[:,:,:,2] = np.array([times_05,n_comps_05,wnorms_05,MSEs_05])
Results_95[:,:,:,2] = np.array([times_95,n_comps_95,wnorms_95,MSEs_95])




p = 5
prob_one = 1/p

times = np.load("1times_all"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
wnorms = np.load("1Wnorms"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
MSEs = np.load("1MSES"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")
n_comps = np.load("1n_comps"+"p="+str(p)+"conc="+str(conc)+"noise_sd="+str(noise_sd)+"prob_one="+str(prob_one)+".npy")


times.shape
times_med = np.median(times,axis=2)
times_05 = np.quantile(times, 0.05, axis=2)
times_95 = np.quantile(times, 0.95, axis=2)

n_comps.shape
n_comps_mean = np.mean(n_comps,axis=2)
n_comps_med = np.median(n_comps_mean,axis=1)
n_comps_med = np.transpose(np.append([n_comps_med], [[p**2,p**2,p**2]],axis=0))
n_comps_05 = np.quantile(n_comps_mean, 0.05, axis=1)
n_comps_05 = np.transpose(np.append([n_comps_05], [[p**2,p**2,p**2]],axis=0))
n_comps_95 = np.quantile(n_comps_mean, 0.95, axis=1)
n_comps_95 = np.transpose(np.append([n_comps_95], [[p**2,p**2,p**2]],axis=0))


wnorms.shape
wnorms_mean = np.mean(wnorms,axis=(3))
wnorms_mean.shape
wnorms_med = np.median(wnorms_mean,axis=2)
wnorms_05 = np.quantile(wnorms_mean, 0.05, axis=2)
wnorms_95 = np.quantile(wnorms_mean, 0.95, axis=2)

MSEs.shape
MSEs_mean = np.mean(MSEs, axis=(3,4,5))
MSEs_mean.shape
MSEs_med = np.median(MSEs_mean, axis=2)
MSEs_05 = np.quantile(MSEs_mean, 0.05, axis=2)
MSEs_95 = np.quantile(MSEs_mean, 0.95, axis=2)

Results[:,:,:,3] = np.array([times_med,n_comps_med,wnorms_med,MSEs_med])
Results_05[:,:,:,3] = np.array([times_05,n_comps_05,wnorms_05,MSEs_05])
Results_95[:,:,:,3] = np.array([times_95,n_comps_95,wnorms_95,MSEs_95])



############ Produce plots


fig, axs = plt.subplots(2,2,figsize=(9,7), layout="constrained")

fig.suptitle("Diagonal", fontsize=20)

axs[0,0].plot([2,3,4,5],Results[0,0,1],marker="o", label = "regular")
axs[0,0].fill_between([2,3,4,5], Results_05[0,0,1], Results_95[0,0,1], alpha=0.5)
axs[0,0].plot([2,3,4,5],Results[0,0,0],marker="o", label = "sparse")
axs[0,0].fill_between([2,3,4,5], Results_05[0,0,0], Results_95[0,0,0], alpha=0.5)
axs[0,0].set_title("Computing Time")
axs[0,0].set_xticks([2,3,4,5])


axs[0,1].plot([2,3,4,5],Results[1,0,1],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,0,1], Results_95[1,0,1], alpha=0.5)
axs[0,1].plot([2,3,4,5],Results[1,0,0],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,0,0], Results_95[1,0,0], alpha=0.5)
axs[0,1].set_title("Non-Zero Entries")
axs[0,1].set_xticks([2,3,4,5])

axs[1,0].plot([2,3,4,5],Results[2,0,1],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,0,1], Results_95[2,0,1], alpha=0.5)
axs[1,0].plot([2,3,4,5],Results[2,0,0],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,0,0], Results_95[2,0,0], alpha=0.5)
axs[1,0].set_title("Wasserstein Distances")
axs[1,0].set_xticks([2,3,4,5])

axs[1,1].plot([2,3,4,5],Results[3,0,1],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,0,1], Results_95[3,0,1], alpha=0.5)
axs[1,1].plot([2,3,4,5],Results[3,0,0],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,0,0], Results_95[3,0,0], alpha=0.5)
axs[1,1].set_title("Mean Square Error")
axs[1,1].set_xticks([2,3,4,5])

handles, labels = axs[0,0].get_legend_handles_labels()

fig.legend(handles,labels,loc='center right', bbox_to_anchor=(1.15,0.5))

fig.supxlabel("p")


plt.show()




fig, axs = plt.subplots(2,2,figsize=(9,7), layout="constrained")

fig.suptitle("Diagonal", fontsize=20)

axs[0,0].plot([2,3,4,5],Results[0,1,1],marker="o", label = "regular")
axs[0,0].fill_between([2,3,4,5], Results_05[0,1,1], Results_95[0,1,1], alpha=0.5)
axs[0,0].plot([2,3,4,5],Results[0,1,0],marker="o", label = "sparse")
axs[0,0].fill_between([2,3,4,5], Results_05[0,1,0], Results_95[0,1,0], alpha=0.5)
axs[0,0].set_title("Computing Time")
axs[0,0].set_xticks([2,3,4,5])


axs[0,1].plot([2,3,4,5],Results[1,1,1],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,1,1], Results_95[1,1,1], alpha=0.5)
axs[0,1].plot([2,3,4,5],Results[1,1,0],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,1,0], Results_95[1,1,0], alpha=0.5)
axs[0,1].set_title("Non-Zero Entries")
axs[0,1].set_xticks([2,3,4,5])

axs[1,0].plot([2,3,4,5],Results[2,1,1],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,1,1], Results_95[2,1,1], alpha=0.5)
axs[1,0].plot([2,3,4,5],Results[2,1,0],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,1,0], Results_95[2,1,0], alpha=0.5)
axs[1,0].set_title("Wasserstein Distances")
axs[1,0].set_xticks([2,3,4,5])

axs[1,1].plot([2,3,4,5],Results[3,1,1],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,1,1], Results_95[3,1,1], alpha=0.5)
axs[1,1].plot([2,3,4,5],Results[3,1,0],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,1,0], Results_95[3,1,0], alpha=0.5)
axs[1,1].set_title("Mean Square Error")
axs[1,1].set_xticks([2,3,4,5])

handles, labels = axs[0,0].get_legend_handles_labels()

fig.legend(handles,labels,loc='center right', bbox_to_anchor=(1.15,0.5))

fig.supxlabel("p")


plt.show()



fig, axs = plt.subplots(2,2,figsize=(9,7), layout="constrained")

fig.suptitle("Diagonal", fontsize=20)

axs[0,0].plot([2,3,4,5],Results[0,2,1],marker="o", label = "regular")
axs[0,0].fill_between([2,3,4,5], Results_05[0,2,1], Results_95[0,2,1], alpha=0.5)
axs[0,0].plot([2,3,4,5],Results[0,2,0],marker="o", label = "sparse")
axs[0,0].fill_between([2,3,4,5], Results_05[0,2,0], Results_95[0,2,0], alpha=0.5)
axs[0,0].set_title("Computing Time")
axs[0,0].set_xticks([2,3,4,5])


axs[0,1].plot([2,3,4,5],Results[1,2,1],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,2,1], Results_95[1,2,1], alpha=0.5)
axs[0,1].plot([2,3,4,5],Results[1,2,0],marker="o")
axs[0,1].fill_between([2,3,4,5], Results_05[1,2,0], Results_95[1,2,0], alpha=0.5)
axs[0,1].set_title("Non-Zero Entries")
axs[0,1].set_xticks([2,3,4,5])

axs[1,0].plot([2,3,4,5],Results[2,2,1],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,2,1], Results_95[2,2,1], alpha=0.5)
axs[1,0].plot([2,3,4,5],Results[2,2,0],marker="o")
axs[1,0].fill_between([2,3,4,5], Results_05[2,2,0], Results_95[2,2,0], alpha=0.5)
axs[1,0].set_title("Wasserstein Distances")
axs[1,0].set_xticks([2,3,4,5])

axs[1,1].plot([2,3,4,5],Results[3,2,1],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,2,1], Results_95[3,2,1], alpha=0.5)
axs[1,1].plot([2,3,4,5],Results[3,2,0],marker="o")
axs[1,1].fill_between([2,3,4,5], Results_05[3,2,0], Results_95[3,2,0], alpha=0.5)
axs[1,1].set_title("Mean Square Error")
axs[1,1].set_xticks([2,3,4,5])

handles, labels = axs[0,0].get_legend_handles_labels()

fig.legend(handles,labels,loc='center right', bbox_to_anchor=(1.15,0.5))

fig.supxlabel("p")


plt.show()

















