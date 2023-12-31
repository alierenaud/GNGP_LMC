# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:38:32 2023

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt

white = np.loadtxt("effWhite.csv",delimiter=",",skiprows=1)
center = np.loadtxt("effCenter.csv",delimiter=",",skiprows=1)
inter = np.loadtxt("effInter.csv",delimiter=",",skiprows=1)


fig, axs = plt.subplots(3, 2, figsize=(9,11), layout='constrained')



axs[0,0].boxplot([center[:,0],inter[:,0],white[:,0]])
axs[0,0].tick_params(bottom=False)
axs[0,0].set(xticklabels=[])  
axs[0,0].set_title("C_12(0) (StN=0.1)")
axs[0,1].boxplot([center[:,1],inter[:,1],white[:,1]],labels=["Centered","Interweaved","Whitened"])
axs[0,1].tick_params(bottom=False)
axs[0,1].set(xticklabels=[])  
axs[0,1].set_title("C_12(0.1) (StN=0.1)")

axs[1,0].boxplot([center[:,2],inter[:,2],white[:,2]],labels=["Centered","Interweaved","Whitened"])
axs[1,0].tick_params(bottom=False)
axs[1,0].set(xticklabels=[])  
axs[1,0].set_title("C_12(0) (StN=1)")
axs[1,1].boxplot([center[:,3],inter[:,3],white[:,3]],labels=["Centered","Interweaved","Whitened"])
axs[1,1].tick_params(bottom=False)
axs[1,1].set(xticklabels=[])  
axs[1,1].set_title("C_12(0.1) (StN=1)")

axs[2,0].boxplot([center[:,4],inter[:,4],white[:,4]],labels=["Centered","Interweaved","Whitened"])
axs[2,0].set_title("C_12(0) (StN=10)")
axs[2,1].boxplot([center[:,5],inter[:,5],white[:,5]],labels=["Centered","Interweaved","Whitened"])
axs[2,1].set_title("C_12(0.1) (StN=10)")


fig.supylabel('Effective Sample Size')

plt.savefig("meffSamp2.pdf", format="pdf", bbox_inches="tight")
# plt.show()