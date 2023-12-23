# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:51:00 2023

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt

res_vec = np.loadtxt("res_vec.csv",delimiter=",")
res_mat = np.loadtxt("res_mat.csv",delimiter=",")


xs = np.array([2,4,6,8])


fig, axs = plt.subplots(2, 2, figsize=(7,7), layout='constrained')


axs[0,0].plot(xs,res_vec[0],marker="o",label="Vector")
axs[0,0].plot(xs,res_mat[0],marker="o",label="Matrix")
axs[0,0].set_title("n=100")

axs[0,1].plot(xs,res_vec[1],marker="o")
axs[0,1].plot(xs,res_mat[1],marker="o")
axs[0,1].set_title("n=200")

axs[1,0].plot(xs,res_vec[2],marker="o")
axs[1,0].plot(xs,res_mat[2],marker="o")
axs[1,0].set_title("n=300")

axs[1,1].plot(xs,res_vec[3],marker="o")
axs[1,1].plot(xs,res_mat[3],marker="o")
axs[1,1].set_title("n=400")

handles, labels = axs[0,0].get_legend_handles_labels()

fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

fig.supxlabel('p')
fig.supylabel('Time (sec)')

# plt.savefig("likeEval2.pdf", format="pdf", bbox_inches="tight")
plt.show()

