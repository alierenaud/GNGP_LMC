#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:43:48 2024

@author: homeboy
"""

import numpy as np
import matplotlib.pyplot as plt


n = np.array([1000,2000,3000,4000],dtype=int)

ti_reg_n = np.array([0.4442291746139526,2.509103304386139,8.128716357707978,18.32778899884224])
ti_nn_n = np.array([0.8495849268436432,1.7509430069923402,2.6494313311576843,3.609885468006134])

p = np.array([1,2,3,4],dtype=int)

ti_reg_p = np.array([1.3975968220233916,2.732190872192383,3.9299866199493407,5.449501956224442])
ti_nn_p = np.array([0.8653918871879578,1.6631958248615264,2.4697423369884492,3.6544807150363923])


plt.figure(figsize=(6,4))
plt.plot(n,ti_reg_n,linestyle="-", marker="o")
plt.plot(n,ti_nn_n,linestyle="-", marker="o")
plt.xticks(n)
plt.legend(["Standard", "NNGP"], loc ="upper left") 
plt.xlabel("n")
plt.ylabel("Time (sec)")

# plt.savefig("time_n_mcmc.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(p,ti_reg_p,linestyle="-", marker="o")
plt.plot(p,ti_nn_p,linestyle="-", marker="o")
plt.xticks(p)
plt.legend(["Standard", "NNGP"], loc ="upper left") 
plt.xlabel("p")
plt.ylabel("Time (sec)")

# plt.savefig("time_p_mcmc.pdf", format="pdf", bbox_inches="tight")
plt.show()