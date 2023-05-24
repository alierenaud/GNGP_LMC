#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:02:31 2023

@author: homeboy
"""

import numpy as np


arr = np.load("results.npy")




for i in range(3):
    for j in range(4):
        
        
        if i == 0:
            method = "inter"
        elif i == 1:
            method = "center"
        elif i == 2:
            method = "white"
        
        if j == 0:
            quant = "prange1"
        elif j == 1:
            quant = "prange2"
        elif j == 2:
            quant = "cov0"
        elif j == 3:
            quant = "cov0p1"
        
        np.savetxt(method+quant+".csv", arr[i,j], delimiter=",")
        

