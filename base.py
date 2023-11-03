#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:05:22 2023

@author: homeboy
"""

import numpy as np
from scipy.special import gamma, kv

def matern_kernel(r, phi = 1, v = 0.4):
    r = np.abs(r)
    r[r == 0] = 1e-8
    part1 = 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * r / phi) ** v
    part3 = kv(v, np.sqrt(2 * v) * r / phi)
    return part1 * part2 * part3


# def matern_kernel(r, phi = 1):
    
#     return 0.5*(np.exp(-r/phi) + np.exp(-r/phi/2))

# def matern_kernel(r, phi = 1, alpha = 0.5):
    
#     return (1+(r/phi)**2)**(-alpha)

def fct(s):
    
    return(np.sin(s)/(0.1*s**2+1)*2)
