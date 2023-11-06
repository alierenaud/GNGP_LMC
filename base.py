#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:05:22 2023

@author: homeboy
"""

import numpy as np
from scipy.special import gamma, kv
from scipy.stats import norm

# def matern_kernel(r, phi = 1, v = 0.6):
#     r = np.abs(r)
#     r[r == 0] = 1e-8
#     part1 = 2 ** (1 - v) / gamma(v)
#     part2 = (np.sqrt(2 * v) * r / phi) ** v
#     part3 = kv(v, np.sqrt(2 * v) * r / phi)
#     return part1 * part2 * part3 + (r<1e-7)*0.00001


# def matern_kernel(r, phi = 1):
    
#     return np.exp(-(r/phi)**2) + (r==0)*0.00001

def matern_kernel(r, phi = 1):
    
    return np.exp(-(r/phi)) + (r<1e-7)*0.00001


# def matern_kernel(r, phi = 1, alpha = 0.5):
    
#     return (1+(r/phi)**2)**(-alpha)

def fct(s):
    
    return(np.sin(s)/(0.1*s**2+1))

# def fct(s):
    
#     res = 0.25*(s<-6) + 0.5*((s>=-6)*(s<-2)) + 0.75*((s>=-2)*(s<2)) + 0.5*((s>=2)*(s<6)) + 0.25*(s>=6)
    
#     return(norm.ppf(res))






