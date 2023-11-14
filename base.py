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
    
#     return np.exp(-(r/phi)**2) + (r==0)*0.001

def matern_kernel(r, phi = 1):
    
    return np.exp(-(r/phi)) + (r==0)*0.001


# def matern_kernel(r, phi = 1, alpha = 2):
    
#     return (1+(r/phi)**2)**(-alpha) + (r==0)*0.001

def fct(s):
    
    return(2*np.sin(s)/(0.1*s**2+1))

# def fct(s):
    
#     res = 0*(s<-8) + 0.5*((s>=-8)*(s<-4)) + 1*((s>=-4)*(s<4)) + 0.5*((s>=4)*(s<8)) + 0*(s>=8)
    
#     return(norm.ppf(res))


# def fct(s):
    
#     return(2*np.sin(s))



def fct2(s):
    
    c = s - [0.5,0.5]
    
    r = np.sqrt(c[:,0]**2+c[:,1]**2)
    
    t = (12*r-np.pi)
    
    return(2*np.sin(t))


# def fct2(s):
    
    
    
#     return(-2*(s[:,0]+s[:,1])**2+2)


# def fct2(s):
    
#     c = s - [0.5,0.5]
    
#     theta = np.pi/4
    
#     Rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
#     cr = c@np.transpose(Rot) 
    
#     r = np.min(cr**2,axis=1)
    
#     t = (50*r+np.pi/2)
    
#     return(2*np.sin(t))


# def fct2(s):
    
#     res = (s[:,0]<0.5)*(s[:,1]<0.5) * 2 + (s[:,0]>=0.5)*(s[:,1]>=0.5) * -2
#     return res




# def fct2(s):
    
    
    
#     return(np.exp(-(s[:,0]**2+s[:,1]**2)))


def vec_inv(A, nrow):
    
    N = A.shape[0]
    ncol = N//nrow
    
    
    return(np.reshape(A,newshape=(nrow,ncol),order='F'))

def makeGrid(x,y):
    return np.dstack(np.meshgrid(x,y)).reshape(-1, 2)

