#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:10:35 2023

@author: homeboy
"""

import numpy as np
# from numpy import random

# # random.seed(42)

# p = 0.3
# n = 5

# M = random.binomial(1, p, size = (n,n))

# # M = np.array([[0,1,1,0,1],
# #               [1,0,0,1,0],
# #               [1,0,0,1,0],
# #               [0,1,0,1,0],
# #               [1,1,0,0,0]])


# # M = np.array([[1, 1, 0, 0, 0],
# #               [0, 0, 1, 1, 0],
# #               [1, 1, 1, 1, 0],
# #               [1, 1, 0, 0, 0],
# #               [0, 0, 1, 1, 1]])
# A = random.normal(size = (n,n))




# # print(M)
# print(np.linalg.det(M*A))




# cols = np.sum(M,axis=0)
# rows = np.sum(M,axis=1)

# # print(cols)
# # print(rows)


# # print(M[:,np.argsort(cols)])


def verif(M):
    
    
    n = M.shape[0]
    cols = np.sum(M,axis=0)
    rows = np.sum(M,axis=1)
    
    
    
    
    if n==1:
        return(M[0,0])
    elif np.prod(cols) == 0 or np.prod(rows) == 0:
        return(0)
    elif np.sum(M) > n**2-n:
        return(1)
    
    else:
        
        minrownum = np.argmin(rows)
        mincolnum = np.argmin(cols)
        
        if rows[minrownum]<cols[mincolnum]:
            
            ### find ones in row
            whereOne = np.where(M[minrownum] == 1)
            
            
            
            for i in whereOne[0]:
                
                rep = verif(np.delete(np.delete(M,minrownum,axis=0),i,axis=1))
                
                if rep == 1:
                    return(1)
                
            return(0)
            
            
        else:
            
            ### find ones in row
            whereOne = np.where(M[:,mincolnum] == 1)
            
            
            
            for i in whereOne[0]:
                
                rep = verif(np.delete(np.delete(M,mincolnum,axis=1),i,axis=0))
                
                if rep == 1:
                    return(1)
                
            return(0)
    
    
    
    
    
# print(verif(M))  
    
    
    
    
    
    
    
    
    
    
    


