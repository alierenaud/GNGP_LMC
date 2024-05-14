#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:48:31 2024

@author: homeboy
"""

import numpy as np
from numpy import random

## 1 index correspondance

def kay1c(ind,m,n_grid):
    
    i = ind % (n_grid+1)
    j = ind // (n_grid+1)
    
    if (i > m) & (j > m):
        return(m*(m+1)+m)
    elif (i > m) & (j <= m):
        return(j*(m+1)+m)
    elif (i <= m) & (j > m):
        return(m*(m+1)+i)
    else:
        return(j*(m+1)+i)

def phis_move(phis_current,phis_prop,min_phi,max_phi,alphas,betas,A_invVmmu1_current,A_invV_gridmmu1_current,gNei,ogNei,dist_nei_grid,dist_pnei_grid,dist_nei_ogrid,dist_pnei_ogrid,gbs,grs,ogbs,ogrs):
    
    p = phis_current.shape[0]
    npat = dist_nei_grid.shape[0]
    n_obs = dist_pnei_ogrid.shape[0]
    
    m = int(np.sqrt(ogNei.shape[1]) - 1)
    n_grid = int(np.sqrt(gNei.shape[0])-1)
    
    range_phi = max_phi - min_phi
    
    acc_phis = np.zeros(p)
    
    gbs_new = np.zeros((p,npat),dtype=object)
    grs_new = np.zeros((p,npat))

    ogbs_new = np.zeros((p,n_obs,(m+1)**2))
    ogrs_new = np.zeros((p,n_obs))
    
    for j in range(p):
        
        phis_new = phis_current[j] + phis_prop[j]*random.normal()
        
        if (phis_new > min_phi)  &  (phis_new < max_phi):
            
            ### prior
            
            
            phis_new_star_j = (phis_new - min_phi)/range_phi
            phis_current_star_j = (phis_current[j] - min_phi)/range_phi
            
            log_rat_prior = (alphas[j]-1) * (np.log(phis_new_star_j) - np.log(phis_current_star_j)) + (betas[j]-1) * (np.log(1-phis_new_star_j) - np.log(1-phis_current_star_j))
            

            ### grid

            for i in range(npat):
                
                    
                R_j_Ni_inv = np.linalg.inv(np.exp(-dist_nei_grid[i]*phis_new))
                r_j_Nii = np.exp(-dist_pnei_grid[i]*phis_new)
                
                gb = R_j_Ni_inv@r_j_Nii
                
                gbs_new[j,i] = gb
                grs_new[j,i] = 1 - np.inner(r_j_Nii,gb)
                


            
            log_rat_grid = -1/2 * np.sum([[  (A_invV_gridmmu1_current[j,jc*(n_grid+1) + ic] - np.inner(A_invV_gridmmu1_current[j,gNei[jc*(n_grid+1) + ic]],gbs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]))**2/ grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] + np.log(grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]) for ic in range(n_grid+1) ]  for jc in range(n_grid+1)  ]) 
            
            
            log_rat_grid_new = -1/2 * np.sum([[  (A_invV_gridmmu1_current[j,jc*(n_grid+1) + ic] - np.inner(A_invV_gridmmu1_current[j,gNei[jc*(n_grid+1) + ic]],gbs_new[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]))**2/ grs_new[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] + np.log(grs_new[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)])  for ic in range(n_grid+1) ]  for jc in range(n_grid+1)  ]) 
            
            
            
            ### obs
            
            R_j_N_inv = np.linalg.inv(np.exp(-dist_nei_ogrid*phis_new))
            
            
            for i in range(n_obs):
            
                r_j_Nii = np.exp(-dist_pnei_ogrid[i]*phis_new)
            
                ogb = R_j_N_inv@r_j_Nii
                
                ogbs_new[j,i] = ogb
                ogrs_new[j,i] = 1 - np.inner(r_j_Nii,ogb)



            log_rat_obs = -1/2 * np.sum([  (A_invVmmu1_current[j,ic] - np.inner(A_invV_gridmmu1_current[j,ogNei[ic]],ogbs[j,ic]))**2/ ogrs[j,ic] + np.log(ogrs[j,ic]) for ic in range(n_obs) ] ) 
            
            
            log_rat_obs_new = -1/2 * np.sum([  (A_invVmmu1_current[j,ic] - np.inner(A_invV_gridmmu1_current[j,ogNei[ic]],ogbs_new[j,ic]))**2/ ogrs_new[j,ic] + np.log(ogrs_new[j,ic])  for ic in range(n_obs) ] ) 
                                
            
            
            rat = log_rat_grid_new - log_rat_grid + log_rat_obs_new - log_rat_obs + log_rat_prior
            

            
            if np.log(random.uniform()) < rat:
                phis_current[j] = phis_new
                
                gbs[j] = gbs_new[j]
                grs[j] = grs_new[j]
                ogbs[j] = ogbs_new[j]
                ogrs[j] = ogrs_new[j]
                
                acc_phis[j] = 1
                
    return(phis_current,gbs,grs,ogbs,ogrs,acc_phis)


def A_move_slice(A_current, A_invVmmu1_current, A_invV_gridmmu1_current, Vmmu1_current, V_gridmmu1_current, gNei, ogNei, gbs, grs, ogbs, ogrs, sigma_A, mu_A, sigma_slice):

    
    p = A_current.shape[0] 
    n_obs = A_invVmmu1_current.shape[1]
    n_grid = int(np.sqrt(A_invV_gridmmu1_current.shape[1])-1)
    
    m = int(np.sqrt(ogNei.shape[1]) - 1)
    
    ### threshold
    
    log_rat_grid = -1/2 * np.sum([[[  (A_invV_gridmmu1_current[j,jc*(n_grid+1) + ic] - np.inner(A_invV_gridmmu1_current[j,gNei[jc*(n_grid+1) + ic]],gbs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]))**2/ grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]  for ic in range(n_grid+1) ]  for jc in range(n_grid+1)  ] for j in range(p)])  - (n_grid+1)**2 * np.log( np.abs(np.linalg.det(A_current)))
    log_rat_obs = -1/2 * np.sum([[  (A_invVmmu1_current[j,ic] - np.inner(A_invV_gridmmu1_current[j,ogNei[ic]],ogbs[j,ic]))**2/ ogrs[j,ic]  for ic in range(n_obs) ] for j in range(p)] ) - n_obs * np.log( np.abs(np.linalg.det(A_current)))
    
    log_rat_pior = - 1/2 * 1/sigma_A**2 * np.sum((A_current-mu_A)**2)
    
    z =  log_rat_grid + log_rat_obs + log_rat_pior - random.exponential(1,1)
    
    L = A_current - random.uniform(0,sigma_slice,(p,p))
    # L[0] = np.maximum(L[0],0)
    
    U = L + sigma_slice
        
    while True:
    
        
        
        A_prop = random.uniform(L,U)
        A_inv_prop = np.linalg.inv(A_prop)
        A_invVmmu1_prop = A_inv_prop @ Vmmu1_current
        A_invV_gridmmu1_prop = A_inv_prop @ V_gridmmu1_current
        
        log_rat_grid_prop = -1/2 * np.sum([[[  (A_invV_gridmmu1_prop[j,jc*(n_grid+1) + ic] - np.inner(A_invV_gridmmu1_prop[j,gNei[jc*(n_grid+1) + ic]],gbs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]))**2/ grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] for ic in range(n_grid+1) ]  for jc in range(n_grid+1)  ] for j in range(p)])  - (n_grid+1)**2 * np.log( np.abs(np.linalg.det(A_prop)))
        log_rat_obs_prop = -1/2 * np.sum([[  (A_invVmmu1_prop[j,ic] - np.inner(A_invV_gridmmu1_prop[j,ogNei[ic]],ogbs[j,ic]))**2/ ogrs[j,ic] for ic in range(n_obs) ] for j in range(p)] ) - n_obs * np.log( np.abs(np.linalg.det(A_prop)))
        
        log_rat_pior_prop = - 1/2 * 1/sigma_A**2 * np.sum((A_prop-mu_A)**2)
        
        acc = z < log_rat_grid_prop + log_rat_obs_prop + log_rat_pior_prop
            
        if acc:
            return(A_prop,A_inv_prop,A_invVmmu1_prop,A_invV_gridmmu1_prop)
        else:
            for ii in range(p):
                for jj in range(p):
                    if A_prop[ii,jj] < A_current[ii,jj]:
                        L[ii,jj] = A_prop[ii,jj]
                    else:
                        U[ii,jj] = A_prop[ii,jj]

def mu_move(A_inv_current,gNei,ogNei,gbs,grs,ogbs,ogrs,V_current,V_grid_current,sigma_mu,mu_mu):
    
    n_obs = V_current.shape[1]
    n_grid = int(np.sqrt(V_grid_current.shape[1])-1)
    m = int(np.sqrt(ogNei.shape[1])-1)
    p = V_current.shape[0]
    npat = gbs.shape[1]
    
    ms_grid = np.zeros((p,npat)) 
    
    for i in range(npat):
        for j in range(p):
            ms_grid[j,i] = np.sum(gbs[j,i])
    
    ms_obs = np.sum(ogbs,axis=2)
    
    outsies = [np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)]
    
    M_grid = np.sum([[[ (1-ms_grid[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)])**2 / grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] * outsies[j] for ic in range(n_grid+1)] for jc in range(n_grid+1)] for j in range(p)], axis=(0,1,2))
    M_obs = np.sum([[ (1-ms_obs[j,i])**2 / ogrs[j,i] * outsies[j] for i in range(n_obs)] for j in range(p)], axis=(0,1))
    M_prior = np.identity(p)/sigma_mu
    
    M = M_grid + M_obs + M_prior
    
    
    b_grid = np.sum([[[ (1-ms_grid[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]) / grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] * np.inner(A_inv_current[j],V_grid_current[:,jc*(n_grid+1) + ic] - V_grid_current[:,gNei[jc*(n_grid+1) + ic]]@gbs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]) * A_inv_current[j]  for ic in range(n_grid+1)] for jc in range(n_grid+1)] for j in range(p)], axis=(0,1,2))
    b_obs = np.sum([[ (1-ms_obs[j,i]) / ogrs[j,i] * np.inner(A_inv_current[j],V_current[:,i] - V_grid_current[:,ogNei[i]]@ogbs[j,i]) * A_inv_current[j]  for i in range(n_obs)] for j in range(p)], axis=(0,1))
    b_prior = mu_mu/sigma_mu
    
    b = b_grid + b_obs + b_prior
    
    M_inv = np.linalg.inv(M)
    
    mu_current = np.linalg.cholesky(M_inv)@random.normal(size=p) + M_inv@b
    
    Vmmu1_current = V_current - np.outer(mu_current,np.ones(n_obs))
    A_invVmmu1_current = A_inv_current @ Vmmu1_current
    
    V_gridmmu1_current = V_grid_current - np.outer(mu_current,np.ones((n_grid+1)**2))
    A_invV_gridmmu1_current = A_inv_current @ V_gridmmu1_current
    
    return(mu_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current)

def V_move_conj_scale(ogbs, ogrs, ogNei, A_inv_current, taus_current, Dm1_current, Dm1Y_current, Y, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current):
    
    p = A_inv_current.shape[0]
    n = ogbs.shape[1]
    
    outsies = np.array([np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)])
    
    
    
    for i in range(n):
        delta_i = np.sum([outsies[j]/ogrs[j,i] for j in range(p)],axis=0)
        M = delta_i + Dm1_current
        Minv = np.linalg.inv(M)

        b = Dm1Y_current[:,i] + delta_i@mu_current + np.sum([np.inner(A_invV_gridmmu1_current[j,ogNei[i]],ogbs[j,i])/ogrs[j,i]*A_inv_current[j]  for j in range(p)],axis=0)
        
        V_current[:,i] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b
        Vmmu1_current[:,i] = V_current[:,i] - mu_current
        A_invVmmu1_current[:,i] = A_inv_current@Vmmu1_current[:,i]

  
    
    VmY_current = V_current - Y
    VmY_inner_rows_current = np.array([ np.inner(VmY_current[j], VmY_current[j]) for j in range(p) ])
    

    
    return(V_current, Vmmu1_current, VmY_current, VmY_inner_rows_current, A_invVmmu1_current)

def V_grid_move_scale(gbs, ogbs, grs, ogrs, gNei, ogNei, agNei, agInd, aogNei, aogInd, A_inv_current, V_current, V_grid_current, Vmmu1_current, V_gridmmu1_current, A_invVmmu1_current, A_invV_gridmmu1_current, mu_current):
    
    p = A_inv_current.shape[0]
    n_grid = int(np.sqrt(gNei.shape[0]) - 1)
    m = int(np.sqrt(ogNei.shape[1]) - 1)
    
    outsies = np.array([np.outer(A_inv_current[j],A_inv_current[j]) for j in range(p)])
    
    
    
    for ic in range(n_grid+1):
        for jc in range(n_grid+1):
            
            
            M1 = np.sum([outsies[j]/grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)] for j in range(p)],axis=0)
            M2 = np.sum([[outsies[j]/grs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni], m, n_grid)]*gbs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni], m, n_grid)][agInd[jc*(n_grid+1)+ic][ni]]**2 for ni in range(len(agNei[jc*(n_grid+1)+ic]))] for j in range(p)],axis=(0,1))
            M3 = np.sum([[outsies[j]/ogrs[j,aogNei[jc*(n_grid+1)+ic][ni]]*ogbs[j,aogNei[jc*(n_grid+1)+ic][ni],aogInd[jc*(n_grid+1)+ic][ni]]**2 for ni in range(len(aogNei[jc*(n_grid+1)+ic]))] for j in range(p)],axis=(0,1))
            
            M = M1+M2+M3
            Minv = np.linalg.inv(M)
    
    
            b1 = np.sum([np.inner(A_invV_gridmmu1_current[j,gNei[jc*(n_grid+1)+ic]],gbs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)])/grs[j,kay1c(jc*(n_grid+1)+ic, m, n_grid)]*A_inv_current[j]  for j in range(p)],axis=0)
            b2 = np.sum([[(A_invV_gridmmu1_current[j,agNei[jc*(n_grid+1)+ic][ni]] - np.inner(A_invV_gridmmu1_current[j,gNei[agNei[jc*(n_grid+1)+ic][ni]]],gbs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni],m,n_grid)]) + A_invV_gridmmu1_current[j,gNei[agNei[jc*(n_grid+1)+ic][ni]]][agInd[jc*(n_grid+1)+ic][ni]]*gbs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni],m,n_grid)][agInd[jc*(n_grid+1)+ic][ni]])/grs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni], m, n_grid)]*gbs[j,kay1c(agNei[jc*(n_grid+1)+ic][ni], m, n_grid)][agInd[jc*(n_grid+1)+ic][ni]]*A_inv_current[j] for ni in range(len(agNei[jc*(n_grid+1)+ic]))] for j in range(p)],axis=(0,1))
            b3 = np.sum([[(A_invVmmu1_current[j,aogNei[jc*(n_grid+1)+ic][ni]] - np.inner(A_invV_gridmmu1_current[j,ogNei[aogNei[jc*(n_grid+1)+ic][ni]]],ogbs[j,aogNei[jc*(n_grid+1)+ic][ni]]) + A_invV_gridmmu1_current[j,ogNei[aogNei[jc*(n_grid+1)+ic][ni]]][aogInd[jc*(n_grid+1)+ic][ni]]*ogbs[j,aogNei[jc*(n_grid+1)+ic][ni]][aogInd[jc*(n_grid+1)+ic][ni]])/ogrs[j,aogNei[jc*(n_grid+1)+ic][ni]]*ogbs[j,aogNei[jc*(n_grid+1)+ic][ni]][aogInd[jc*(n_grid+1)+ic][ni]]*A_inv_current[j] for ni in range(len(aogNei[jc*(n_grid+1)+ic]))] for j in range(p)],axis=(0,1))
            

            b = b1+b2+b3 + M@mu_current
            V_grid_current[:,jc*(n_grid+1)+ic] = np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b
            V_gridmmu1_current[:,jc*(n_grid+1)+ic] = V_grid_current[:,jc*(n_grid+1)+ic] - mu_current
            A_invV_gridmmu1_current[:,jc*(n_grid+1)+ic] = A_inv_current@V_gridmmu1_current[:,jc*(n_grid+1)+ic]

  
    


    
    return(V_grid_current, V_gridmmu1_current, A_invV_gridmmu1_current)


def taus_move(taus_current,VmY_inner_rows_current,Y,a,b,n):
    
    p = VmY_inner_rows_current.shape[0]
    
    for j in range(p):
        
        taus_current[j] = random.gamma(a + n/2, 1/( b + VmY_inner_rows_current[j]/2), 1)
        
    Dm1_current = np.diag(taus_current)
    Dm1Y_current = Dm1_current @ Y
    
    return(taus_current, Dm1_current, Dm1Y_current)