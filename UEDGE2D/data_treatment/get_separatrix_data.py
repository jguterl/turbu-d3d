#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:49:12 2023

@author: peretm
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d

#file2load ='/fusion/projects/boundary/peretm/simulations/2D_Hplasma/SaveDir/rd_newgrid_nc2.00e+19_pcore1.80e+06_flat/save_290623_165302.npy'
#file2load = '/fusion/projects/boundary/peretm/simulations/2D_Hplasma/SaveDir/rd_newgrid_nc2.00e+19_pcore1.80e+06_flat/last.npy'
def get_separatrix_data(file2load):
    '''
    Get the data along the separatrix contour
    Parameters
    ----------
    file2load : string 
        Use the path to the the simulation output file

    Returns
    -------
    sep : Dictionary
        Contains the profiles along the separatrix for the 2D UEDGE simulations 
        (ne, te, ti, ng, up, M, R, Z, s=curvilinear coordinate)

    '''
    sim = np.load(file2load, allow_pickle=True).tolist()

    sep = {}
    ee = 1.602e-19
    mi = 1.67e-27
    
    iysep = sim[3][0]['com.iysptrx']
    sep['te'] = sim[3][0]['bbb.te'][:,iysep]/ee
    sep['ti'] = sim[3][0]['bbb.ti'][:,iysep]/ee
    sep['ne'] = sim[3][0]['bbb.ni'][:,iysep,0]
    sep['ng'] = sim[3][0]['bbb.ng'][:,iysep,0]
    sep['up'] = sim[3][0]['bbb.up'][:,iysep,0]
    sep['M'] = sep['up']/np.sqrt(ee/mi*(sep['te']+sep['ti']))
    sep['R'] = sim[3][0]['com.rm'][:,iysep,0]
    sep['Z'] = sim[3][0]['com.zm'][:,iysep,0]
    
    ds = np.append(0,np.sqrt(np.diff(sep['R'])**2+np.diff(sep['Z'])**2))
    sep['s'] = np.cumsum(ds) #poloidal curvilinear coordinate/ pitch angle missing for parallel calculation
    
    return sep

def get_Xpt_position(R,Z):
    '''
    Get the X-point radial and vertical position
    Parameters
    ----------
    R : array
        R coordinate of separatrix
    Z : array
        Z coordinate of separatrix

    Returns
    -------
    Rx : float
        Radial position of X-point
    Zx : float
        Vertical position of X-point

    '''
    itop = np.argmax(Z)
    
    R1 = R[0:itop]
    R2 = R[itop+1:-1]
    Z1 = Z[0:itop]
    Z2 = Z[itop+1:-1]
    
    ixpt = 0
    jxpt = 0
    Rstart = 1e6
    #A = np.zeros((len(R1),len(R2)))
    for i in range(len(R1)):
        for j in range(len(R2)):
            if (i!=j):
                A = np.sqrt((R1[i]-R2[j])**2+(Z1[i]-Z2[j])**2)
                if (A<Rstart):
                    ixpt = i
                    jxpt = j
                    Rstart = A

    Rx = 0.5 * (R1[ixpt] + R2[jxpt])
    Zx = 0.5 * (Z1[ixpt] + Z2[jxpt])
    return Rx, Zx, ixpt, jxpt+itop+1

def get_sep_geom_params(R,Z):
    '''
    Get the shaping of the separatrix

    Parameters
    ----------
    R : array
        R coordinate of separatrix
    Z : array
        Z coordinate of separatrix

    Returns
    -------
    Rmin : float
        minimum radial position at separatrix
    Rmax : float
        maximal radial position at separatrix
    Zmin : float
        minimum vertical position at separatrix
    Zmax : float
        maximal vertical position at separatrix
    R0 : float
        separarix center radial position (major radius)
    Z0 : float
        separatrix center vertical position 
    a : float
        minor radius
    kappa : float
        ellongation
    delta_up : float
        upper triangularity
    delta_low : float
        lower triangularity
    delta : float
        mean triangularity
    Rxpt : float
        Radial position of X-point
    Zxpt : float
        Vertical position of X-point

    '''
    Rxpt, Zxpt, a, b = get_Xpt_position(R,Z) 
    
    Rmin = np.min(R)
    Rmax = np.max(R)
    Zmin = Zxpt
    Zmax = np.max(Z)
    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmin + Zmax)
    a = 0.5 * (Rmax - Rmin)
    kappa = (Zmax - Zmin)/(Rmax - Rmin)
    delta_up = (R0 - R[np.argmax(Z)])/a
    delta_low = (R0 - Rxpt)/a
    delta = 0.5 *(delta_up + delta_low)
    
    return Rmin, Rmax, Zmin, Zmax, R0, Z0, a, kappa, delta_up, delta_low, delta, Rxpt, Zxpt

def get_uxt_indices(R, Z):
    '''
    Get indices of upstream, X-point and target positions

    Parameters
    ----------
    R : array
        R coordinate of separatrix
    Z : array
        Z coordinate of separatrix

    Returns
    -------
    i_u : integer
        index of the upstream condition
    i_x : integer
        index of the X-point condition
    i_t : integer
        index of the target condition

    '''
    i_t = len(R)-1
    i_u = np.argmax(R)
    Rxpt, Zxpt, i0, i1 = get_Xpt_position(R,Z) 
    R1 = R[np.argmax(Z):-1]
    i_x = np.argmax(Z)+(np.argmin(np.abs(R1 - Rxpt)))
    return i_u, i_x, i_t

def get_u2x2t_indices(R, Z):
    '''
    Get indices of upstream, X-point and target positions at both sides of X-pt and both targets

    Parameters
    ----------
    R : array
        R coordinate of separatrix
    Z : array
        Z coordinate of separatrix

    Returns
    -------
    i_u : integer
        index of the upstream condition
    i_x : integer
        index of the X-point condition
    i_t : integer
        index of the target condition

    '''
    i_t1 = len(R)-1
    i_t2 = 0
    i_u = np.argmax(R)
    Rxpt, Zxpt, i_x1, i_x2 = get_Xpt_position(R,Z) 
    # R1 = R[np.argmax(Z):-1]
    # i_x = np.argmax(Z)+(np.argmin(np.abs(R1 - Rxpt)))
    return i_u, i_x1, i_x2, i_t1, i_t2

def get_separatrix_par_data(file2load,rrfile):
    '''
    Get the data along the separatrix contour
    Parameters
    ----------
    file2load : string 
        Use the path to the the simulation output file

    Returns
    -------
    sep : Dictionary
        Contains the profiles along the separatrix for the 2D UEDGE simulations 
        (ne, te, ti, ng, up, M, R, Z, s=curvilinear coordinate)

    '''
    sim = np.load(file2load, allow_pickle=True).tolist()
    

    sep = {}
    ee = 1.602e-19
    mi = 1.67e-27
    
    iysep = sim[3][0]['com.iysptrx']
    sep['te'] = sim[3][0]['bbb.te'][:,iysep]/ee
    sep['ti'] = sim[3][0]['bbb.ti'][:,iysep]/ee
    sep['ne'] = sim[3][0]['bbb.ni'][:,iysep,0]
    sep['ng'] = sim[3][0]['bbb.ng'][:,iysep,0]
    sep['up'] = sim[3][0]['bbb.up'][:,iysep,0]
    sep['M'] = sep['up']/np.sqrt(ee/mi*(sep['te']+sep['ti']))
    sep['R'] = sim[3][0]['com.rm'][:,iysep,0]
    sep['Z'] = sim[3][0]['com.zm'][:,iysep,0]
 
    sep['rr'] = np.load(rrfile, allow_pickle=True).tolist()[3][0]['com.rr'][:,iysep]
    ds = np.append(0,np.sqrt(np.diff(sep['R'])**2+np.diff(sep['Z'])**2))
    dspar = np.append(0,np.tan(sep['rr'][0:-1])**(-1)*np.sqrt(np.diff(sep['R'])**2+np.diff(sep['Z'])**2))
    
    sep['s'] = np.cumsum(ds) #poloidal curvilinear coordinate/ pitch angle missing for parallel calculation
    sep['spar'] = np.cumsum(dspar)
    return sep
#sep = get_separatrix_data(file2load)
#i_u, i_x, i_t = get_uxt_indices(sep['R'],sep['Z'])
#Rmin, Rmax, Zmin, Zmax, R0, Z0, a, kappa, delta_up, delta_low, delta, Rxpt, Zxpt = get_sep_geom_params(sep['R'], sep['Z'])
# plt.figure()
# plt.plot(sep['s'], sep['R'],'bo')
# plt.plot(sep['s'], sep['Z'], 'ro')
# plt.plot(sep['s'], np.sqrt(sep['R']**2+sep['Z']**2),'go')

# plt.figure()
# plt.plot(sep['s'],sep['up'], 'ko')
# plt.plot(sep['s'][i_u], sep['up'][i_u],'go')
# file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/final_state_100723_135235.npy'
# rrfile = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/rr.npy'
# sep = get_separatrix_par_data(file2load, rrfile)
# i_u, i_x1, i_x2, i_t1, i_t2 = get_u2x2t_indices(sep['R'],sep['Z'])
# Rmin, Rmax, Zmin, Zmax, R0, Z0, a, kappa, delta_up, delta_low, delta, Rxpt, Zxpt = get_sep_geom_params(sep['R'], sep['Z'])

# fig, ax = plt.subplots(1)
# ax.plot(sep['R'], sep['Z'], 'ko')
# ax.plot(Rxpt, Zxpt, 'ro')
# ax.plot(R0, Z0, 'bo')
# ax.plot(sep['R'][i_u], sep['Z'][i_u], 'go')
# ax.plot(sep['R'][i_x2], sep['Z'][i_x2], 'go')
# ax.plot(sep['R'][i_t1], sep['Z'][i_t1], 'go')
# ax.plot(sep['R'][i_x1], sep['Z'][i_x1], 'mo')
# ax.plot(sep['R'][i_t2], sep['Z'][i_t2], 'mo')
# ax.axis('equal')

