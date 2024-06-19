#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:08:26 2023

@author: peretm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:34:15 2023

@author: peretm
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from omfit_classes.omfit_mds import OMFITmdsValue
from omfit_classes.omfit_thomson import OMFITthomson
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from scipy import integrate
from scipy.ndimage.filters import uniform_filter1d

def get_G0_ks_2(shotnumber, k00, database, file2load):
    database = np.load(file2load,allow_pickle=True).tolist()
    shot = shotnumber

    Filter = {'core': {'redchisq_limit': 10.0,'frac_temp_err_hot_max': 0.3,
                        'frac_temp_err_cold_max': 0.95,
                        'frac_dens_err_max': 0.3}}
    ts = OMFITthomson('DIII-D', shot, 'EFIT01', -1, ['core', 'tangential'], quality_filters=Filter)
    ts()
    eq0 = ts['efit_data']#read_basic_eq_from_mds(device='DIII-D', shot=shotnumber, tree='EFIT01', quiet=False, toksearch_mds=None)
    
    efit_type = 'EFIT01'

    time = np.mean(database['eq']['atime'])#[0]
    device = 'DIII-D'
    
    # eq0 = OMFITgeqdsk('g%06d.%05d' % (shot, time)).from_mdsplus(
    #     device=device, shot=shot, time=time, SNAPfile=efit_type)
    

    eq = database['eq']
    R0 = np.mean(eq['rmaxis'])
    Z0 = np.mean(eq['zmaxis'])

    R = eq['r']
    Z = eq['z']
    ind = np.where(eq0['atime']>=eq['atime'][0])[0]
    ind = ind[np.where(eq0['atime'][ind]>=eq['atime'][-1])[0]]

    psi = np.mean(eq0['psirz'][ind,:,:], axis=0)

    R_int = np.arange(np.nanmin(R), np.nanmax(R), 3e-4)
    Z_int = np.arange(np.nanmin(Z), np.nanmax(Z), 3e-4)

    psi_interp = interp2d(R, Z, psi)
    psi_int = psi_interp(R_int, Z_int)


    R_int2D, Z_int2D = np.meshgrid(R_int,Z_int)

    BT = np.abs(database['mean']['BT']) * R0 / R_int2D

    BR = -1/R_int2D[0:-1,:] * np.diff(psi_int, axis = 0)/np.diff(Z_int2D, axis = 0)
    BZ =  1/R_int2D[:,0:-1] * np.diff(psi_int, axis = 1)/np.diff(R_int2D, axis = 1)

    dRpsi = np.diff(psi_int, axis = 1)/np.diff(R_int2D, axis = 1)
    dZpsi = np.diff(psi_int, axis = 0)/np.diff(Z_int2D, axis = 0)

    BR = np.delete(BR, -1, 1)
    dZpsi = np.delete(dZpsi,-1,1)
    dRpsi = np.delete(dRpsi, -1, 0)
    BZ = np.delete(BZ, -1, 0)
    BT = np.delete(BT, -1, 0)
    BT = np.delete(BT, -1, 1)
    psi_int = np.delete(psi_int, -1, 1)
    psi_int = np.delete(psi_int, -1, 0)
    R_int = np.delete(R_int,-1,0)
    Z_int = np.delete(Z_int, -1, 0)

    cs = plt.contour(R, Z, np.mean(eq['psin'],axis=0), levels = [1])
    for item in cs.collections:
       for i in item.get_paths():
          v = i.vertices
          R_sep = v[:, 0]
          Z_sep = v[:, 1]
    plt.close()


    dR_FT = 0e-2
    r_FT = 5e-3

    ind_OMP = np.where(R_sep>=R0)[0]
    ind_OMP = ind_OMP[np.argmin(np.abs(Z_sep[ind_OMP]-Z0))]
    R0_FT = R_sep[ind_OMP]+dR_FT

    n_FL = 50

    theta = np.linspace(0.0, 1.999*math.pi, n_FL)



    # plt.figure()
    # plt.contour(R, Z, np.mean(eq['psin'],axis=0), levels = [1], colors = ['#C0C0C0'])
    # plt.contour(R, Z, np.mean(eq['psin'],axis=0), levels = [0.96,0.98,1.02], colors=['#808080','#808080','#808080'])
    # plt.plot(eq['lim'][:,0], eq['lim'][:,1], 'k')
    # plt.axis('equal')   

    par_step = 0.005
    n_par = 2*math.pi*database['mean']['q95']*database['mean']['R0']/par_step
    if database['mean']['Z_OSP']<0:
        n_par_step_down = int(np.floor(0.3*n_par))
        n_par_step_up = int(np.floor(0.7*n_par))
    else:
        n_par_step_up = int(np.floor(0.3*n_par))
        n_par_step_down =int(np.floor(0.7*n_par))
    G0 = np.zeros(n_par_step_up+n_par_step_down+1)
    s = np.zeros(n_par_step_up+n_par_step_down+1)
    ks = np.zeros(n_par_step_up+n_par_step_down+1)
    R00_FT = np.zeros(n_par_step_up+n_par_step_down+1)
    Z00_FT = np.zeros(n_par_step_up+n_par_step_down+1)
    theta_FT = np.zeros(n_par_step_up+n_par_step_down+1)

    R_FT = np.zeros((n_par_step_up+n_par_step_down+1, n_FL))
    Z_FT = np.zeros((n_par_step_up+n_par_step_down+1, n_FL))
    psi_FT = np.zeros((n_par_step_up+n_par_step_down+1, n_FL))

    R_FT[n_par_step_down,:] = R0_FT + r_FT*np.cos(theta)
    Z_FT[n_par_step_down,:] = Z0 + r_FT*np.sin(theta)

    R00_FT[n_par_step_down] = R0_FT
    Z00_FT[n_par_step_down] = Z0
    theta_FT[n_par_step_down] = 0.0

    G0[n_par_step_down] = 2.0
    s[n_par_step_down] = 0.0
    for j in range(n_FL):
        psi_FT[n_par_step_down,j] = psi_interp(R_FT[n_par_step_down,j], Z_FT[n_par_step_down,j])

    # plt.plot(R_FT[n_par_step_down,:], Z_FT[n_par_step_down,:] ,'k')
     

    er = interp2d(R_int, Z_int, BR/np.sqrt(BR**2+BZ**2+BT**2))
    ez = interp2d(R_int, Z_int, BZ/np.sqrt(BR**2+BZ**2+BT**2))

    epsir = interp2d(R_int, Z_int, dRpsi/np.sqrt(dRpsi**2+dZpsi**2))
    epsiz = interp2d(R_int, Z_int, dZpsi/np.sqrt(dRpsi**2+dZpsi**2))

    for i in range(n_par_step_up):
        for j in range(n_FL):
            R_FT[n_par_step_down+i+1,j] = R_FT[n_par_step_down + i,j] + par_step * er(R_FT[n_par_step_down+i,j], Z_FT[n_par_step_down+i,j])
            Z_FT[n_par_step_down+i+1,j] = Z_FT[n_par_step_down + i,j] + par_step * ez(R_FT[n_par_step_down+i,j], Z_FT[n_par_step_down+i,j])
            psi_FT[n_par_step_down+i+1,j] = psi_interp(R_FT[n_par_step_down+i+1,j], Z_FT[n_par_step_down+i+1,j])
        dR = (R_FT[n_par_step_down+i+1,np.argmax(psi_FT[n_par_step_down+i+1,:])]-R_FT[n_par_step_down+i+1,np.argmin(psi_FT[n_par_step_down+i+1,:])])
        dZ = (Z_FT[n_par_step_down+i+1,np.argmax(psi_FT[n_par_step_down+i+1,:])]-Z_FT[n_par_step_down+i+1,np.argmin(psi_FT[n_par_step_down+i+1,:])])       
        R00_FT[n_par_step_down+i+1] = np.mean(R_FT[n_par_step_down+i+1,:])
        Z00_FT[n_par_step_down+i+1] = np.mean(Z_FT[n_par_step_down+i+1,:])
        
        if ((np.sign(Z00_FT[n_par_step_down+i+1]-Z0)<0) and (np.sign(R00_FT[n_par_step_down+i+1]-R0)<0)):
            add = 2* math.pi
        else:
            add = 0
        theta_FT[n_par_step_down+i+1] = np.angle(1j*(Z00_FT[n_par_step_down+i+1]-Z0)+(R00_FT[n_par_step_down+i+1]-R0)) +add
        # G0[n_par_step_down+i+1] = 2.0 * np.cos(np.arctan(epsiz(R00_FT[n_par_step_down+i+1],Z00_FT[n_par_step_down+i+1])/epsir(R00_FT[n_par_step_down+i+1],Z00_FT[n_par_step_down+i+1]))-np.arctan(dZ/dR))
        G0[n_par_step_down+i+1] = 2.0 * np.sign(dR) * np.cos(np.arctan(dZ/dR))
        ks[n_par_step_down+i+1] = -np.tan(np.arccos((epsiz(R00_FT[n_par_step_down+i+1],Z00_FT[n_par_step_down+i+1])*dZ+epsir(R00_FT[n_par_step_down+i+1],Z00_FT[n_par_step_down+i+1])*dR)/np.sqrt(dR**2+dZ**2)))
        s[n_par_step_down+i+1] = s[n_par_step_down+i]+par_step
        # if np.mod(i,25)==0:    
        #     plt.plot(R_FT[n_par_step_down+i+1,:], Z_FT[n_par_step_down+i+1,:] ,'r')   
            
    for i in range(n_par_step_down):
        for j in range(n_FL):        
            R_FT[n_par_step_down-i-1, j] = R_FT[n_par_step_down - i,j] - par_step * er(R_FT[n_par_step_down-i,j], Z_FT[n_par_step_down-i,j])
            Z_FT[n_par_step_down-i-1, j] = Z_FT[n_par_step_down - i,j] - par_step * ez(R_FT[n_par_step_down-i,j], Z_FT[n_par_step_down-i,j]) 
            psi_FT[n_par_step_down-i-1,j] = psi_interp(R_FT[n_par_step_down-i-1,j], Z_FT[n_par_step_down-i-1,j])
        dR = (R_FT[n_par_step_down-i-1,np.argmax(psi_FT[n_par_step_down-i-1,:])]-R_FT[n_par_step_down-i-1,np.argmin(psi_FT[n_par_step_down-i-1,:])])
        dZ = (Z_FT[n_par_step_down-i-1,np.argmax(psi_FT[n_par_step_down-i-1,:])]-Z_FT[n_par_step_down-i-1,np.argmin(psi_FT[n_par_step_down-i-1,:])])       
        R00_FT[n_par_step_down-i-1] = np.mean(R_FT[n_par_step_down-i-1,:])
        Z00_FT[n_par_step_down-i-1] = np.mean(Z_FT[n_par_step_down-i-1,:])
        if ((np.sign(Z00_FT[n_par_step_down-i-1]-Z0)>=0) and (np.sign(R00_FT[n_par_step_down-i-1]-R0)<0)):
            add = -2* math.pi
        else:
            add = 0
        theta_FT[n_par_step_down-i-1] = np.angle(1j*(Z00_FT[n_par_step_down-i-1]-Z0)+(R00_FT[n_par_step_down-i-1]-R0))+add 
        # G0[n_par_step_down-i-1] = 2.0 * np.cos(np.arctan(epsiz(R00_FT[n_par_step_down-i-1],Z00_FT[n_par_step_down-i-1])/epsir(R00_FT[n_par_step_down-i-1],Z00_FT[n_par_step_down-i-1]))+np.arctan(dZ/dR))    
        G0[n_par_step_down-i-1] = 2.0 * np.sign(dR) * np.cos(np.arctan(dZ/dR))
        ks[n_par_step_down-i-1] = np.tan(np.arccos((epsiz(R00_FT[n_par_step_down-i-1],Z00_FT[n_par_step_down-i-1])*dZ + epsir(R00_FT[n_par_step_down-i-1],Z00_FT[n_par_step_down-i-1])*dR)/np.sqrt(dR**2+dZ**2)))
        s[n_par_step_down-i-1] = s[n_par_step_down-i]-par_step
        # if np.mod(i,25)==0:  
        #     plt.plot(R_FT[n_par_step_down-i-1,:], Z_FT[n_par_step_down-i-1,:] ,'b')

    # plt.figure()
    # plt.plot(theta_FT, G0, '-ko')

    # plt.figure()
    # plt.plot(theta_FT, ks, '-bo')
    # plt.plot(theta_FT,uniform_filter1d(ks, size = 150), 'k')
    
    # print(np.mean(uniform_filter1d(ks[G0>=0]*G0[G0>=0]/np.mean(G0[G0>=0]), size = 150)))
    if database['mean']['Z_OSP']<0:
        ind = np.where(G0<0)[0]
        try:
            g = np.mean(uniform_filter1d(G0[0:np.min(ind)], size = 150))
            ks0 = np.mean(uniform_filter1d(ks[0:np.min(ind)]*G0[0:np.min(ind)]/np.mean(G0[0:np.min(ind)]), size = 150))
        except:
            []
            # breakpoint()
            # print('test')
    else:
        ind = np.where(G0<0)[0]
        g = np.mean(uniform_filter1d(G0[np.max(ind):], size = 150))
        ks0 = np.mean(uniform_filter1d(ks[np.max(ind):]*G0[np.max(ind):]/np.mean(G0[np.max(ind):]), size = 150))
        
    return g, ks0

