#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:42:27 2023

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
from scipy import integrate
from scipy.ndimage.filters import uniform_filter1d
from omfit_classes.omfit_eqdsk import read_basic_eq_from_mds
from omfit_classes.omfit_mds import OMFITmdsValue
from intersect import intersection
from random import seed
from random import random

def find_Xpt(R, Z, Bp, Rwall, Zwall):
    ind0 = np.where(Z>=0)[0]
    ind1 = np.where(Z<0)[0]
    RXpt0 = np.zeros(len(R))
    RXpt1 = np.zeros(len(R))
    ZXpt0 = np.zeros(len(R))
    ZXpt1 = np.zeros(len(R))   
    Bp0 = np.zeros(len(R))  
    Bp1 = np.zeros(len(R))  
    Rout = np.zeros(2)
    Zout = np.zeros(2)
    for i in range(len(R)):
        ZXpt0[i]=Z[ind0[np.nanargmin(Bp[ind0,i])]]
        Bp0[i] = Bp[ind0[np.nanargmin(Bp[ind0,i])],i]
        ZXpt1[i]=Z[ind1[np.nanargmin(Bp[ind1,i])]]
        Bp1[i] = Bp[ind1[np.nanargmin(Bp[ind1,i])],i]
    for i in range(len(R)):
        Rpt = np.arange(R[i], 3.0, 1e-3)
        Zpt0 = Rpt * 0.0 +ZXpt0[i]
        Zpt1 = Rpt * 0.0 +ZXpt1[i]
        x0, y0 = intersection(Rpt, Zpt0,Rwall, Zwall)
        x0 = np.unique(x0)
        y0 = np.unique(y0)
        if x==[]:
            ZXpt0[i] = math.nan
            Bp0[i] = math.nan
                
        elif np.floor(len(x0)/2)==len(x0)/2:
            ZXpt0[i] = math.nan
            Bp0[i] = math.nan
    
        x1, y1 = intersection(Rpt, Zpt1,Rwall, Zwall)
        x1 = np.unique(x1)
        y1 = np.unique(y1)
        if x==[]:
            ZXpt1[i] = math.nan
            Bp1[i] = math.nan
                
        elif np.floor(len(x1)/2)==len(x1)/2:
            ZXpt1[i] = math.nan
            Bp1[i] = math.nan
    Rout[0] = R[np.nanargmin(Bp0)]
    Rout[1] = R[np.nanargmin(Bp1)]
    Zout[0] = ZXpt0[np.nanargmin(Bp0)]
    Zout[1] = ZXpt1[np.nanargmin(Bp1)]    
    return Rout, Zout
seed(1)

plt.close('all')

colors = ['r', 'b', 'g', 'm', 'c']
shotnumber = [192024, 186841, 195508]
# shotnumber = [192024, 186841]
# shotnumber = [192024]
# shotnumber = [180520]
lab = ['favorable configuration', 'unfavorable configuration', 'neg-T']
# shot = 195508
# shot = 192024

g = np.zeros(len(shotnumber))
ks0 = np.zeros(len(shotnumber))
L_par = np.zeros(len(shotnumber))

time = 2000
i0 = -1
for shot in shotnumber:
    i0 = i0+1
    Filter = {'core': {'redchisq_limit': 10.0,'frac_temp_err_hot_max': 0.3,
                        'frac_temp_err_cold_max': 0.95,
                        'frac_dens_err_max': 0.3}}
    ts = OMFITthomson('DIII-D', shot, 'EFIT01', -1, ['core', 'tangential'], quality_filters=Filter)
    ts()
    eq0 = ts['efit_data']#read_basic_eq_from_mds(device='DIII-D', shot=shotnumber, tree='EFIT01', quiet=False, toksearch_mds=None)
    
    eq = eq0
    
    
    R = eq['r']
    Z = eq['z']
    
    ind = np.nanargmin(np.abs(eq0['atime']-time))
    R0 = np.mean(eq['rmaxis'][ind])
    Z0 = np.mean(eq['zmaxis'][ind])
    
    psi = eq0['psirz'][ind,:,:]
    psin = eq0['psin'][ind,:,:]
    
    for i in range(len(R)):
        for j in range(len(Z)):
            Rpt = np.arange(R[i], 3.0, 1e-3)
            Zpt = Rpt * 0.0 +Z[j]
            x, y = intersection(Rpt, Zpt,eq['lim'][:,0], eq['lim'][:,1])
            x = np.unique(x)
            y = np.unique(y)
            if x==[]:
                psi[j,i] = 2.0+random()
                psin[j,i] = 2.0+random()
                
            elif np.floor(len(x)/2)==len(x)/2:
                psi[j,i] = 2.0+random()
                psin[j,i] = 2.0+random()
    
    
    R_int = np.arange(np.nanmin(R), np.nanmax(R), 6e-4)
    Z_int = np.arange(np.nanmin(Z), np.nanmax(Z), 6e-4)
    
    psi_interp = interp2d(R, Z, psi)
    psi_int = psi_interp(R_int, Z_int)
    
    
    R_int2D, Z_int2D = np.meshgrid(R_int,Z_int)
    
    BT = eq0['bcentr'][ind]* R0 / R_int2D #  np.abs(database['mean']['BT']) * R0 / R_int2D
    
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
    
    
    Rxpt, Zxpt = find_Xpt(R_int, Z_int, np.sqrt(BR**2+BZ**2), eq['lim'][:,0], eq['lim'][:,1])
    
    plt.figure()
    cs = plt.contour(R, Z, psin, levels = [1])
    for item in cs.collections:
       for i in item.get_paths():
          v = i.vertices
          R_sep = v[:, 0]
          Z_sep = v[:, 1]
    plt.close()
    
    
    dR_FT = -1e-2
    r_FT = 5e-3
    
    ind_OMP = np.where(R_sep>=R0)[0]
    ind_OMP = ind_OMP[np.argmin(np.abs(Z_sep[ind_OMP]-Z0))]
    R0_FT = R_sep[ind_OMP]+dR_FT
    
    n_FL = 50
    
    theta = np.linspace(0.0, 1.999*math.pi, n_FL)
    
    
    database = {}
    
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='ZVSOUT')
    data0 = {}
    data0['time'] = tmp2.dim_of(0)
    data0['data'] = tmp2.data()
    
    ind0 = np.nanargmin(np.abs(data0['time']-time))
    database['Z_OSP'] = data0['data'][ind0]
    
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='Q95')
    data0 = {}
    data0['time'] = tmp2.dim_of(0)
    data0['data'] = tmp2.data()
    
    ind0 = np.nanargmin(np.abs(data0['time']-time))
    database['q95'] = data0['data'][ind0]
    
    
    plt.figure(int(i0+5))
    plt.contour(R, Z, psin, levels = [1], colors = ['#C0C0C0'])
    plt.contour(R, Z, psin, levels = [0.96,0.98,1.02], colors=['#808080','#808080','#808080'])
    plt.plot(eq['lim'][:,0], eq['lim'][:,1], 'k')
    plt.plot(Rxpt, Zxpt, 'xg')
    plt.axis('equal')   
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    # plt.xlim((1.0,3.0))

    # par_step = 0.005
    par_step = 0.005
    n_par = 2*math.pi*database['q95']*R0/par_step
    if database['Z_OSP']<0:
        n_par_step_down = int(np.floor(0.3*n_par))
        n_par_step_up = int(np.floor(0.7*n_par))
    else:
        n_par_step_up = int(np.floor(0.3*n_par))
        n_par_step_down =int(np.floor(0.7*n_par))
    # if database['mean']['Z_OSP']<0:
    #     n_par_step_down = 2200
    #     n_par_step_up = 4700
    # else:
    #     n_par_step_up = 3000
    #     n_par_step_down = 5800
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
    
    plt.plot(R_FT[n_par_step_down,:], Z_FT[n_par_step_down,:] ,'k')
     
    
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
        if np.mod(i,100)==0:    
            plt.figure(int(i0+5))
            plt.plot(R_FT[n_par_step_down+i+1,:], Z_FT[n_par_step_down+i+1,:] ,'r')   
            
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
        if np.mod(i,100)==0:  
            plt.figure(int(i0+5))
            plt.plot(R_FT[n_par_step_down-i-1,:], Z_FT[n_par_step_down-i-1,:] ,'b')
            
            
            
    thetaxpt = np.zeros(2)
    for i in range(len(Zxpt)):
        # if ((np.sign(Zxpt[i]-Z0)<0) and (np.sign(Rxpt[i]-R0)<0)):
        #     add = 2* math.pi
        # elif ((np.sign(Zxpt[i]-Z0)<0) and (np.sign(Rxpt[i]-R0)<0)):
        # else:
        #     add = 0
        
        thetaxpt[i] = np.angle(1j*(Zxpt[i]-Z0)+(Rxpt[i]-R0))
    
    
        
    
    
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.plot(theta_FT, G0, '-o'+colors[i0], label=lab[i0])
    for i in range(len(Rxpt)):
        plt.plot([thetaxpt[i],thetaxpt[i]], [-2,2], '--'+colors[i0])
        plt.legend()
    plt.grid(visible=True)
    plt.xlabel('Local azimutal coordinate [rad]', fontsize=12)
    plt.ylabel('G0', fontsize=12)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.figure(2)
    plt.subplot(1,3,2)
    plt.plot(theta_FT, ks, '-o'+colors[i0])
    plt.plot(theta_FT,uniform_filter1d(ks, size = 150), 'k')
    for i in range(len(Rxpt)):
        plt.plot([thetaxpt[i],thetaxpt[i]], [-20,20], '--'+colors[i0])
    plt.xlabel('Local azimutal coordinate [rad]', fontsize=12)
    plt.ylabel(r'$\alpha_{s}$', fontsize=12)
    plt.grid(visible=True)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    
    plt.figure(2)
    
    # g = np.mean(G0)
    
    # ks0 = np.mean(ks)
    
    # print(np.mean(uniform_filter1d(ks[ks<=6], size = 150)))
    
    if database['Z_OSP']<0:
        ind = np.where(G0<0)[0]
        # ind = ind[np.where(np.abs(ks[ind])<=6)[0]]
        g[i0] = np.mean(uniform_filter1d(G0[0:np.min(ind)], size = 150))
        ks0[i0] = np.mean(uniform_filter1d(ks[0:np.min(ind)]*G0[0:np.min(ind)]/np.mean(G0[0:np.min(ind)]), size = 150))
        L_par[i0] = -s[0]
        plt.subplot(1,3,3)
        plt.plot(theta_FT[0:np.min(ind)],ks[0:np.min(ind)]*G0[0:np.min(ind)]/np.mean(G0[0:np.min(ind)]), '-o'+colors[i0])
        for i in range(len(Rxpt)):
            plt.plot([thetaxpt[i],thetaxpt[i]], [-3,3], '--'+colors[i0])
        plt.xlabel('Local azimutal coordinate [rad]', fontsize=12)
        plt.ylabel(r'$\alpha_{s}G0/<G0>$', fontsize=12)
        plt.grid(visible=True)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
    else:
        ind = np.where(G0<0)[0]
        # ind = ind[np.where(np.abs(ks[ind])<=6)[0]]
        g[i0] = np.mean(uniform_filter1d(G0[np.max(ind):], size = 150))
        ks0[i0] = np.mean(uniform_filter1d(ks[np.max(ind):]*G0[np.max(ind):]/np.mean(G0[np.max(ind):]), size = 150))
        L_par[i0] = s[-1]
        plt.subplot(1,3,3)
        plt.plot(theta_FT[np.max(ind):], ks[np.max(ind):]*G0[np.max(ind):]/np.mean(G0[np.max(ind):]),'-o'+colors[i0])
        for i in range(len(Rxpt)):
            plt.plot([thetaxpt[i],thetaxpt[i]], [-3,3], '--'+colors[i0])
        plt.xlabel('Local azimutal coordinate [rad]', fontsize=12)
        plt.ylabel(r'$\alpha_{s}G0/<G0>$', fontsize=12)
        plt.grid(visible=True)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
print(ks0)
print(g)
print(L_par)

lncorr = (2/3*g)**(3/11)*(1.0+(-0.3+ks0)**2)**(-9/11) * (L_par/L_par[1])**(6/11)
print(lncorr/lncorr[1])

alpha_ExB = np.linspace(-3.0, 3.0, 100)
X1 = 0.0 * alpha_ExB + 0.0
X2 = 0.0 * alpha_ExB + 0.0
for i in range(len(alpha_ExB)):
    X1[i] = np.mean(uniform_filter1d((1+(ks[0:np.min(ind)]+alpha_ExB[i])**2)**(-9/11)*G0[0:np.min(ind)]/np.mean(G0[0:np.min(ind)]), size = 150))
    X2[i] = (1+(ks0[i0]+alpha_ExB[i])**2)**(-9/11)

plt.figure()
plt.plot(alpha_ExB, X1, '-o', label=r'$<(1+(\alpha_{s}+\alpha_{E{\times}B})^{2})^{-9/11}>_{\parallel}$')
plt.plot(alpha_ExB, X2, '-o', label=r'$(1+(<\alpha_{s}>_{\parallel}+\alpha_{E{\times}B})^{2})^{-9/11}$')
plt.xlabel(r'$\alpha_{E{\times}B}$')
plt.ylabel('Shear correction')
plt.grid('on')
plt.legend()

    # fig, ax =plt.subplots(1)
    # im = ax.pcolormesh(R_int,Z_int,psi_int, cmap='plasma', shading='auto', vmin=psi.min(), vmax=psi.max())
    # fig.colorbar(im,ax=ax)
    # plt.title(r'$\Psi$')
    # ax.set_xlabel('R [m]')
    # ax.set_ylabel('Z [m]')
    # ax.contour(R, Z, psi, levels = [1])