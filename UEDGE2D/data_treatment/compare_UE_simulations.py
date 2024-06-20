#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:06:53 2023

@author: peretm
"""

import numpy as np
from plot_raw_data import plot_raw_data
from get_profiles import get_upstream_2target_2Xpt_profiles
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
warnings.filterwarnings("ignore")

def plot_remapped_compare_1D(sim1, sim2):
    '''
    Plot of the profiles at the target, Xpt and upstream remapped at the midplane

    Parameters
    ----------
    sim1 : simulation profiles
    sim2 : simulation profiles


    Returns
    -------
    None.

    '''
    symb = ['o', '+']
    label = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    ylabel = ['Electron density', 'Ion density', 'Neutral density', 'Electron temperature', 'Ion temperature', 'Plasma parallel velocity', 'Mach number']
    yunits = [r' $[m^{-3}]$', r' $[m^{-3}]$', r' $[m^{-3}]$', ' [eV]', ' [eV]', ' [m/s]', ' ']
    
    for i in range(len(label)):
        fig, ax = plt.subplots(1)
        for k in range(2):
            if (any(sim1['upstream'][label[i]]) and k==0):
                ax.plot(sim1['upstream']['rsep'],sim1['upstream'][label[i]],'b'+symb[k], label='upstream')
            if (any(sim2['upstream'][label[i]]) and k==1):
                ax.plot(sim2['upstream']['rsep'],sim2['upstream'][label[i]],'b'+symb[k], label='upstream')
                 
            if (any(sim1['target1'][label[i]]) and k==0):
                ax.plot(sim1['upstream']['rsep'], sim1['target1'][label[i]], 'r'+symb[k], label='target out')
            if (any(sim2['target1'][label[i]]) and k==1):
                ax.plot(sim2['upstream']['rsep'], sim2['target1'][label[i]], 'r'+symb[k], label='target out')
            
            if (any(sim1['Xpt1'][label[i]]) and k==0):
                ax.plot(sim1['upstream']['rsep'], sim1['Xpt1'][label[i]], 'g-'+symb[k], label='X-point in')  
            if (any(sim2['Xpt1'][label[i]]) and k==1):
                ax.plot(sim2['upstream']['rsep'], sim2['Xpt1'][label[i]], 'g-'+symb[k], label='X-point in')  
    
            if (any(sim1['target2'][label[i]]) and k==0):
                ax.plot(sim1['upstream']['rsep'], sim1['target2'][label[i]], 'r-'+symb[k], label='target in')    
            if (any(sim2['target2'][label[i]]) and k==1):
                ax.plot(sim2['upstream']['rsep'], sim2['target2'][label[i]], 'r-'+symb[k], label='target in')    
 
            if (any(sim1['Xpt2'][label[i]]) and k==0):
                ax.plot(sim1['upstream']['rsep'], sim1['Xpt2'][label[i]], 'g'+symb[k], label='X-point out')         
            if (any(sim2['Xpt2'][label[i]]) and k==1):
                ax.plot(sim2['upstream']['rsep'], sim2['Xpt2'][label[i]], 'g'+symb[k], label='X-point out')         
        
        ax.legend()
        ax.set_ylabel(ylabel[i]+yunits[i])
        ax.set_xlabel(r'$(r-r_{sep})_{upstream} [m]$')
        ax.grid(True) 

def plot_compare_2D(sim1, sim2):
    label = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    ylabel = ['Electron density', 'Ion density', 'Neutral density', 'Electron temperature', 'Ion temperature', 'Plasma parallel velocity', 'Mach number']
    yunits = [r' $[m^{-3}]$', r' $[m^{-3}]$', r' $[m^{-3}]$', ' [eV]', ' [eV]', ' [m/s]', ' ']
    
    for i in range(len(label)):
        fig1, ax = plt.subplots(1,2)
        for k in range(2):
            if (label[i] in sim1 and k==0):
                if (i<3):
                    im1 = ax[0].pcolormesh(sim1['R'],sim1['Z'], sim1[label[i]], cmap='hot', shading='auto', norm=colors.LogNorm(vmin=sim1[label[i]].min(), vmax=sim1[label[i]].max()))
                else:
                    im1 = ax[0].pcolormesh(sim1['R'],sim1['Z'], sim1[label[i]], cmap='hot', shading='auto', norm=colors.Normalize(vmin=sim1[label[i]].min(), vmax=sim1[label[i]].max()))
                fig1.colorbar(im1, ax=ax[0])
            if (label[i] in sim2 and k==1):
                if (i<3):
                    im2 = ax[1].pcolormesh(sim2['R'],sim2['Z'], sim2[label[i]],cmap='hot', shading='auto', norm=colors.LogNorm(vmin=sim2[label[i]].min(), vmax=sim2[label[i]].max()))
                else:
                    im2 = ax[1].pcolormesh(sim2['R'],sim2['Z'], sim2[label[i]],cmap='hot', shading='auto', norm=colors.Normalize(vmin=sim2[label[i]].min(), vmax=sim2[label[i]].max()))
                fig1.colorbar(im2, ax=ax[1])
            ax[0].title.set_text(ylabel[i]+yunits[i])
            ax[1].title.set_text(ylabel[i]+yunits[i])
            ax[0].set_xlabel('R [m]')
            ax[0].set_ylabel('Z [m]')
            ax[1].set_xlabel('R [m]')
            ax[1].set_ylabel('Z [m]')
            



file2load1 = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore1.00e+06/final_state_060723_170107.npy'
file2load2 ='/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/final_state_060723_185744.npy'

sim1 = plot_raw_data(file2load1, plot=False)
sim2 = plot_raw_data(file2load2, plot=False)
plot_compare_2D(sim1, sim2)

prof1 = get_upstream_2target_2Xpt_profiles(file2load1)
prof2 = get_upstream_2target_2Xpt_profiles(file2load2)
plot_remapped_compare_1D(prof1, prof2)

