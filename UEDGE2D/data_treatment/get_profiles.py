#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:33:40 2023

@author: peretm
"""
import numpy as np
import matplotlib.pyplot as plt
import get_separatrix_data as get_sep

ee = 1.602e-19
mi = 1.67e-27
    
def get_upstream_target_Xpt_profiles(file2load, **kwargs):
    '''
    Get the radial profiles of physical quantities at upstream, X-point and target positions
    
    Parameters
    ----------
    file2load : string 
        Use the path to the simulation output file
    **kwargs : accepted key arguments are field and position

        position correspond to the position along the fieldline (upstream, Xpt, target) / by default all positions
        examples: position=['upstream'] or position=['upstream', 'Xpt']
    ---------   
        field correspond to the quantities of interest (ne, ni, ng, te, ti, up, M) / by default all quantities
        examples: field=['ne'] or field=['ne', 'M', 'up']
    ---------
    Returns
    -------
    output : Dictionary
        output has three items corresponding to the positions
        and then subitems correspond to the radial profiles of the quantities of interest

    '''
    
    sim = np.load(file2load, allow_pickle=True).tolist()
    sep = get_sep.get_separatrix_data(file2load)
    i_u, i_x, i_t = get_sep.get_uxt_indices(sep['R'],sep['Z'])
    output = {'upstream':{}, 'Xpt':{}, 'target':{}}
    label = {}
    label['te'] = 'bbb.te'
    label['ti'] = 'bbb.ti'
    label['ne'] = 'bbb.ni'
    label['ni'] = 'bbb.ni'
    label['ng'] = 'bbb.ng'
    label['up'] = 'bbb.up'
    
    output['upstream']['index'] = i_u
    output['Xpt']['index'] = i_x
    output['target']['index'] = i_t
    field = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    position = ['upstream', 'target', 'Xpt']   
    if any(kwargs.items()):
        for key, value in kwargs.items():
            if (key=='field'):
                field=value
            
            if (key=='position'):
                position=value


    try:
        for i in range(len(field)):
            for j in range(len(position)): 
                output[position[j]]['rsep'] = sim[3][0]['com.rm'][output[position[j]]['index'],:,0]-sep['R'][output[position[j]]['index']]
                
                if (field[i]=='M'):                   
                    output[position[j]][field[i]] = sim[3][0][label['up']][output[position[j]]['index'],:,0]/np.sqrt((sim[3][0][label['te']][output[position[j]]['index'],:]+sim[3][0][label['ti']][output[position[j]]['index'],:])/mi)
                
                elif (field[i]=='te' or field[i]=='ti'):
                    output[position[j]][field[i]] = sim[3][0][label[field[i]]][output[position[j]]['index'],:]/ee
                
                else:
                    output[position[j]][field[i]] = sim[3][0][label[field[i]]][output[position[j]]['index'],:,0]
    

    except:
        []

    return output

def plot_remapped(sim):
    '''
    Plot of the profiles at the target, Xpt and upstream remapped at the midplane

    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    label = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    ylabel = ['Electron density', 'Ion density', 'Neutral density', 'Electron temperature', 'Ion temperature', 'Plasma parallel velocity', 'Mach number']
    yunits = [r' [m^{-3}]', r' [m^{-3}]', r' [m^{-3}]', ' [eV]', ' [eV]', ' [m/s]', ' ']
    for i in range(len(label)):
        fig, ax = plt.subplots(1)
        if any(sim['upstream']['ne']):
            ax.plot(sim['upstream']['rsep'],sim['upstream'][label[i]],'bo', label='upstream')
        if any(sim['target']['ne']):
            ax.plot(sim['upstream']['rsep'], sim['target'][label[i]], 'ro', label='target')
        if any(sim['Xpt']['ne']):
            ax.plot(sim['upstream']['rsep'], sim['Xpt'][label[i]], 'go', label='X-point')  
        ax.legend()
        ax.set_ylabel(ylabel[i]+yunits[i])
        ax.set_xlabel('(r-rsep)_{upstream} [m]')
        ax.grid(True)
        

def get_upstream_2target_2Xpt_profiles(file2load, **kwargs):
    '''
    Get the radial profiles of physical quantities at upstream, X-point and target positions
    
    Parameters
    ----------
    file2load : string 
        Use the path to the the simulation output file
    **kwargs : accepted key arguments are field and position
        position correspond to the position along the fieldline (upstream, Xpt, target) / by default all positions
        examples: position=['upstream'] or position=['upstream', 'Xpt']
    ---------   
        field correspond to the quantities of interest (ne, ni, ng, te, ti, up, M) / by default all quantities
        examples: field=['ne'] or field=['ne', 'M', 'up']
    ---------
    Returns
    -------
    output : Dictionary
        output has three items corresponding to the positions
        and then subitems correspond to the radial profiles of the quantities of interest

    '''
    
    sim = np.load(file2load, allow_pickle=True).tolist()
    sep = get_sep.get_separatrix_data(file2load)
    i_u, i_x1, i_x2, i_t1, i_t2 = get_sep.get_u2x2t_indices(sep['R'],sep['Z'])
    output = {'upstream':{}, 'Xpt1':{}, 'Xpt2':{}, 'target1':{}, 'target2':{}}
    label = {}
    label['te'] = 'bbb.te'
    label['ti'] = 'bbb.ti'
    label['ne'] = 'bbb.ni'
    label['ni'] = 'bbb.ni'
    label['ng'] = 'bbb.ng'
    label['up'] = 'bbb.up'
    
    output['upstream']['index'] = i_u
    output['Xpt1']['index'] = i_x1
    output['Xpt2']['index'] = i_x2
    output['target1']['index'] = i_t1
    output['target2']['index'] = i_t2
    
    field = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    position = ['upstream', 'target1', 'target2', 'Xpt1', 'Xpt2']   
    if any(kwargs.items()):
        for key, value in kwargs.items():
            if (key=='field'):
                field=value
            
            if (key=='position'):
                position=value


    try:
        for i in range(len(field)):
            for j in range(len(position)): 
                output[position[j]]['rsep'] = sim[3][0]['com.rm'][output[position[j]]['index'],:,0]-sep['R'][output[position[j]]['index']]
                
                if (field[i]=='M'):                   
                    output[position[j]][field[i]] = sim[3][0][label['up']][output[position[j]]['index'],:,0]/np.sqrt((sim[3][0][label['te']][output[position[j]]['index'],:]+sim[3][0][label['ti']][output[position[j]]['index'],:])/mi)
                
                elif (field[i]=='te' or field[i]=='ti'):
                    output[position[j]][field[i]] = sim[3][0][label[field[i]]][output[position[j]]['index'],:]/ee
                
                else:
                    output[position[j]][field[i]] = sim[3][0][label[field[i]]][output[position[j]]['index'],:,0]
    

    except:
        []

    return output

def plot_remapped2(sim):
    '''
    Plot of the profiles at the target, Xpt and upstream remapped at the midplane

    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    label = ['ne', 'ni', 'ng', 'te', 'ti', 'up', 'M']
    ylabel = ['Electron density', 'Ion density', 'Neutral density', 'Electron temperature', 'Ion temperature', 'Plasma parallel velocity', 'Mach number']
    yunits = [r' $[m^{-3}]$', r' $[m^{-3}]$', r' $[m^{-3}]$', ' [eV]', ' [eV]', ' [m/s]', ' ']
    for i in range(len(label)):
        fig, ax = plt.subplots(1)
        if any(sim['upstream'][label[i]]):
            ax.plot(sim['upstream']['rsep'],sim['upstream'][label[i]],'bo', label='upstream')
        if any(sim['target1'][label[i]]):
            ax.plot(sim['upstream']['rsep'], sim['target1'][label[i]], 'ro', label='target out')
        if any(sim['Xpt1'][label[i]]):
            ax.plot(sim['upstream']['rsep'], sim['Xpt1'][label[i]], 'gs', label='X-point in')  
        if any(sim['target2'][label[i]]):
            ax.plot(sim['upstream']['rsep'], sim['target2'][label[i]], 'rs', label='target in')
        if any(sim['Xpt2'][label[i]]):
            ax.plot(sim['upstream']['rsep'], sim['Xpt2'][label[i]], 'go', label='X-point out')         
        
        ax.legend()
        ax.set_ylabel(ylabel[i]+yunits[i])
        ax.set_xlabel(r'$(r-r_{sep})_{upstream} [m]$')
        ax.grid(True)        
        
# #file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc1.00e+19_pcore1.00e+06/final_state_060723_120845.npy'
# #file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore1.00e+06/final_state_060723_170107.npy'
# file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/final_state_060723_185744.npy'
# #data = get_upstream_target_Xpt_profiles(file2load)
# data2 = get_upstream_2target_2Xpt_profiles(file2load)
# #plot_remapped(data)
# plot_remapped2(data2)