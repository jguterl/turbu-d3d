#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:48:52 2023

@author: peretm
"""

import matplotlib.pyplot as plt
from omfit_classes.utils_fusion import calc_h98_d3d
from omfit_classes.omfit_eqdsk import read_basic_eq_from_mds
from omfit_classes.omfit_mds import OMFITmdsValue
import numpy as np
import math
from omfit_classes.omfit_thomson import OMFITthomson




def get_data(shotnumber, data, treename0, flag, index):
    if data=='tri':
        tmp2 = OMFITmdsValue('DIII-D', treename=treename0, shot=shotnumber, TDI='TRIBOT')
        tmp = OMFITmdsValue('DIII-D', treename=treename0, shot=shotnumber, TDI='TRITOP')
        
        data0 = {}
        data0['name'] = data
        data0['treename'] = treename0
        data0['shot'] = shotnumber
        data0['flag'] = flag
        if tmp2!=None:
            data0['time'] = tmp2.dim_of(0)
            data0['time_units'] = tmp2.units_dim_of(0)
            data0['data_units'] = tmp2.units()
            data0['data'] = 0.5 * (tmp2.data() + tmp.data())
    else:
        tmp2 = OMFITmdsValue('DIII-D', treename=treename0, shot=shotnumber, TDI=flag)
        
        data0 = {}
        
        data0['name'] = data
        data0['treename'] = treename0
        data0['shot'] = shotnumber
        data0['flag'] = flag
        data0['data'] = None
        if (tmp2.dim_of(0)) is not None:
            
            data0['time'] = tmp2.dim_of(0)
            data0['time_units'] = tmp2.units_dim_of(0)
            data0['data_units'] = tmp2.units()
            if index >1000:
                data0['data'] = tmp2.data()
            else:
                data0['data'] = tmp2.data()[index.astype(int),:]  
        
    return data0

def fetch_data(shotnumber):
    data = ['nl', 'Pohm', 'PNBI', 'PECH', 'PICH', 'R0', 'a', 'kappa', 'tri_up', 'tri_bot', 'tri', 'tauE', 'q95', 'BT', 'Ip', 'R_OSP', 'Z_OSP']
    flag = ['DENSITY', 'POH', 'PINJ', 'ECHPWR', 'ICHPWRC', 'R0', 'AMINOR', 'KAPPA', 'TRITOP', 'TRIBOT', 'TRI', 'TAUE', 'Q95', 'BT', 'IP', 'RVSOUT', 'ZVSOUT']
    treename = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    index = np.zeros(len(data))+1e6
    # index[0] = -1
    # index[1] = -1
    # index[2] =  0

    out = {}
    out['shot'] = []


    for i in range(len(data)):
        out[data[i]] = []
    
    out['ts_data'] = []
    out['eq'] = []
    for i in range(len(data)):    
        try:
            out[data[i]] = get_data(shotnumber, data[i], treename[i], flag[i], index[i])
        except:
            []
            # database[str(shotnumber)][data[i]]= []
            
    # Need to find a way to  get the divertor data !!!!!!!!!!!!!
    
    
    # try:
    #     Filter = {'core': {'redchisq_limit': 10.0,'frac_temp_err_hot_max': 0.3,
    #                     'frac_temp_err_cold_max': 0.95,
    #                     'frac_dens_err_max': 0.3}}
    #     ts = OMFITthomson('DIII-D', shotnumber, 'EFIT01', -1, ['divertor'], quality_filters=Filter)
    #     ts()
    #     # breakpoint()
    #     out['ts_data'] = ts['filtered']['divertor']
    #     out['eq'] = ts['efit_data']#read_basic_eq_from_mds(device='DIII-D', shot=shotnumber, tree='EFIT01', quiet=False, toksearch_mds=None)
            
    # except:
    #     out['ts_data'] = []
    #     out['eq'] = []
    return out

def get_TS_data(shotnumber, t0, t1):
    ts_data = {}
    eq = {}
    fields = ['temp', 'temp_e', 'density', 'density_e', 'psin_TS', 'time', 'r', 'z']
    fields_eq = ['r', 'z', 'psin', 'zmaxis', 'rmaxis', 'atime', 'lim']
    for i in range(len(t0)):
        ts_data[str(i)] = []
        eq[str(i)] = []
        
    
    try:

        Filter = {'core': {'redchisq_limit': 10.0,'frac_temp_err_hot_max': 0.3,
                    'frac_temp_err_cold_max': 0.95,
                    'frac_dens_err_max': 0.3}}
        ts = OMFITthomson('DIII-D', int(shotnumber), 'EFIT01', -1, ['core', 'tangential'])#, quality_filters=Filter)
        ts()
        # breakpoint()
        # print('test')

        for i in range(len(t0)):
            ind = np.where(ts['filtered']['core']['time']>=t0[i])[0]
            ind = ind[np.where(ts['filtered']['core']['time'][ind]<=t1[i])[0]]
            ts_data[str(i)] = {} 
            eq[str(i)] = {}
            for j in fields:
                if j in ('density', 'density_e', 'temp', 'temp_e', 'psin_TS'):
                    ts_data[str(i)][j] =  ts['filtered']['core'][j][:,ind]
                elif j in ('time'):
                    ts_data[str(i)][j] = ts['filtered']['core'][j][ind]
                elif j in ('r', 'z'):
                    ts_data[str(i)][j] = ts['filtered']['core'][j]
                    
            ind = np.where(ts['efit_data']['atime']>=t0[i])[0]
            ind = ind[np.where(ts['efit_data']['atime'][ind]<=t1[i])[0]]        
            for j in fields_eq:
                if j in ('psin'):
                    eq[str(i)][j] =  ts['efit_data'][j][ind,:,:]
                elif j in ('atime', 'zmaxis', 'rmaxis'):
                    eq[str(i)][j] = ts['efit_data'][j][ind]
                elif j in ('r', 'z'):
                    eq[str(i)][j] = ts['efit_data'][j]  
                elif j in ('lim'):
                    eq[str(i)][j] = ts['efit_data'][j]  
            # eq[str(i)] = ts['efit_data']#read_basic_eq_from_mds(device='DIII-D', shot=shotnumber, tree='EFIT01', quiet=False, toksearch_mds=None)
            
    except:
        for i in range(len(t0)):
            ts_data[str(i)] = []
            eq[str(i)] = []
    
    return ts_data, eq

def find_plateaus(F, min_length=200, tolerance = 0.75, smoothing=25):
    '''
    Finds plateaus of signal using second derivative of F.

    Parameters
    ----------
    F : Signal.
    min_length: Minimum length of plateau.
    tolerance: Number between 0 and 1 indicating how tolerant
        the requirement of constant slope of the plateau is.
    smoothing: Size of uniform filter 1D applied to F and its derivatives.
    
    Returns
    -------
    plateaus: array of plateau left and right edges pairs
    dF: (smoothed) derivative of F
    d2F: (smoothed) Second Derivative of F
    '''
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    
    # calculate smooth gradients
    smoothF = uniform_filter1d(F, size = smoothing)
    dF = uniform_filter1d(np.gradient(smoothF),size = smoothing)/smoothF
    d2F = uniform_filter1d(np.gradient(dF),size = smoothing)
    
    def zero_runs(x):
        '''
        Helper function for finding sequences of 0s in a signal
        https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
        '''
        iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges
    
    # Find ranges where second derivative is zero
    # Values under eps are assumed to be zero.
    # eps = np.quantile(abs(d2F),tolerance) 
    smalld2F = (abs(dF) <= tolerance)
    # Find repititions in the mask "smalld2F" (i.e. ranges where d2F is constantly zero)
    p = zero_runs(np.diff(smalld2F))
    
    
    # np.diff(p) gives the length of each range found.
    # only accept plateaus of min_length
    
    plat =[]
    try:
        plat = p[(np.diff(p) > min_length).flatten()]
    except:
        []
    plateaus = []
    
    if plat!=[]:
        for i in range(len(plat)):
            if np.abs(np.mean(dF[plat[i,0]:plat[i,1]]))<tolerance:
                plateaus.append(plat[i,:])
    
    return (plateaus, dF, d2F, smoothF)
        


###############################################################################
###### Build the database #####################################################
###############################################################################
tmin = 2000
tmax = 5500
shotmin = 180000
shotmax = 200000
shot = list(np.round(np.linspace(shotmin,shotmax,shotmax-shotmin)).astype(int))


database = {}
for i in shot:
    shotnumber = i
    database[str(i)] = {}
    try:
        out = fetch_data(shotnumber)

        ind = np.where(out['Ip']['time']>=tmin)[0]
        ind = ind[np.where(out['Ip']['time'][ind]<=tmax)[0]]

        plateaus, dF, d2F, smoothF = find_plateaus(out['Ip']['data'][ind], min_length=200, tolerance = 0.75, smoothing=25)

        t0 = []
        t1 = []
        t0.append(out['Ip']['time'][ind[plateaus[0][0]]])
        t1.append(out['Ip']['time'][ind[plateaus[0][1]]])

        out['ts_data'], out['eq'] = get_TS_data(shotnumber, t0, t1)
        
        for j in out:
            if ((j!='shot') and (j!='eq') and (j!='ts_data')):
                database[str(i)][j] = {}
                database[str(i)][j]['time'] = []
                database[str(i)][j]['data'] = []
                ind = np.where(out[j]['time']>=tmin)[0]
                ind = ind[np.where(out[j]['time'][ind]<=tmax)[0]]
                database[str(i)][j]['time'] = out[j]['time'][ind]
                database[str(i)][j]['data'] = out[j]['data'][ind]
            if (j=='ts_data'):
                database[str(i)][j] = {}
                database[str(i)][j] = out[j]['0']

        if (t1[0]<=tmax-200):
            database[str(i)]['valid'] = 0
        else:
            database[str(i)]['valid'] = 1
            
    except:
        del database[str(i)]
        continue
        
    
