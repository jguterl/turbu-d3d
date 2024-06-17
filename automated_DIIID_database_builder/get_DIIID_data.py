#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:35:33 2023

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
    data = ['ne0', 'Te0', 'Ti0', 'nl', 'Pohm', 'PNBI', 'PECH', 'PICH', 'Prad', 'R0', 'a', 'kappa', 'tri_up', 'tri_bot', 'tri', 'Rsep', 'H98', 'tauE', 'tauMHD', 'tauth', 'NEUT', 'q95', 'BT', 'Ip', 'puff', 'Prad_div', 'R_OSP', 'Z_OSP']
    flag = ['TSNE_CORE', 'TSTE_CORE', 'CERQTI', 'DENSITY', 'POH', 'PINJ', 'ECHPWR', 'ICHPWRC' , '.PRAD_01.POWER.BOL_U17_P', 'R0', 'AMINOR', 'KAPPA', 'TRITOP', 'TRIBOT', 'TRI', 'RMIDOUT', 'H_THH98Y2', 'TAUE', 'TAUMHD', 'TAUTH', 'FNR', 'Q95', 'BT', 'IP', 'GASA_CAL', 'PRAD_DIVL', 'RVSOUT', 'ZVSOUT']
    treename = [None, None, None, None, None, None, None, None, 'BOLOM', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    index = np.zeros(len(data))+1e6
    index[0] = -1
    index[1] = -1
    index[2] =  0

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
        ts = OMFITthomson('DIII-D', int(shotnumber), 'EFIT01', -1, ['core', 'tangential'], quality_filters=Filter)
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

        

            