#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:00:38 2024

@author: peretm
"""
import numpy as np
def save_data_plateau(t0, t1, i_plateau, shotnumber, ts_data, eq, data_fit, data, scan):
    filename = './saved_plateaus/no_shift_2/shot_' + str(shotnumber) + '_t0_' + str(int(t0[i_plateau])) + '_t1_' + str(int(t1[i_plateau])) + scan + '.npy'
    data_save = {}
    data_save['t0'] = t0[i_plateau]
    data_save['t1'] = t1[i_plateau]
    data_save['mean'] = {}
    data_save['std'] = {}
    for i in data.keys():
        if data[i] != []: 
            data_save['mean'][i] = []
            data_save['std'][i] = []
            try:
    
                ind = np.where(data[i]['time']>=t0[i_plateau])[0]
                ind = ind[np.where(data[i]['time'][ind]<=t1[i_plateau])[0]]
                ind = ind[np.where(np.isnan(data[i]['data'][ind])!=1)[0]]
    
                data_save['std'][i] = np.std((data[i]['data'][ind]).astype(np.float64))
                data_save['mean'][i] = np.mean((data[i]['data'][ind]).astype(np.float64))
            except:
                []
            
    data_save['ts_data'] = []
    data_save['ts_data'] = ts_data[str(i_plateau)]
    data_save['eq'] = []
    data_save['eq'] = eq[str(i_plateau)]
    data_save['ts_fit'] = []
    data_save['ts_fit'] = data_fit[str(i_plateau)]
    
    np.save(filename, data_save, allow_pickle=True)
    print('data saved!')