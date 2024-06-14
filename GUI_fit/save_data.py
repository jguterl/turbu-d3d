#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:36:12 2023

@author: peretm
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
from screeninfo import get_monitors
from get_DIIID_data import *
from plot_tools import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def save_data_plateau(t0, t1, shotnumber, ts_data, eq, data_fit, data, scan):
    win_height = get_monitors()[0].height
    win_width = get_monitors()[0].width

    matplotlib.use('TkAgg')
    w, h = size = (np.floor(0.8*win_width), np.floor(0.7*win_height))     # figure size
    names = []
    for i in ts_data.keys():
        names.append(str(int(i)+1))
    layout = [[sg.Cancel(),sg.Text('Choose plateau number'),sg.Combo(names, font=('Arial Bold', 10),  expand_x=True, enable_events=True,  readonly=False, key='-COMBO-'), sg.Button('Save data')]]
    window = sg.Window('Save data', layout, finalize=True, element_justification='center', font='Helvetica 14')
    
    while True:             
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event in ('Save data'):
            if values['-COMBO-']=='':
                print('Select a plateau number to save.')
            else:
                i_plateau = int(values['-COMBO-'])-1
                if data_fit[str(i_plateau)]=={}:
                    print('No fitted data to save.')
                else:
                    filename = './saved_plateaus/shot_' + str(int(shotnumber)) + '_t0_' + str(int(t0[i_plateau])) + '_t1_' + str(int(t1[i_plateau])) + scan + '.npy'
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
    window.close()
