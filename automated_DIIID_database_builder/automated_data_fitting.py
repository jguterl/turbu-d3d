# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
from screeninfo import get_monitors
from get_DIIID_data import *
from fit_functions import *
from scipy import signal
from save_data import *
import os

plt.close('all')
# shotnumber = int(169400 + np.floor(np.random.rand(1)*100))
shot_list = []
# for i in range(500):
#     shot_list.append(int(169000 + np.floor(np.random.rand(1)*1000)))

tmin = 3000
tmax = 5000
dt = 300
dn = 0.4e19

t0 = []
t1 = []

# t0.append(tmin)
# while t0[-1]<=tmax-2*dt:
#     t1.append(t0[-1]+dt)
#     t0.append(t1[-1])

# t1.append(t0[-1]+dt)

path = './saved_plateaus/'
dir_list = os.listdir(path)


for name in dir_list:
    if name[0:4]=='shot':
        database = np.load(path+name, allow_pickle=True).tolist()
        shot_list.append(int(name[5:11]))
        t0.append(database['t0'])
        t1.append(database['t1'])
j = 0
for shotnumber in shot_list:
    j=j+1
    print(str(j)+'/'+str(len(shot_list)))
    try:
        data = fetch_data(shotnumber)
        win_dt = 30
        time = np.linspace(tmin, tmax, 1000)
        win_size = int(5 * np.floor(win_dt/np.mean(np.diff(data['nl']['time']))))
        nl = np.interp(time,data['nl']['time'],signal.savgol_filter(data['nl']['data']*1e6,win_size,3))
        win_size = int(5 * np.floor(win_dt/np.mean(np.diff(data['nl']['time']))))
        H98 = np.interp(time,data['H98']['time'],signal.savgol_filter(data['H98']['data'],win_size,3))  
        n_win = int(np.floor((tmax-tmin)/win_dt))
        t_av = []
        nl_av = []
        H98_av = []
        for j in range(n_win):
            t_av.append((j+0.5)*win_dt+ tmin)
            ind_av = np.where(time>=j*win_dt+tmin)[0]
            ind_av = ind_av[np.where(time[ind_av]<=(j+1)*win_dt+tmin)[0]]
            nl_av.append(np.mean(nl[ind_av]))
            H98_av.append(np.mean(H98[ind_av]))
                    
        t_av = np.array(t_av)
        nl_av = np.array(nl_av)
        H98_av = np.array(H98_av)
        ts_data, eq = get_TS_data(shotnumber, t0, t1)
    except:
        []
        
    
    try:
        for i in range(len(t0)):
        
            ind = np.where(data['nl']['time']>=t0[i])[0]
            ind = ind[np.where(data['nl']['time'][ind]<t1[i])[0]]
            
            ind_av = np.where(t_av>=t0[i])[0]
            ind_av = ind_av[np.where(t_av[ind_av]<t1[i])[0]]
                   
            if (np.max(nl_av[ind_av])-np.min(nl_av[ind_av]))<=dn:
                ind = np.where(data['H98']['time']>=t0[i])[0]
                ind = ind[np.where(data['H98']['time'][ind]<t1[i])[0]]
                
                if np.min(H98_av[ind_av])>=1.0:
                    data_fit = study_TS_data(ts_data, eq, i)
                    save_data_plateau(t0, t1, i, shotnumber, ts_data, eq, data_fit, data, 'auto')
    
    except:
        []

    

