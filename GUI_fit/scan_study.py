#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:12:13 2023

@author: peretm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:23:38 2023

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
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', expand=1)


class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    # toolitems = [t for t in NavigationToolbar2Tk.toolitems if
    #              t[0] in ('Home', 'Pan', 'Zoom')]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    toolitems = [t for t in NavigationToolbar2Tk.toolitems]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)




def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
def init_fig_maker():
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(111)
    ax.plot([],[],'bo')
    ax.grid(color='k', linestyle='-')       
    return fig



def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
def init_fig_maker():
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(111)
    ax.plot([],[],'bo')
    ax.grid(color='k', linestyle='-')       
    return fig

def fig_maker(database):
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(141)
    ax.plot(database['Ip']['time'],database['Ip']['data']*1e-6,'r', label = 'Ip [MA]')
    ax.legend()
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel('time [ms]')
    ax.set_xlim((-100,7000))
    
    ax1 = fig.add_subplot(142)
    ax1.plot(database['nl']['time'],database['nl']['data']*1e-13,'b', label = r'$n_{lid} [\times 10^{19}m^{-3}]$')    
    ax1.legend()
    ax1.grid(color='k', linestyle='-')       
    ax1.set_xlabel('time [ms]')
    ax1.set_xlim((-100,7000))

    
    
    if len(database['PICH']['time'])!=1:
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
    ax3 = fig.add_subplot(143) 
    ax3.plot(time, Ptot,'c', label = r'$P_{tot} [MW]$')
    ax3.legend()
    ax3.grid(color='k', linestyle='-')       
    ax3.set_xlabel('time [ms]')
    ax3.set_xlim((-100,7000))
    
    ax4 = fig.add_subplot(144) 
    ax4.plot(database['tauth']['time'],database['tauth']['data']*1e3,'m', label = r'$\tau_{th} [ms]$')
    ax4.legend()
    ax4.grid(color='k', linestyle='-')       
    ax4.set_xlabel('time [ms]')
    ax4.set_xlim((-100,7000))  
    return fig


def open_window():
    layout = [[sg.Text("New Window", key="new")]]
    window = sg.Window("Second Window", layout, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

#matplotlib.figure.Figure(figsize=figsize)
# dpi = fig.get_dpi()
      # canvas size
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
def win_average(database):
    if len(database['PICH']['time'])!=1:
        if len(database['PECH']['time'])!=1:
            time0 = np.max([1500, np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([1500, np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([1500, np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([1500, np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
    # if len(database['PICH']['time'])!=1:
    #     time0 = np.max([2000, np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
    #     time1 = np.min([6000, np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
    #     time = np.linspace(time0, time1, 3000)
        
    #     Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    # else : 
    #     time0 = np.max([2000, np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
    #     time1 = np.min([6000, np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
    #     time = np.linspace(time0, time1, 3000)
        
    #     Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
    # nl = np.interp(time,database['nl']['time'],database['nl']['data']*1e-13) 
    # Ip = np.interp(time, database['Ip']['time'],database['Ip']['data']*1e-6)
    # tauth =  np.interp(time, database['tauth']['time'],database['tauth']['data'])
    win_dt = 150
    win_size = int(5 * np.floor(win_dt/np.mean(np.diff(database['nl']['time']))))
    nl = np.interp(time,database['nl']['time'],signal.savgol_filter(database['nl']['data']*1e-13,win_size,3))
    win_size = int(5 * np.floor(win_dt/np.mean(np.diff(database['Ip']['time']))))
    Ip = np.interp(time,database['Ip']['time'],signal.savgol_filter(database['Ip']['data']*1e-6,win_size,3))
    win_size = int(5 * np.floor(win_dt/np.mean(np.diff(database['tauth']['time']))))
    tauth = np.interp(time,database['tauth']['time'],signal.savgol_filter(database['tauth']['data'],win_size,3))
    
    
    n_win = int(np.floor((time1-time0)/win_dt))
    
    t_av = []
    nl_av = []
    Ip_av = []
    Ptot_av = []
    tauth_av = []
    
    
    for i in range(n_win):
        t_av.append((i+0.5)*win_dt+ time0)
        ind = np.where(time>=i*win_dt+time0)[0]
        ind = ind[np.where(time[ind]<=(i+1)*win_dt+time0)[0]]
        nl_av.append(np.mean(nl[ind]))
        Ip_av.append(np.mean(Ip[ind]))
        Ptot_av.append(np.mean(Ptot[ind]))
        tauth_av.append(np.mean(tauth[ind]))
    return t_av, nl_av, Ip_av, Ptot_av, tauth_av
    
    
    
def fig_maker_av(database, qtity):
    
    if qtity=='PTOT':
        if len(database['PICH']['time'])!=1:
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else: 
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        A = Ptot
    else:
        A = database[qtity]['data']*10**(-(np.floor(math.log10(np.max(database[qtity]['data'])))))
        time = database[qtity]['time']
        
    dt_min = 400

    di_min = int(np.floor(dt_min/np.mean(np.diff(time))))
    dt_smooth = 150
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(time))))
    
    # calculate smooth gradients
    A_av = uniform_filter1d(A, size = di_smooth)
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(111)
    
    if qtity=='PTOT':
        ax.plot(time, A, 'r', label = qtity + ' [MW]')
        ax.plot(time, A_av, '--k', label = 'smoothed')
    else: 
        ax.plot(database[qtity]['time'],database[qtity]['data']*10**(-(np.floor(math.log10(np.max(database[qtity]['data']))))),'r', label = qtity + database[qtity]['data_units'])
        ax.plot(database[qtity]['time'], A_av, '--k', label = 'smoothed')
    ax.legend()
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel('time [ms]')
    ax.set_xlim((-100,7000))
    
    
    
    return fig        

def fig_maker_av_bounds(database, qtity, t0, t1):
    
    if qtity=='PTOT':
        if len(database['PICH']['time'])!=1:
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else: 
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        A = Ptot
    else:
        A = database[qtity]['data']*10**(-(np.floor(math.log10(np.max(database[qtity]['data'])))))
        time = database[qtity]['time']
        
    dt_min = 400

    di_min = int(np.floor(dt_min/np.mean(np.diff(time))))
    dt_smooth = 150
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(time))))
    
    # calculate smooth gradients
    A_av = uniform_filter1d(A, size = di_smooth)
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(111)
    
    if qtity=='PTOT':
        ax.plot(time, A, 'r', label = qtity + ' [MW]')
        ax.plot(time, A_av, '--k', label = 'smoothed')
    else: 
        ax.plot(database[qtity]['time'],database[qtity]['data']*10**(-(np.floor(math.log10(np.max(database[qtity]['data']))))),'r', label = qtity + database[qtity]['data_units'])
        ax.plot(database[qtity]['time'], A_av, '--k', label = 'smoothed')
        
    for i in range(len(t0)):
        ax.plot([t0[i], t0[i]], [0.0, 1.05*np.max(A)], '--b')
        ax.plot([t1[i], t1[i]], [0.0, 1.05*np.max(A)], '--b')
        
    ax.legend()
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel('time [ms]')
    ax.set_xlim((-100,7000))
    
    
    
    return fig        

def fig_maker_plateau_bound(database, t0, t1):
    # plt.clf()
    # plt.close()
    
    t_av, nl_av, Ip_av, Ptot_av, tauth_av = win_average(database)
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    
    ax = fig.add_subplot(141)
    ax.plot(database['Ip']['time'],database['Ip']['data']*1e-6,'r', label = 'Ip [MA]')
    ax.plot(t_av, Ip_av, '--k', label = 'window average')
    for i in range(len(t0)):
        ax.plot([t0[i], t0[i]], [0.0, 1.05*np.max(database['Ip']['data']*1e-6)], '--r')
        ax.plot([t1[i], t1[i]], [0.0, 1.05*np.max(database['Ip']['data']*1e-6)], '--r')
    ax.legend()
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel('time [ms]')
    ax.set_xlim((-100,7000))
    
    ax1 = fig.add_subplot(142)
    ax1.plot(database['nl']['time'],database['nl']['data']*1e-13,'b', label = r'$n_{lid} [\times 10^{19}m^{-3}]$')    
    ax1.plot(t_av, nl_av, '--k', label = 'window average')
    for i in range(len(t0)):
        ax1.plot([t0[i], t0[i]], [0.0, 1.05*np.max(database['nl']['data']*1e-13)], '--r')
        ax1.plot([t1[i], t1[i]], [0.0, 1.05*np.max(database['nl']['data']*1e-13)], '--r')
    ax1.legend()
    ax1.grid(color='k', linestyle='-')       
    ax1.set_xlabel('time [ms]')
    ax1.set_xlim((-100,7000))

    
    
    if len(database['PICH']['time'])!=1:
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 3000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
    ax3 = fig.add_subplot(143) 
    ax3.plot(time, Ptot,'c', label = r'$P_{tot} [MW]$')
    ax3.plot(t_av, Ptot_av, '--k', label = 'window average')
    for i in range(len(t0)):
        ax3.plot([t0[i], t0[i]], [0.0, 1.05*np.max(Ptot)], '--r')
        ax3.plot([t1[i], t1[i]], [0.0, 1.05*np.max(Ptot)], '--r')
    ax3.legend()
    ax3.grid(color='k', linestyle='-')       
    ax3.set_xlabel('time [ms]')
    ax3.set_xlim((-100,7000))
    
    ax4 = fig.add_subplot(144) 
    ax4.plot(database['tauth']['time'],database['tauth']['data']*1e3,'m', label = r'$\tau_{th} [ms]$')
    ax4.plot(t_av, np.array(tauth_av)*1e3, '--k', label = 'window average')
    for i in range(len(t0)):
        ax4.plot([t0[i], t0[i]], [0.0, 1.05*np.max(database['tauth']['data']*1e3)], '--r')
        ax4.plot([t1[i], t1[i]], [0.0, 1.05*np.max(database['tauth']['data']*1e3)], '--r')
    ax4.legend()
    ax4.grid(color='k', linestyle='-')       
    ax4.set_xlabel('time [ms]')
    ax4.set_xlim((-100,7000))    
    
    return fig    


def get_step_bounds(database, qtity, time0, time1, A_step):
    if qtity=='PTOT':
        if len(database['PICH']['time'])!=1:
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else: 
            if len(database['PECH']['time'])!=1:
                time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
            else:
                time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
                time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
            
                time = np.linspace(time0, time1, 3000)
            
                Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        A = Ptot
    else:
        A = database[qtity]['data']*10**(-(np.floor(math.log10(np.max(database[qtity]['data'])))))
        time = database[qtity]['time']
    dt_min = 400

    di_min = int(np.floor(dt_min/np.mean(np.diff(time))))
    dt_smooth = 150
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(time))))
    
    # calculate smooth gradients
    A_av = uniform_filter1d(A, size = di_smooth)
    ind = np.where(time>=time0)[0]
    ind = ind[np.where(time[ind]<=time1)[0]]
    
    Amin = np.nanmin(A_av[ind])
    Amax = np.nanmax(A_av[ind])
    
    n_step = int(np.floor((Amax-Amin)/A_step))
    
    t0 = np.zeros(n_step)
    t1 = np.zeros(n_step)
    
    for i in range(n_step):
        ind0 = np.argmin(np.abs(A_av[ind]-(Amin+i*A_step)))
        ind1 = np.argmin(np.abs(A_av[ind]-(np.min([Amin+(i+1)*A_step, Amax]))))
        t0[i] = time[ind[ind0]]
        t1[i] = time[ind[ind1]]
    
    
    return t0, t1
    
    
def scan_bounds(database):

    win_height = get_monitors()[0].height
    win_width = get_monitors()[0].width

    matplotlib.use('TkAgg')
    w, h = size = (np.floor(0.4*win_width), np.floor(0.4*win_height))     # figure size

    fig = init_fig_maker()
    names =['nl', 'Ip', 'Pohm', 'P_NBI', 'PTOT']
    column1 = [[sg.Canvas(size=size, key='-CANVAS-')],
              [sg.Canvas(key='controls_cv')],
             [sg.Cancel(),sg.Text('Choose a quantity to plot'),sg.Combo(names, font=('Arial Bold', 10),  expand_x=True, enable_events=True,  readonly=False, key='-COMBO-'), sg.Button('Plot time trace')]]
    layout = [ [sg.Column(column1),
          sg.VSeperator(),
     sg.Column([[sg.Text('Enter min time of scan'), sg.InputText()],
                [sg.Text('Enter max time of scan'), sg.InputText()],
                [sg.Text('Enter amplitude of steps'), sg.InputText()],
                [sg.Button('Step bounds')],])]]
    window = sg.Window('Plot and get step bounds', layout, finalize=True, element_justification='center', font='Helvetica 12')
   
    
    

    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    # delete_fig_agg(fig_agg)
    #fig = fig_maker(database)
    # fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
    # plt.close()

    t0 = []
    t1 = []
    time0 = []
    time1 = []
    A_step = []
    qtity = []
    fig_maker(database)
    while True:             
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        elif event in ('Plot time trace'):    
            if values['-COMBO-']=='':
                print('Choose quantity to plot.')
            else:
                qtity = values['-COMBO-']
            # delete_fig_agg(fig_agg)
            
            
            ts_av = []
            fig = fig_maker_av(database, qtity)
            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            plt.close()
        elif event in ('Step bounds'):        
            if values[1]=='':
                print('Add a minimal value of time.') 
            else:
                time0 = float(values[1])
                
            if values[2]=='':
                print('Add a maximal value of time.')
            else:
                time1 = float(values[2])
            if values[3]=='':
                print('Add a step for scan decomposition.')
            else:
                A_step = float(values[3])
            if qtity==[]:
                print('Plot the wanted quantity.')
            if ((time0!=[]) and (time1!=[]) and (A_step!=[]) and (qtity!=[])):    
                t0, t1 = get_step_bounds(database, qtity, time0, time1, A_step)
                fig = fig_maker_av_bounds(database, qtity, t0, t1)
                fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
                plt.close()
            
            # t0 = 2500
            # t1 = 3000
    
    window.close()
    print(str(len(t0)) + ' plateaus have been found')
    return t0, t1
            
            
