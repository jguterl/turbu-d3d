#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:07:21 2023

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
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
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
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([1500, np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([1500, np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([1500, np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([6000, np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
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
    
    
    
def fig_maker_av(database):
    # plt.clf()
    # plt.close()
    
    t_av, nl_av, Ip_av, Ptot_av, tauth_av = win_average(database)
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(141)
    ax.plot(database['Ip']['time'],database['Ip']['data']*1e-6,'r', label = 'Ip [MA]')
    ax.plot(t_av, Ip_av, '--k', label = 'window average')
    ax.legend()
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel('time [ms]')
    ax.set_xlim((-100,7000))
    
    ax1 = fig.add_subplot(142)
    ax1.plot(database['nl']['time'],database['nl']['data']*1e-13,'b', label = r'$n_{lid} [\times 10^{19}m^{-3}]$')    
    ax1.plot(t_av, nl_av, '--k', label = 'window average')
    ax1.legend()
    ax1.grid(color='k', linestyle='-')       
    ax1.set_xlabel('time [ms]')
    ax1.set_xlim((-100,7000))

    
    # breakpoint()
    if len(database['PICH']['time'])!=1:
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        
    ax3 = fig.add_subplot(143) 
    ax3.plot(time, Ptot,'c', label = r'$P_{tot} [MW]$')
    ax3.plot(t_av, Ptot_av, '--k', label = 'window average')
    ax3.legend()
    ax3.grid(color='k', linestyle='-')       
    ax3.set_xlabel('time [ms]')
    ax3.set_xlim((-100,7000))
    
    ax4 = fig.add_subplot(144) 
    ax4.plot(database['tauth']['time'],database['tauth']['data']*1e3,'m', label = r'$\tau_{th} [ms]$')
    ax4.plot(t_av, np.array(tauth_av)*1e3, '--k', label = 'window average')
    ax4.legend()
    ax4.grid(color='k', linestyle='-')       
    ax4.set_xlabel('time [ms]')
    ax4.set_xlim((-100,7000))    
    
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
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PICH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PICH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PICH']['time'], database['PICH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
    else: 
        if len(database['PECH']['time'])!=1:
            time0 = np.max([np.min(database['Pohm']['time']), np.min(database['PECH']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']), np.max(database['PECH']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
            Ptot = np.interp(time, database['Pohm']['time'], database['Pohm']['data']*1e-6) + np.interp(time, database['PECH']['time'], database['PECH']['data']) + np.interp(time, database['PNBI']['time'], database['PNBI']['data']*1e-3) 
        else:
            time0 = np.max([np.min(database['Pohm']['time']),np.min(database['PNBI']['time'])])
            time1 = np.min([np.max(database['Pohm']['time']),np.max(database['PNBI']['time'])])
        
            time = np.linspace(time0, time1, 1000)
        
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


def get_plateau(t_nl, nl, t_Ip, Ip,  t_tauth, tauth):
    
    tol_nl = 1e-4  # Good for nl 
    tol_Ip = 4e-5
    # tol_Ptot = 0.006 # Good for Pohm
    tol_tau = 0.03  # Good for tauth
    dt_min = 400

    di_min = int(np.floor(dt_min/np.mean(np.diff(t_nl))))
    dt_smooth = 300
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(t_nl))))

    plateaus_nl, dF, d2F, smoothdata = find_plateaus(
        nl, min_length=di_min, tolerance=tol_nl, smoothing=di_smooth)
    t0nl = []
    t1nl = []

    if plateaus_nl != []:
        for i in range(len(plateaus_nl)):
            t0nl.append(t_nl[plateaus_nl[i][0]])
            t1nl.append(t_nl[plateaus_nl[i][1]])


    di_min = int(np.floor(dt_min/np.mean(np.diff(t_Ip))))
    dt_smooth = 250
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(t_Ip))))
    
    plateaus_Ip, dF, d2F, smoothdata = find_plateaus(Ip, min_length=di_min, tolerance = tol_Ip, smoothing=di_smooth)
    t0Ip = []
    t1Ip = []
    
    if plateaus_Ip!=[]:
        for i in range(len(plateaus_Ip)):
            t0Ip.append(t_Ip[plateaus_Ip[i][0]])
            t1Ip.append(t_Ip[plateaus_Ip[i][1]])
    
    di_min = int(np.floor(dt_min/np.mean(np.diff(t_tauth))))
    dt_smooth = 250
    di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(t_tauth))))
    
    plateaus_tau, dF, d2F, smoothdata = find_plateaus(tauth, min_length=di_min, tolerance = tol_tau, smoothing=di_smooth)
    t0tau = []
    t1tau = []
    
    if plateaus_tau!=[]:
        for i in range(len(plateaus_tau)):
            t0tau.append(t_tauth[plateaus_tau[i][0]])
            t1tau.append(t_tauth[plateaus_tau[i][1]])
            

    t00 = []
    t10 = []  
    for i in range(len(t0nl)):
        for j in range(len(t0Ip)):
            for k in range(len(t0tau)):
                if (np.max([t0nl[i], t0Ip[j], t0tau[k]])<np.min([t1nl[i], t1Ip[j], t1tau[k]])):
                    if (np.min([t1nl[i], t1Ip[j], t1tau[k]])-np.max([t0nl[i], t0Ip[j], t0tau[k]])>=dt_min):
                            t00.append(np.max([t0nl[i], t0Ip[j], t0tau[k]]))
                            t10.append(np.min([t1nl[i], t1Ip[j], t1tau[k]]))
  
    # breakpoint()
    # print('test')
    return t00, t10
    
def plateau_bounds(database):

    win_height = get_monitors()[0].height
    win_width = get_monitors()[0].width
    matplotlib.use('TkAgg')
    w, h = size = (np.floor(0.8*win_width), np.floor(0.7*win_height))     # figure size

    fig = init_fig_maker()
    layout = [ [sg.Canvas(size=size, key='-CANVAS-')],
          [sg.Canvas(key='controls_cv')],
          [sg.Cancel(), sg.Button('Find plateaus')]]
    window = sg.Window('Find plateaus', layout, finalize=True, element_justification='center', font='Helvetica 14')
    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    delete_fig_agg(fig_agg)
    fig = fig_maker(database)
    fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
    plt.close()

    t0 = []
    t1 = []
    fig_maker(database)
    while True:             
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        elif event in ('Find plateaus'):           
            # delete_fig_agg(fig_agg)
            fig = fig_maker_av(database)
            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            plt.close()
            t_av, nl_av, Ip_av, Ptot_av, tauth_av = win_average(database)
            
            t_nl = database['nl']['time']
            nl = database['nl']['data']*1e-13
            t_Ip = database['Ip']['time']
            Ip = database['Ip']['data']*1e-6
            t_tauth = database['tauth']['time']
            tauth = database['tauth']['data']*1e2
            
            t0, t1 = get_plateau(t_nl, nl, t_Ip, Ip, t_tauth, tauth)
            fig = fig_maker_plateau_bound(database, t0, t1)
            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            plt.close()
            
            # t0 = 2500
            # t1 = 3000
    
    window.close()
    print(str(len(t0)) + ' plateaus have been found')
    return t0, t1
            
            
