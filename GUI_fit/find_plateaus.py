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

def get_plateau(t_av, nl_av, Ip_av, Ptot_av, tauth_av):
    
    t_av = np.array(t_av)
    nl_av = np.array(nl_av)
    Ip_av = np.array(Ip_av)
    Ptot_av = np.array(Ptot_av)
    tauth_av = np.array(tauth_av)
    t00 = []
    t10 = []  
    dt_min = 200
    # tau_max = 0.18
    di_min = np.floor(dt_min/np.mean(np.diff(t_av)))
    # plt.figure()
    # plt.plot(t_av[0:-1],2*1e3*np.abs(np.diff(nl_av)/np.diff(t_av))/(nl_av[0:-1]+nl_av[1:]), 'o')
    
    # plt.figure()
    # plt.plot(t_av[0:-1],np.abs(np.diff(Ip_av)/np.diff(t_av))/(Ip_av[0:-1]+Ip_av[1:])*2.0*1e3, 'o')
    
    # plt.figure()
    # plt.plot(t_av[0:-1],np.abs(np.diff(Ptot_av)/np.diff(t_av))/(Ptot_av[0:-1]+Ptot_av[1:])*2.0*1e3, 'o')
    
    # plt.figure()
    # plt.plot(t_av[0:-1],np.abs(np.diff(tauth_av)/np.diff(t_av))/(tauth_av[0:-1]+tauth_av[1:])*2.0*1e3, 'o')   
    # breakpoint()
    # print('text')
    # ind_nl = np.where(2*np.abs(np.diff(nl_av))/(nl_av[0:-1]+nl_av[1:])<= tau_max)[0]
    # ind_nlA = np.append(-1, np.where(np.diff(ind_nl)>2)[0])
    # if len(ind_nlA)==1:
    #     ind_nlA = np.append(ind_nlA, ind_nl[-1])
    # t0nl =[]
    # t1nl = []
    # for i in range(len(ind_nlA)-1):
    #     if (ind_nlA[i+1]-ind_nlA[i]-1)>=di_min:
    #         t0nl.append(ind_nl[ind_nlA[i]+1])
    #         t1nl.append(ind_nlA[i+1])
    
    ind_nl = np.where(2*1e3*np.abs(np.diff(nl_av)/np.diff(t_av))/(nl_av[0:-1]+nl_av[1:])<=0.18)[0]
    ind_nl0 = np.where(np.diff(ind_nl)>2)[0]
    ind_nlA = []
    ind_nlA.append(ind_nl[0])
    ind_nlB = []
    # ind_IpA.append(int(0))
    for i in range(len(ind_nl0)):
        ind_nlB.append(ind_nl[ind_nl0[i]])
        ind_nlA.append(ind_nl[ind_nl0[i]])
        # ind_IpA.append(ind_Ip[ind_Ip0[i+1]])
    
    ind_nlB.append(ind_nl[-1])
    # ind_IpA = np.append(-1, np.where(np.diff(ind_Ip)>2)[0])
    
    # if len(ind_IpA)==1:
    #     ind_IpA = np.append(ind_IpA, ind_Ip[-1])
        
    t0nl = []
    t1nl = []
    for i in range(len(ind_nlA)):
        if (ind_nlB[i]-ind_nlA[i])>=di_min:
            t0nl.append(t_av[ind_nlA[i]])
            t1nl.append(t_av[ind_nlB[i]])

    

    ind_Ip = np.where(2*1e3*np.abs(np.diff(Ip_av)/np.diff(t_av))/(Ip_av[0:-1]+Ip_av[1:])<=0.03)[0]
    ind_Ip0 = np.where(np.diff(ind_Ip)>2)[0]
    ind_IpA = []
    ind_IpA.append(ind_Ip[0])
    ind_IpB = []
    # ind_IpA.append(int(0))
    for i in range(len(ind_Ip0)):
        ind_IpB.append(ind_Ip[ind_Ip0[i]])
        ind_IpA.append(ind_Ip[ind_Ip0[i]])
        # ind_IpA.append(ind_Ip[ind_Ip0[i+1]])
    
    ind_IpB.append(ind_Ip[-1])
    # ind_IpA = np.append(-1, np.where(np.diff(ind_Ip)>2)[0])
    
    # if len(ind_IpA)==1:
    #     ind_IpA = np.append(ind_IpA, ind_Ip[-1])
        
    t0Ip = []
    t1Ip = []
    for i in range(len(ind_IpA)):
        if (ind_IpB[i]-ind_IpA[i])>=di_min:
            t0Ip.append(t_av[ind_IpA[i]])
            t1Ip.append(t_av[ind_IpB[i]])
            
            
    ind_tau = np.where(2*1e3*np.abs(np.diff(tauth_av)/np.diff(t_av))/(tauth_av[0:-1]+tauth_av[1:])<= 1.0)[0]
    ind_tau0 = np.where(np.diff(ind_tau)>2)[0]
    ind_tauA = []
    ind_tauA.append(ind_tau[0])
    ind_tauB = []
    # ind_IpA.append(int(0))
    for i in range(len(ind_tau0)):
        ind_tauB.append(ind_tau[ind_tau0[i]])
        ind_tauA.append(ind_tau[ind_tau0[i]])
        # ind_IpA.append(ind_Ip[ind_Ip0[i+1]])
    
    ind_tauB.append(ind_tau[-1])
    # ind_IpA = np.append(-1, np.where(np.diff(ind_Ip)>2)[0])
    
    # if len(ind_IpA)==1:
    #     ind_IpA = np.append(ind_IpA, ind_Ip[-1])
        
    t0tau = []
    t1tau = []
    for i in range(len(ind_tauA)):
        if (ind_tauB[i]-ind_tauA[i])>=di_min:
            t0tau.append(t_av[ind_tauA[i]])
            t1tau.append(t_av[ind_tauB[i]])

    # ind_tau = np.where(2*np.abs(np.diff(tauth_av))/(tauth_av[0:-1]+tauth_av[1:])<= 10.0 * tau_max)[0]
    # ind_tauA = np.append(-1, np.where(np.diff(ind_tau)>2)[0])
    # if len(ind_tauA)==1:
    #     ind_tauA = np.append(ind_tauA, ind_tau[-1])
    # t0tau = []
    # t1tau = []
    # for i in range(len(ind_tauA)-1):
    #     if (ind_tauA[i+1]-ind_tauA[i]-1)>=di_min:
    #         t0tau.append(ind_tau[ind_tauA[i]+1])
    #         t1tau.append(ind_tauA[i+1])

    # ind_Ptot = np.where(2*np.abs(np.diff(Ptot_av))/(Ptot_av[0:-1]+Ptot_av[1:])<= tau_max)[0]
    # ind_PtotA = np.append(-1, np.where(np.diff(ind_Ptot)>2)[0])
    # if len(ind_PtotA)==1:
    #     ind_PtotA = np.append(ind_PtotA, ind_Ptot[-1])
    # t0Ptot = []
    # t1Ptot = []
    # for i in range(len(ind_PtotA)-1):
    #     if (ind_PtotA[i+1]-ind_PtotA[i]-1)>=di_min:
    #         t0Ptot.append(ind_Ptot[ind_PtotA[i]+1])
    #         t1Ptot.append(ind_PtotA[i+1])
            
    # for i in range(len(t0nl)):
    #     for j in range(len(t0Ip)):
    #         for k in range(len(t0Ptot)):
    #             if (np.max([t0nl[i], t0Ip[j], t0Ptot[k]])<np.min([t1nl[i], t1Ip[j], t1Ptot[k]])):
    #                 if (t_av[np.min([t1nl[i], t1Ip[j], t1Ptot[k]])]-t_av[np.max([t0nl[i], t0Ip[j], t0Ptot[k]])]>=400):
    #                     t0.append(t_av[np.max([t0nl[i], t0Ip[j], t0Ptot[k]])])
    #                     t1.append(t_av[np.min([t1nl[i], t1Ip[j], t1Ptot[k]])])
                        
    for i in range(len(t0nl)):
        for j in range(len(t0Ip)):
            for k in range(len(t0tau)):
                if (np.max([t0nl[i], t0Ip[j], t0tau[k]])<np.min([t1nl[i], t1Ip[j], t1tau[k]])):
                    if (np.min([t1nl[i], t1Ip[j], t1tau[k]])-np.max([t0nl[i], t0Ip[j], t0tau[k]])>=dt_min):
                            t00.append(np.max([t0nl[i], t0Ip[j], t0tau[k]]))
                            t10.append(np.min([t1nl[i], t1Ip[j], t1tau[k]]))
    
    t0 = []
    t1 = []
    dnl = 0.9
    dIp = 0.2
    dtau = 1e-2
    if t00!=[]:
        for i_plateau in range(len(t00)):
            indplateau = np.where(t_av>=t00[i_plateau])[0]
            indplateau = indplateau[np.where(t_av[indplateau]<=t10[i_plateau])[0]]
            ind0 = indplateau[0]
            ind = ind0
            nlref = nl_av[ind0]
            Ipref = Ip_av[ind0]
            tauref = tauth_av[ind0]
            for i in range(len(indplateau)):
                if ((np.abs(nl_av[indplateau[i]]-nlref)<=dnl) and (np.abs(Ip_av[indplateau[i]]-Ipref)<=dIp) and (np.abs(tauth_av[indplateau[i]]-tauref)<=dtau)):
                    ind = indplateau[i]
                else:
                    # print(i_plateau)
                    # print(i)
                    # print(t_av[ind0])
                    # print(t_av[ind-1])
                    # print(nlref)
                    # print(np.abs(nl_av[indplateau[i]]-nlref))
                    # print(Ipref)
                    # print(np.abs(Ip_av[indplateau[i]]-Ipref))
                    # print(tauref)
                    # print(np.abs(tauth_av[indplateau[i]]-tauref))
                    if (ind-ind0-1>di_min):

                        t0.append(t_av[ind0])
                        t1.append(t_av[ind-1])
                    ind0 = indplateau[i-1]
                    ind = ind0
                    # ind = ind0
                    nlref = nl_av[ind0]
                    Ipref = Ip_av[ind0]
                    tauref = tauth_av[ind0]
                    # i = i-1
                    
                

    # breakpoint()
    # print('test')
 
    
    return t0, t1
    
def plateau_bounds(database):

    win_height = get_monitors()[0].height
    win_width = get_monitors()[0].width
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
            t0, t1 = get_plateau(t_av, nl_av, Ip_av, Ptot_av, tauth_av)
            fig = fig_maker_plateau_bound(database, t0, t1)
            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            plt.close()
            
            # t0 = 2500
            # t1 = 3000
    
    window.close()
    print(str(len(t0)) + ' plateaus have been found')
    return t0, t1
            
            
