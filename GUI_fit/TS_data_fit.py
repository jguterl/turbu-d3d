#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:45:02 2023

@author: peretm
"""

import matplotlib.pyplot as plt
from omfit_classes.utils_fusion import calc_h98_d3d
from omfit_classes.omfit_eqdsk import read_basic_eq_from_mds
from omfit_classes.omfit_mds import OMFITmdsValue
import numpy as np
import math
from omfit_classes.omfit_thomson import OMFITthomson
import matplotlib
from fit_functions import *
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
from screeninfo import get_monitors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


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
def smooth_profiles(psin, A):
    n_pts = 300
    psi = np.zeros(n_pts)
    A_av = np.zeros(n_pts)
    A_std = np.zeros(n_pts)
    A = A.flatten()
    psin = psin.flatten()
    # breakpoint()
    # print('test')
    for i in range(n_pts):
        ind = np.where(psin>=i*(1.1-0.8)/n_pts+0.8)[0]
        ind = ind[np.where(psin[ind]<(i+1)*(1.1-0.8)/n_pts+0.8)[0]]
        psi[i] = 0.8 + (i+0.5) * (1.1-0.8)/n_pts
        A_av[i] = np.mean(A[ind])
        A_std[i] = np.std(A[ind])
    return psi, A_av, A_std

def fig_maker_TS_data(ts_data, i_plateau):
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(211)
    for i in range(len(ts_data[str(i_plateau)]['time'])):      
        ax.errorbar(ts_data[str(i_plateau)]['psin_TS'][:,i], ts_data[str(i_plateau)]['density'][:,i]*1e-19, ts_data[str(i_plateau)]['density_e'][:,i]*1e-19, None, 'bo')
    # ax.legend()
    psi, ne_av, ne_std = smooth_profiles(ts_data[str(i_plateau)]['psin_TS'], ts_data[str(i_plateau)]['density']*1e-19)
    ax.errorbar(psi, ne_av, ne_std, None, 'ko')
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel(r'$\Psi_{N}$')
    ax.set_ylabel(r'$n_{e} [\times 10^{19}m^{-3}]$')
    ax.set_xlim((0.8,1.1))
    
    ax1 = fig.add_subplot(212)
    for i in range(len(ts_data[str(i_plateau)]['time'])):      
        ax1.errorbar(ts_data[str(i_plateau)]['psin_TS'][:,i], ts_data[str(i_plateau)]['temp'][:,i], ts_data[str(i_plateau)]['temp_e'][:,i], None, 'ro')
    # ax.legend()
    psi, Te_av, Te_std = smooth_profiles(ts_data[str(i_plateau)]['psin_TS'], ts_data[str(i_plateau)]['temp'])
    ax1.errorbar(psi, Te_av, Te_std, None, 'ko')
    ax1.grid(color='k', linestyle='-')       
    ax1.set_xlabel(r'$\Psi_{N}$')
    ax1.set_ylabel(r'$T_{e} [eV]$')
    ax1.set_xlim((0.8,1.1))
    
    ts_av = {}
    ts_av['psi'] = psi
    ts_av['ne'] = ne_av
    ts_av['ne_err'] = ne_std
    ts_av['Te'] = Te_av
    ts_av['Te_err'] = Te_std
    return fig, ts_av

def fig_maker_TS_data_fit(ts_data, ts_av, data_fitted, dpsi, i_plateau):
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    ax = fig.add_subplot(211)
    for i in range(len(ts_data[str(i_plateau)]['time'])):      
        ax.errorbar(ts_data[str(i_plateau)]['psin_TS'][:,i]+ dpsi, ts_data[str(i_plateau)]['density'][:,i]*1e-19, ts_data[str(i_plateau)]['density_e'][:,i]*1e-19, None, 'bo')
    # ax.legend()
        
    ax.errorbar(ts_av['psi'], ts_av['ne'], ts_av['ne_err'], None, 'ko')
    if data_fitted['ne']!=[]:
        ax.errorbar(data_fitted['psi'], data_fitted['ne'], None, None, 'g', linewidth=3)
    ax.grid(color='k', linestyle='-')       
    ax.set_xlabel(r'$\Psi_{N}$')
    ax.set_ylabel(r'$n_{e} [\times 10^{19}m^{-3}]$')
    ax.set_xlim((0.8,1.1))
    
    ax1 = fig.add_subplot(212)
    for i in range(len(ts_data[str(i_plateau)]['time'])):      
        ax1.errorbar(ts_data[str(i_plateau)]['psin_TS'][:,i]+dpsi, ts_data[str(i_plateau)]['temp'][:,i], ts_data[str(i_plateau)]['temp_e'][:,i], None, 'ro')
    # ax.legend()
    ax1.errorbar(ts_av['psi'], ts_av['Te'], ts_av['Te_err'], None, 'ko')
    if data_fitted['Te']!=[]:
        ax1.errorbar(data_fitted['psi'], data_fitted['Te'], None, None, 'g', linewidth=3)
    ax1.grid(color='k', linestyle='-')       
    ax1.set_xlabel(r'$\Psi_{N}$')
    ax1.set_ylabel(r'$T_{e} [eV]$')
    ax1.set_xlim((0.8,1.1))
    ax1.set_ylim((0, 1.05 * np.nanmax(ts_av['Te'])))

    

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

def study_TS_data(ts_data, eq):

    win_height = get_monitors()[0].height
    win_width = get_monitors()[0].width

    matplotlib.use('TkAgg')
    w, h = size = (np.floor(0.4*win_width), np.floor(0.4*win_height))     # figure size
    names = []
    for i in ts_data.keys():
        names.append(str(int(i)+1))
    names_fit = ['mtanh', 'exp', 'linear', 'linear log']
    fig = init_fig_maker()
    column1 = [[sg.Canvas(size=size, key='-CANVAS-')],
              [sg.Canvas(key='controls_cv')],
             [sg.Cancel(),sg.Text('Choose plateau number'),sg.Combo(names, font=('Arial Bold', 10),  expand_x=True, enable_events=True,  readonly=False, key='-COMBO-'), sg.Button('Plot profiles')]]
    layout = [ [sg.Column(column1),
          sg.VSeperator(),
     sg.Column([[sg.Button('Fit profiles')],
                [sg.Text('Choose profile to fit'),sg.Combo(['electron density', 'electron temperature', 'both'], font=('Arial Bold', 10),  expand_x=True, enable_events=True,  readonly=False, key='-COMBO2-')],
                [sg.Text('Choose fitting method'),sg.Combo(names_fit, font=('Arial Bold', 10),  expand_x=True, enable_events=True,  readonly=False, key='-COMBO3-')],
                [sg.Text('Enter min psi value'), sg.InputText()],
                [sg.Text('Enter max psi value'), sg.InputText()],
                [sg.Text('Apply a shift [in mm]'), sg.InputText()],
                [sg.Button('Equilibrium evolution')],])]]
    window = sg.Window('Plot and fit TS data', layout, finalize=True, element_justification='center', font='Helvetica 12')
    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    # delete_fig_agg(fig_agg)
    # fig = fig_maker(database)
    # fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
    # plt.close()


    # fig_maker(database)
    ts_av = []
    data_out = {}
    for i_plateau in range(len(names)):
        data_out[str(i_plateau)] = {}

    i_plateau = []

    while True:             
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break

        elif event in ('Plot profiles'):  
            if values['-COMBO-']=='':
                print('Choose a plateau.')
            else:
                i_plateau = int(values['-COMBO-'])-1
                # delete_fig_agg(fig_agg)
                ts_av = []
                fig, ts_av = fig_maker_TS_data(ts_data, i_plateau)
                fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
                plt.close()
                # for i in range(len(ts_data.keys())):
                #     data_out[str(i)] = {}
        elif event in ('Fit profiles'):
            if (ts_av==[]):
                print('No data to fit. Please plot profiles first')
            else:
                if values['-COMBO2-']=='':
                    print('Select profile(s) to fit.')
                    psimin = []
                    psimax = []
                else:    
                    print('Fit ' + values['-COMBO2-'] +' profile(s)!')
                    prof = values['-COMBO2-']
                    
                    if values[1]=='':
                        print('Add a minimal value of psi for the fit range.') 
                        psimin = []
                    else:
                        psimin = float(values[1])
                        
                    if values[2]=='':
                        print('Add a maximal value of psi for the fit range.')
                        psimax = [] 
                    else:
                        psimax = float(values[2])
                    if values[3]=='':
                        dR_shift=0
                    else:
                        dR_shift = float(values[3])
                            
                    if ((psimin==[]) or (psimax==[])):
                        []
                    else:
                        if values['-COMBO3-']=='':
                            print('Choose a fitting method.')
                        else:
                            fit = values['-COMBO3-']
                            print('Fitting launched...')
                            data_fitted, ts_av2, dpsi = fit_TS_data(ts_av, eq, dR_shift*1e-3, prof, fit, psimin, psimax)
                            print('Fitting Done!')
                            fig = fig_maker_TS_data_fit(ts_data, ts_av2, data_fitted, dpsi, i_plateau)
                            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
                            plt.close()
                            data_out[str(i_plateau)]['ts_av']=ts_av2
                            data_out[str(i_plateau)][fit] = {}
                            data_out[str(i_plateau)][fit] = data_fitted
                            data_out[str(i_plateau)][fit]['psimin'] = psimin
                            data_out[str(i_plateau)][fit]['psimax'] = psimax
                            data_out[str(i_plateau)][fit]['dR_shift'] = dR_shift*1e-3
                            data_out[str(i_plateau)][fit]['dpsi'] = dpsi
                            
        elif event in ('Equilibrium evolution'):
            if values['-COMBO-']=='':
                print('Choose a plateau.')
            else:
                i_plateau = int(values['-COMBO-'])-1
                plt.figure()
                for i in range(len(eq[str(i_plateau)])):
                    plt.contour(eq[str(i_plateau)]['r'], eq[str(i_plateau)]['z'], eq[str(i_plateau)]['psin'][i,:,:], levels=[0.1, 0.5, 0.8, 1.0, 1.05], colors=['#808080','#808080','#808080', '#C0C0C0','#808080'])
                    plt.plot(eq[str(i_plateau)]['lim'][:,0], eq[str(i_plateau)]['lim'][:,1], 'k', linewidth = 3)
                    plt.xlabel('R [m]')
                    plt.ylabel('Z [m]')
                    plt.axis('equal')
                    plt.pause(0.4)
                    plt.clf()
                plt.close()

            # t_av, nl_av, Ip_av, Ptot_av = win_average(database)
            # t0, t1 = get_plateau(t_av, nl_av, Ip_av, Ptot_av)
            # fig = fig_maker_plateau_bound(database, t0, t1)
            # fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            # plt.close()
            
            # t0 = 2500
            # t1 = 3000
    
    window.close()
    return data_out
    # print(str(len(t0)) + ' plateaus have been found')
    # return t0, t1
            
            
