#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:36:04 2024

@author: peretm
"""


import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import interpolate
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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

def smooth_TS_data(ts_data, i_plateau):
    psi, ne_av, ne_std = smooth_profiles(ts_data[str(i_plateau)]['psin_TS'], ts_data[str(i_plateau)]['density']*1e-19)

    psi, Te_av, Te_std = smooth_profiles(ts_data[str(i_plateau)]['psin_TS'], ts_data[str(i_plateau)]['temp'])
    
    ts_av = {}
    ts_av['psi'] = psi
    ts_av['ne'] = ne_av
    ts_av['ne_err'] = ne_std
    ts_av['Te'] = Te_av
    ts_av['Te_err'] = Te_std
    return ts_av

def func(x, a, l, c):
     return a * np.exp(- x / l) + c

def fit_SOL_profiles(psi, R, A, A_err, A_bound, i_plot):   
    ind_sep = np.argmin(np.abs(psi -1.0))
    A_err[A_err==0] = 1e-3*np.mean(A)
                
    popt, pcov = curve_fit(func, R-R[ind_sep], A, bounds=(0, [A_bound, 1.0, A_bound/4]), sigma = A_err)
    perr = np.sqrt(np.diag(pcov))
    A0 = (popt[0]+popt[2])
    lA = popt[1]
    A0_err = (perr[0]+perr[2])
    lA_err = perr[1]    
    A_fit = func(R-R[ind_sep],popt[0], popt[1], popt[2])
    R2 = 100.0 * r2_score(A, A_fit)
    if i_plot==1:
        plt.figure()
        plt.plot(R-R[ind_sep], A,'bo')
        plt.plot(R-R[ind_sep],A_fit,'r')

    # print(pcov)
    return A0, A0_err, lA, lA_err, A_fit, R2

def func_ped(x, a0, x0, alpha, dx, c):
     return a0 * (1-alpha*(x-x0))/(1+np.exp(2*(x-x0)/dx)) + c
 
def fit_ped_profiles(psi, R, A, A_err, A_bound, i_plot):
    ind_sep = np.argmin(np.abs(psi -1.0))
    A_err[A_err==0] = 1e-3*np.mean(A)
    ind0 = 0
    popt, pcov = curve_fit(func_ped, R-R[ind0], A, bounds=(0, [A_bound, 1.0, 1.0, 1.0, A_bound]), sigma = A_err)
    perr = np.sqrt(np.diag(pcov))
    A0 = popt[0]*(1-popt[2]*(R[ind_sep]-R[ind0]-popt[1]))/(1+np.exp(2*(R[ind_sep]-R[ind0]-popt[1])/popt[3])) + popt[4]
    A0_err = (perr[0]/2.0+perr[4])
    # lA = popt[3]/(1+popt[2]*popt[3])
    lA = ((1-popt[2]*(R[ind_sep]-R[ind0]-popt[1]))+popt[4]/popt[0]*(1+np.exp(2*(R[ind_sep]-R[ind0]-popt[1])/popt[3])))/(popt[2]+2/popt[3]*(1-popt[2]*(R[ind_sep]-R[ind0]-popt[1]))*(np.exp(2*(R[ind_sep]-R[ind0]-popt[1])/popt[3]))/(1+np.exp(2*(R[ind_sep]-R[ind0]-popt[1])/popt[3])))
    lA_err = (perr[3]+popt[3]*perr[2])*lA
    A_fit = func_ped(R-R[ind0],popt[0], popt[1], popt[2], popt[3], popt[4])
    R2 = 100.0 * r2_score(A,A_fit)

    
    if i_plot==1:
        plt.figure()
        plt.plot(R-R[ind_sep], A,'bo')
        plt.plot(R-R[ind_sep], A_fit, 'r')
    
    return A0, A0_err, lA, lA_err, A_fit, R2

def func_lin(x, a, b):
     return (1.0 - x / a) * b
 
def fit_lin_profiles(psi, R, A, A_err, A_bound, i_plot):
    A_err[A_err==0] = 1e-3 * np.min(A)
    ind_sep = np.argmin(np.abs(psi -1.0))
    
    popt, pcov = curve_fit(func_lin, R-R[ind_sep], A, bounds=(0, [1.0, A_bound]), sigma = A_err)
    perr = np.sqrt(np.diag(pcov))
    A0 = (popt[1])
    A0_err = (perr[1])
    lA = popt[0]
    lA_err = perr[0]   
    A_fit = func_lin(R-R[ind_sep], popt[0], popt[1])
    R2 = 100.0 * r2_score(A, A_fit)
    if i_plot ==1:
        plt.figure()
        plt.plot(R-R[ind_sep],A,'bo')
        plt.plot(R-R[ind_sep],A_fit,'r')
    
    
    return A0, A0_err, lA, lA_err, A_fit, R2

def func_lin_log(x, a, b):
     return  b - x / a
 
def fit_lin_log_profiles(psi, R, A, A_err, A_bound, i_plot):
    ind_sep = np.argmin(np.abs(psi-1.0))
    A_err[A_err==0] = 1e-3*np.mean(A)
    # breakpoint()
    popt, pcov = curve_fit(func_lin_log, R-R[ind_sep], np.log(A), bounds=(0, [0.5, np.log(A_bound)]), sigma = np.log(A_err))
    perr = np.sqrt(np.diag(pcov))
    # print(perr)
    # print(popt)
    A0_err = np.exp(popt[1]) * perr[1]
    A0 = np.exp(popt[1])
    lA = popt[0]
    lA_err = perr[0]
    R2 = 100.0 * r2_score(np.log(A), func_lin_log(R-R[ind_sep],popt[0], popt[1]))
    A_fit = func_lin_log(R-R[ind_sep],popt[0], popt[1])
    # print(dln/ln)
    # print(dne0/ne0)
    
    if i_plot ==1:
        plt.figure()
        plt.plot(R-R[ind_sep],np.log(A),'bo')
        plt.plot(R-R[ind_sep],A_fit,'r')


    # breakpoint()
    # print(popt)
    # print(pcov)
    return A0, A0_err, lA, lA_err, np.exp(A_fit), R2


def study_TS_data(ts_data, eq, i_plateau):
    data_out = {}
    data_out[str(i_plateau)] = {}
    prof = 'both'
    dR_shift = 0
    fit_names = ['mtanh', 'exp', 'linear', 'linear log']
    psi0 = [0.85, 0.98, 0.99, 0.99]
    psi1 = [1.05, 1.02, 1.02, 1.02]
    
    ts_av = smooth_TS_data(ts_data, i_plateau)
    for i in range(len(fit_names)):
        fit =fit_names[i]
        psimin = psi0[i]
        psimax = psi1[i]
        data_fitted, ts_av2, dpsi = fit_TS_data(ts_av, eq, dR_shift*1e-3, prof, fit, psimin, psimax)
        print('Fitting Done!')
        
        data_out[str(i_plateau)]['ts_av']=ts_av2
        data_out[str(i_plateau)][fit] = {}
        data_out[str(i_plateau)][fit] = data_fitted
        data_out[str(i_plateau)][fit]['psimin'] = psimin
        data_out[str(i_plateau)][fit]['psimax'] = psimax
        data_out[str(i_plateau)][fit]['dR_shift'] = dR_shift*1e-3
        data_out[str(i_plateau)][fit]['dpsi'] = dpsi
    return data_out

def fit_TS_data(ts_av, eq, dR_shift, prof, fit, psimin, psimax):
    ts_av0 = {}
    for i in ts_av.keys():
        ts_av0[i] = []
        ts_av0[i] = ts_av[i]
    i_plot = 0
    fitted_data = {}
    R = np.zeros(len(ts_av0['psi']))
    for i in range(len(eq['0']['atime'])):
        ind = np.argmin(np.abs(eq['0']['z']))
        ind_R = np.where(eq['0']['r']>=eq['0']['rmaxis'][i])[0]
        R = R + 1 / len(eq['0']['atime']) * np.interp(ts_av0['psi'], eq['0']['psin'][i,ind_R,ind], eq['0']['r'][ind_R])
    
    # dR_shift = 3e-3
    dpsi_shift = np.mean(np.diff(ts_av0['psi'])/np.diff(R))*dR_shift
    
    psi = ts_av0['psi'] + dpsi_shift
    ts_av0['psi'] = ts_av0['psi'] + dpsi_shift
    ind_fit = np.where(psi>=psimin)[0]
    ind_fit = ind_fit[np.where(psi[ind_fit]<= psimax)[0]]
    ind_fit = ind_fit[np.where(np.isnan(ts_av0['ne'][ind_fit])==0)[0]]
    ind_fit = ind_fit[np.where(np.isnan(ts_av0['Te'][ind_fit])==0)[0]]
    ind_fit = ind_fit[np.where(np.isnan(ts_av0['ne_err'][ind_fit])==0)[0]]
    ind_fit = ind_fit[np.where(np.isnan(ts_av0['Te_err'][ind_fit])==0)[0]]    
    
    
    if prof == 'electron density':
        profiles = ['ne']
    if prof == 'electron temperature':
        profiles = ['Te']
    if prof == 'both':
        profiles = ['ne', 'Te']
    fitted_data['ne'] = []
    fitted_data['Te'] = []
    
    for i in profiles:
        fitted_data[i + '_U'] = []
        fitted_data[i + '_U_err'] = []
        fitted_data['l' + i + '_U'] = []    
        fitted_data['l' + i + '_U_err'] = []    
        fitted_data['psi'] = psi[ind_fit]
        fitted_data[i] = np.zeros(len(ind_fit))
        fitted_data['R2' + i] = [] 
        try:
            if fit == 'mtanh':
                 if i== 'Te':
                     ts_av0['Te'] = ts_av0['Te'] * 1e-3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e-3
                 A_bound = 100.0
                 fitted_data[i + '_U'], fitted_data[i + '_U_err'],fitted_data['l' + i + '_U'], fitted_data['l' + i + '_U_err'], fitted_data[i], fitted_data['R2' + i] = fit_ped_profiles(psi[ind_fit], R[ind_fit], ts_av0[i][ind_fit], ts_av0[i+ '_err'][ind_fit], A_bound, i_plot)
                 if i== 'Te':
                     fitted_data[i + '_U'] = fitted_data[i + '_U']*1e3
                     fitted_data[i + '_U_err'] = fitted_data[i + '_U_err']*1e3
                     fitted_data[i] = fitted_data[i]*1e3
                     ts_av0['Te'] = ts_av0['Te'] * 1e3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e3
                     
            if fit == 'exp':
                 if i== 'Te':
                     ts_av0['Te'] = ts_av0['Te'] * 1e-3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e-3
                 A_bound = 10.0
                 fitted_data[i + '_U'], fitted_data[i + '_U_err'],fitted_data['l' + i + '_U'], fitted_data['l' + i + '_U_err'], fitted_data[i], fitted_data['R2' + i]  = fit_SOL_profiles(psi[ind_fit], R[ind_fit], ts_av0[i][ind_fit], ts_av0[i+ '_err'][ind_fit], A_bound, i_plot)
                 if i== 'Te':
                     fitted_data[i + '_U'] = fitted_data[i + '_U']*1e3
                     fitted_data[i + '_U_err'] = fitted_data[i + '_U_err']*1e3
                     fitted_data[i] = fitted_data[i]*1e3 
                     ts_av0['Te'] = ts_av0['Te'] * 1e3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e3
                    
            if fit == 'linear':
                 if i== 'Te':
                     ts_av0['Te'] = ts_av0['Te'] * 1e-3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e-3
                 A_bound = 10.0
                 fitted_data[i + '_U'], fitted_data[i + '_U_err'],fitted_data['l' + i + '_U'], fitted_data['l' + i + '_U_err'], fitted_data[i], fitted_data['R2' + i]  = fit_lin_profiles(psi[ind_fit], R[ind_fit], ts_av0[i][ind_fit], ts_av0[i+ '_err'][ind_fit], A_bound, i_plot)
                 if i== 'Te':
                     fitted_data[i + '_U'] = fitted_data[i + '_U']*1e3
                     fitted_data[i + '_U_err'] = fitted_data[i + '_U_err']*1e3
                     fitted_data[i] = fitted_data[i]*1e3 
                     ts_av0['Te'] = ts_av0['Te'] * 1e3
                     ts_av0['Te_err'] = ts_av0['Te_err'] * 1e3
                     
            if fit == 'linear log':
                 if i=='ne':
                     ts_av0['ne'] = ts_av0['ne']*1e19
                     ts_av0['ne_err'] = ts_av0['ne_err']*1e19                 
                     A_bound = 1e20
                 elif i=='Te':
                     A_bound = 1e3
                 fitted_data[i + '_U'], fitted_data[i + '_U_err'],fitted_data['l' + i + '_U'], fitted_data['l' + i + '_U_err'], fitted_data[i], fitted_data['R2' + i]  = fit_lin_log_profiles(psi[ind_fit], R[ind_fit], ts_av0[i][ind_fit], ts_av0[i+ '_err'][ind_fit], A_bound, i_plot)
                 if i== 'ne':
                     fitted_data[i + '_U'] = fitted_data[i + '_U']*1e-19
                     fitted_data[i + '_U_err'] = fitted_data[i + '_U_err']*1e-19
                     fitted_data[i] = fitted_data[i]*1e-19
                     ts_av0['ne'] = ts_av0['ne']*1e-19
                     ts_av0['ne_err'] = ts_av0['ne_err']*1e-19  

        except Exception as e:
            print(e)
            print('Fit did not work...')

    

    
    return fitted_data, ts_av0, dpsi_shift