#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:33:22 2024

@author: peretm
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt

plt.close('all')
exp = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_exp_2.npy',allow_pickle=True).tolist()
model = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_model_2.npy',allow_pickle=True).tolist()
params = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_params_2.npy',allow_pickle=True).tolist()


# exp = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_exp_correc_no_shift_L_mode_2.npy',allow_pickle=True).tolist()
# model = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_model_correc_no_shift_L_mode_2.npy',allow_pickle=True).tolist()
# params = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_params_correc_no_shift_L_mode_2.npy',allow_pickle=True).tolist()



exp_H = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_exp_correc_no_shift_2.npy',allow_pickle=True).tolist()
model_H = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_model_correc_no_shift_2.npy',allow_pickle=True).tolist()
params_H = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_params_correc_no_shift_2.npy',allow_pickle=True).tolist()

err_lim = 0.35
mu0 = 4*math.pi*1e-7

ind_H = np.where(np.array(params_H['H98'])>1.05)[0]
# ind_H = ind_H[np.where(np.array(params_H['H98'])<1.4)[0]]
ind_H = ind_H[np.where(np.abs(np.array(model_H['ks']))[ind_H]<1.0)[0]]
ind_H = ind_H[np.where(np.array(exp_H['lT_err'])[ind_H]/np.array(exp_H['lT'])[ind_H]<=err_lim)[0]]
ind_H = ind_H[np.where(np.array(exp_H['ln_err'])[ind_H]/np.array(exp_H['ln'])[ind_H]<=err_lim)[0]]
ind_H = ind_H[np.where(np.array(exp_H['ne_U_err'])[ind_H]/np.array(exp_H['ne_U'])[ind_H]<=err_lim)[0]]
ind_H = ind_H[np.where(np.array(exp_H['Te_U_err'])[ind_H]/np.array(exp_H['Te_U'])[ind_H]<=err_lim)[0]]
# ind_H = ind_H[np.where(np.sqrt(1.602e-19/9.11e-31*1e3)*(np.array(exp_H['Te_U'])[ind_H]*1e-3)**(2)/(np.array(exp_H['ne_U'])[ind_H]*0.1)/9.11/1.36e5<0.8)[0]]

lp_exp = ((1.0/(1.0/np.array(exp_H['ln'])+1.0/np.array(exp_H['lT']))/np.array(model_H['rhos'])))[ind_H]
ln_exp = (np.array(exp_H['ln'])/np.array(model_H['rhos']))[ind_H]
Bpol_H = mu0*np.abs(np.array(params_H['Ip'])[ind_H])/np.array(params_H['a'])[ind_H]/np.sqrt(0.5*(1.0+np.array(params_H['kappa'])[ind_H]**2))/2.0/math.pi
lp_exp_err = 0.0 * lp_exp 
ln_exp_err = 0.0 * lp_exp 
ln_sol = 0.0 * lp_exp
lp_sol = 0.0 * lp_exp
lp_sol2 = 0.0 * lp_exp
lp_sol3 = 0.0 * lp_exp
lp_sol4 = 0.0 * lp_exp
lp_sol5 = 0.0 * lp_exp

n_sol = 0.0 * lp_exp
gamma_sol = 0.0 * lp_exp
kx_H = 0.0 * lp_exp
ks_H = 0.0 * lp_exp
Te = 0.0 * lp_exp
# i = 154
j = -1

fudge = 2.0
gamma0 = 5.3
for i in ind_H:
    j = j+1
    l0 = 3.9*model_H['g'][i]**(3/11)*(4*model_H['sig_PHI'][i])**(-2/11)*(4*model_H['sig_N'][i])**(-4/11)
    # gamma = (np.array(exp_H['ln'])[i]/lp_exp[j]/np.array(model_H['rhos'])[i])**2
    gamma = gamma0
    alpha_lT = np.sqrt(gamma)/(np.sqrt(gamma)-1)
    lp = np.linspace(1.0, 200, 1000)
    # ks = -1.2
    ks0 = model_H['ks'][i]
    lp_exp_err[j] = (np.array(exp_H['ln_err'])[i]/gamma + (np.sqrt(gamma)-1)**2/gamma*np.array(exp_H['lT_err'])[i])/np.array(model_H['rhos'])[i]+lp_exp[j]*0.5 * np.array(exp_H['Te_U_err'])[i]/np.array(exp_H['Te_U'])[i]
    ln_exp_err[j] = lp_exp_err[j]*np.sqrt(gamma)
    
    # f = lp*(1+(ks0-0.43*fudge*3.0/model_H['g'][i]**(0.5)*lp**(-3/2)/alpha_lT**2)**2)**(9/11)-l0/gamma**(4/11)
    ll = 1.0/(1.0/np.array(exp_H['ln'])[i]+1.0/np.array(exp_H['lT'])[i])/np.array(model_H['rhos'])[i]
    f = lp*(1+(ks0-0.43*fudge*3.0/model_H['g'][i]**(0.5)*ll**(-3/2)/alpha_lT**2)**2)**(9/11)-l0/gamma**(4/11)
    # plt.figure()
    # plt.plot(lp, f, 'o')
    # plt.plot(lp, f2, 'o')
    # breakpoint()
    gamma_sol[j] = gamma

    ind_sol = np.where(np.sign(f[0:-1]*f[1:])<0.0)[0]
    
    kx_H[j] = -0.43*fudge*3.0/model_H['g'][i]**(0.5)*ll**(-3/2)/alpha_lT**2
    ks_H[j] = model_H['ks'][i]
    Te[j] = exp_H['Te_U'][i]
    try:
        n_sol[j] = len(ind_sol)
        lp_sol[j] = lp[ind_sol[-1]]     
        
        ln_sol[j] = np.sqrt(gamma) * lp_sol[j]

    except:
        # plt.figure()
        # plt.plot(lp, f, '-o')
        # breakpoint()
        lp_sol[j] = np.nan
        ln_sol[j] = np.nan
        []
    


# lq = lp_exp / (1.0+0.5*(np.sqrt(gamma_sol)-1.0)/np.sqrt(gamma_sol)) * np.array(model_H['rhos'])[ind_H]

lq = 2/7 * np.array(exp_H['lT'])[ind_H]



ind_L = np.where(np.array(params['H98'])<0.6)[0]
ind_L = ind_L[np.where(np.array(params['H98'])[ind_L]>0.25)[0]]
ind_L = ind_L[np.where(np.array(exp['lT_err'])[ind_L]/np.array(exp['lT'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['ln_err'])[ind_L]/np.array(exp['ln'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['ne_U_err'])[ind_L]/np.array(exp['ne_U'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['Te_U_err'])[ind_L]/np.array(exp['Te_U'])[ind_L]<=err_lim)[0]]
# ind_L = ind_L[np.where(np.sqrt(1.602e-19/9.11e-31*1e3)*(np.array(exp['Te_U'])[ind_L]*1e-3)**(2)/(np.array(exp['ne_U'])[ind_L]*0.1)/9.11/1.36e5<0.8)[0]]


Bpol_L = mu0*np.abs(np.array(params['Ip'])[ind_L])/np.array(params['a'])[ind_L]/np.sqrt(0.5*(1.0+np.array(params['kappa'])[ind_L]**2))/2.0/math.pi

lp_L_exp = ((1.0/(1.0/np.array(exp['ln'])+1.0/np.array(exp['lT']))/np.array(model['rhos'])))[ind_L]
ln_L_exp = (np.array(exp['ln'])/np.array(model['rhos']))[ind_L]


lp_L_exp_err = 0.0 * lp_L_exp 
ln_L_exp_err = 0.0 * lp_L_exp 
lp_L_sol = 0.0 * lp_L_exp +0.0
ln_L_sol = 0.0 * lp_L_exp+0.0
n_sol_L = 0.0 * lp_L_exp+0.0
gamma_L_sol = 0.0 * lp_L_exp+0.0
kx = 0.0 * lp_L_exp+0.0
ks = 0.0 * lp_L_exp+0.0
Te_L = 0.0 * lp_L_exp+0.0
# i = 154
j = -1
for i in ind_L:
    j = j+1
    l0 = 3.9*model['g'][i]**(3/11)*(4*model['sig_PHI'][i])**(-2/11)*(4*model['sig_N'][i])**(-4/11)
    # gamma = (np.array(exp['ln'])[i]/lp_L_exp[j]/np.array(model['rhos'])[i])**2
    gamma = gamma0
    
    alpha_lT = np.sqrt(gamma)/(np.sqrt(gamma)-1)
    lp = np.linspace(1.0, 200, 1000)
    ks0 = model['ks'][i]
    # f = lp*(1+(ks0 - fudge*0.43*3.0/model['g'][i]**(0.5)*lp**(-3/2)/alpha_lT**2)**2)**(9/11)-l0/gamma**(4/11)
    ll = 1.0/(1.0/np.array(exp['ln'])[i]+1.0/np.array(exp['lT'])[i])/np.array(model['rhos'])[i]
    f = lp*(1+(model['ks'][i]- fudge*0.43*3.0/model['g'][i]**(0.5)*ll**(-3/2)/alpha_lT**2)**2)**(9/11)-l0/gamma**(4/11)
    # plt.figure()
    # plt.plot(lp, f, 'o')
    # breakpoint()
    lp_L_exp_err[j] = (np.array(exp['ln_err'])[i]/gamma + (np.sqrt(gamma)-1)**2/gamma*np.array(exp['lT_err'])[i])/np.array(model['rhos'])[i]+lp_L_exp[j]*0.5 * np.array(exp['Te_U_err'])[i]/np.array(exp['Te_U'])[i]
    ln_L_exp_err[j] = lp_L_exp_err[j]*np.sqrt(gamma)
    ind_sol = np.where(np.sign(f[0:-1]*f[1:])<0.0)[0]
    gamma_L_sol[j] = gamma
    
    try:
        n_sol_L[j] = len(ind_sol)
        lp_L_sol[j] = lp[ind_sol[-1]]
        ln_L_sol[j] = lp_L_sol[j] * np.sqrt(gamma)
        kx[j] = -0.43*fudge*3.0/model['g'][i]**(0.5)*ll**(-3/2)/alpha_lT**2
        ks[j] = model['ks'][i]
        Te_L[j] = exp['Te_U'][i]
    except:
        lp_L_sol[j] = np.nan
        ln_L_sol[j] = np.nan
        kx[j] = np.nan
        ks[j] = np.nan
        Te_L[j] = np.nan
        []



lT_sol = 1/(1/lp_sol-1/ln_sol)
lT_L_sol = 1/(1/lp_L_sol-1/ln_L_sol)

lT_exp = (np.array(exp_H['lT'])/np.array(model_H['rhos']))[ind_H]
lT_L_exp = (np.array(exp['lT'])/np.array(model['rhos']))[ind_L]

# lq_L = lp_L_exp / (1.0+0.5*(np.sqrt(gamma_L_sol)-1.0)/np.sqrt(gamma_L_sol))* np.array(model['rhos'])[ind_L]
lq_L = 2/7 * np.array(exp['lT'])[ind_L] # lp_L_exp / (1.0+0.5*(np.sqrt(gamma_L_sol)-1.0)/np.sqrt(gamma_L_sol))* np.array(model['rhos'])[ind_L]

lq_sol = 2/7 *lp_sol *np.sqrt(gamma0)/(np.sqrt(gamma0)-1) * np.array(model_H['rhos'])[ind_H]*1e3 
lq_scale = 0.63 * gamma0**(3/22)/(np.sqrt(gamma0)-1)*(2)**(4/11)*(np.array(params_H['q95'])[ind_H])**(6/11)*(np.abs(np.array(params_H['BT'])[ind_H]))**(-8/11) * (np.array(params_H['R0'])[ind_H])**(3/11)*Te**(4/11)

plt.figure()
plt.plot(lq_sol, 0.63*Bpol_H**(-1.19), 'o')
plt.plot(lq*1e3, 0.63*Bpol_H**(-1.19), 'o')
plt.plot([0,8], [0,8], '--k')

plt.figure()
plt.plot(lp_exp, lp_sol, 'o', label= r'$\beta$ = ' + str(fudge) + r' / gamma = ' + str(gamma))
plt.plot(lp_exp, lp_sol2, 'o', label= r'$\beta$ = ' + str(fudge*1.25) + r' / $\gamma$ = ' + str(gamma))
plt.plot(lp_exp, lp_sol3, 'o', label= r'$\beta$ = ' + str(fudge*0.75) + r' / $\gamma$ = ' + str(gamma))
plt.plot(lp_exp, lp_sol4, 'o', label= r'$\beta$ = ' + str(fudge) + r' / $\gamma$ = ' + str(gamma*1.15))
plt.plot(lp_exp, lp_sol5, 'o', label= r'$\beta$ = ' + str(fudge) + r' / $\gamma$ = ' + str(gamma*0.85))

plt.plot([0,50],[0,50],'--k', label=r'1:1 $\pm 40\%$')
plt.plot([0,50],[0,1.4*50],'--k')
plt.plot([0,50],[0,0.6*50],'--k')
plt.legend()
plt.xlabel(r'Experiment [$\rho_{s}$]')
plt.ylabel(r'Model [$\rho_{s}$]')
plt.title('H-mode')

if 1:
    plt.figure()
    plt.plot(lp_exp, lp_sol, 'o', label = r'$\lambda_{p}$')
    plt.plot(ln_exp, ln_sol, 'o', label = r'$\lambda_{n}$')
    plt.plot([0,100],[0,100], '--k', label = '1:1')
    plt.plot([0,100],[0,150], '--r', label = '1:1.5')
    # plt.plot([0,100],[0,200], '--b', label = '1:2')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title('H-mode')
    
    plt.figure()
    plt.plot(lp_L_exp, lp_L_sol, 'o', label = r'$\lambda_{p}$')
    plt.plot(ln_L_exp, ln_L_sol, 'o', label = r'$\lambda_{n}$')
    plt.plot([0,180],[0,180], '--k', label = '1:1')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title('L-mode')


    Bpol = np.linspace(0.1, 0.5, 100)
    plt.figure()
    plt.plot(Bpol_H, lq*1e3, 'o', label='H-mode')
    plt.plot(Bpol_L, lq_L*1e3, 'o', label='L-mode')
    plt.plot(Bpol, 0.63*Bpol**(-1.19), '--k', label='Eich scaling')
    plt.xlabel('Bpol [T]')
    plt.ylabel(r'$\lambda_{q}$')
    plt.legend()
    
    
    n_bin = 25
    indnan = np.where(np.isnan(lp_L_sol)==0)[0]
    plt.figure()
    ha, a, b, c  = plt.hist2d(lp_sol, lp_exp, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha2, a2, b2, c2 = plt.hist2d(lp_L_sol[indnan], lp_L_exp[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    plt.close()

    plt.figure()
    ha3, a3, b3, c3  = plt.hist2d(ln_sol, ln_exp, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha4, a4, b4, c4 = plt.hist2d(ln_L_sol[indnan], ln_L_exp[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    plt.close()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    # h = ax.hist2d(lp_exp, lp_sol, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    h = ax.imshow(ha, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b), np.max(b),np.min(a), np.max(a)])
    cbar = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\lambda_{p}$')
    ax.plot([0,35],[0,35], '--w', label = r'1:1 $\pm 40\%$')
    ax.plot([0,35],[0,1.4*35], '--w')
    ax.plot([0,35],[0,0.6*35], '--w')
    ax.set_xlim((0,35))
    ax.set_ylim((0,35))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax.set_ylabel(r'Model [$\rho_{s}$]')
    ax.set_title('H-mode')
    ax.legend()
    ax.set_facecolor("black")

    ax2 = fig.add_subplot(2, 2, 2)
    # h2 = ax2.hist2d(lp_L_exp[indnan], lp_L_sol[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    h2 = ax2.imshow(ha2, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b2), np.max(b2),np.min(a2), np.max(a2)])
    cbar2 = fig.colorbar(h2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\lambda_{p}$')
    
    ax2.plot([0,60],[0,60], '--w', label = r'1:1 $\pm 40\%$')
    ax2.plot([0,60],[0,1.4*60], '--w')
    ax2.plot([0,60],[0,0.6*60], '--w')
    ax2.set_xlim((0,60))
    ax2.set_ylim((0,60))
    ax2.set_facecolor("black")
    ax2.set_aspect('equal', 'box')
    ax2.set_title('L-mode')
    ax2.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax2.set_ylabel(r'Model [$\rho_{s}$]')
    ax2.legend()
    fig.tight_layout()
    
    # fig = plt.figure()
    ax3 = fig.add_subplot(2, 2, 3)
    # h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    h3 = ax3.imshow(ha3, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b3), np.max(b3),np.min(a3), np.max(a3)])    
    cbar3 = fig.colorbar(h3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r'$\lambda_{n}$')
    ax3.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax3.plot([0,80],[0,1.4*80], '--w')
    ax3.plot([0,80],[0,0.6*80], '--w')
    ax3.set_xlim((0,80))
    ax3.set_ylim((0,80))
    ax3.set_aspect('equal', 'box')
    ax3.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax3.set_ylabel(r'Model [$\rho_{s}$]')
    ax3.set_title('H-mode')
    ax3.legend()
    ax3.set_facecolor("black")

    ax4 = fig.add_subplot(2, 2, 4)
    # h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    h4 = ax4.imshow(ha4, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b4), np.max(b4),np.min(a4), np.max(a4)]) 
    cbar4 = fig.colorbar(h4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label(r'$\lambda_{n}$')
    ax4.plot([0,150],[0,150], '--w', label = r'1:1 $\pm 40\%$')
    ax4.plot([0,150],[0,1.4*150], '--w')
    ax4.plot([0,150],[0,0.6*150], '--w')
    ax4.set_xlim((0,150))
    ax4.set_ylim((0,150))
    ax4.set_facecolor("black")
    ax4.set_aspect('equal', 'box')
    ax4.set_title('L-mode')
    ax4.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax4.set_ylabel(r'Model [$\rho_{s}$]')
    ax4.legend()
    fig.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    h = ax.hist2d(lp_exp, lp_sol, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    # h = ax.imshow(ha, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b), np.max(b),np.min(a), np.max(a)])
    cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\lambda_{p}$')
    ax.plot([0,35],[0,35], '--w', label = r'1:1 $\pm 40\%$')
    ax.plot([0,35],[0,1.4*35], '--w')
    ax.plot([0,35],[0,0.6*35], '--w')
    ax.set_xlim((0,35))
    ax.set_ylim((0,35))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax.set_ylabel(r'Model [$\rho_{s}$]')
    ax.set_title('H-mode')
    ax.legend()
    ax.set_facecolor("black")

    ax2 = fig.add_subplot(2, 2, 2)
    h2 = ax2.hist2d(lp_L_exp[indnan], lp_L_sol[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    # h2 = ax2.imshow(ha2, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b2), np.max(b2),np.min(a2), np.max(a2)])
    cbar2 = fig.colorbar(h2[3], ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\lambda_{p}$')
    
    ax2.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax2.plot([0,80],[0,1.4*80], '--w')
    ax2.plot([0,80],[0,0.6*80], '--w')
    ax2.set_xlim((0,80))
    ax2.set_ylim((0,80))
    ax2.set_facecolor("black")
    ax2.set_aspect('equal', 'box')
    ax2.set_title('L-mode')
    ax2.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax2.set_ylabel(r'Model [$\rho_{s}$]')
    ax2.legend()
    fig.tight_layout()
    
    # fig = plt.figure()
    ax3 = fig.add_subplot(2, 2, 3)
    h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    # h3 = ax3.imshow(ha3, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b3), np.max(b3),np.min(a3), np.max(a3)])    
    cbar3 = fig.colorbar(h3[3], ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r'$\lambda_{n}$')
    ax3.plot([0,90],[0,90], '--w', label = r'1:1 $\pm 40\%$')
    ax3.plot([0,90],[0,1.4*90], '--w')
    ax3.plot([0,90],[0,0.6*90], '--w')
    ax3.set_xlim((0,90))
    ax3.set_ylim((0,90))
    ax3.set_aspect('equal', 'box')
    ax3.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax3.set_ylabel(r'Model [$\rho_{s}$]')
    ax3.set_title('H-mode')
    ax3.legend()
    ax3.set_facecolor("black")

    ax4 = fig.add_subplot(2, 2, 4)
    h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    # h4 = ax4.imshow(ha4, cmap ='hot', origin = "lower", interpolation = "gaussian", extent=[np.min(b4), np.max(b4),np.min(a4), np.max(a4)]) 
    cbar4 = fig.colorbar(h4[3], ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label(r'$\lambda_{n}$')
    ax4.plot([0,200],[0,200], '--w', label = r'1:1 $\pm 40\%$')
    ax4.plot([0,200],[0,1.4*200], '--w')
    ax4.plot([0,200],[0,0.6*200], '--w')
    ax4.set_xlim((0,200))
    ax4.set_ylim((0,200))
    ax4.set_facecolor("black")
    ax4.set_aspect('equal', 'box')
    ax4.set_title('L-mode')
    ax4.set_xlabel(r'Experiment [$\rho_{s}$]')
    ax4.set_ylabel(r'Model [$\rho_{s}$]')
    ax4.legend()
    fig.tight_layout()
    

    
    plt.figure()
    plt.errorbar(lp_exp, lp_sol, None, lp_exp_err,  'o', label = 'H-mode')
    plt.errorbar(lp_L_exp, lp_L_sol, None, lp_L_exp_err,  'o', label = 'L-mode')
    # plt.plot(lp_L_exp, lp_L_sol, 'o', label = 'L-mode')
    plt.plot([0,100],[0,100], '--k', label = r'1:1 $\pm 40\%$')
    plt.plot([0,100],[0,1.4*100], '--k')
    plt.plot([0,100],[0,100/1.4], '--k')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title(r'$\lambda_{p}$')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.figure()
    plt.errorbar(ln_exp, ln_sol, None, ln_exp_err,  'o', label = 'H-mode')
    plt.errorbar(ln_L_exp, ln_L_sol, None, ln_L_exp_err,  'o', label = 'L-mode')
    plt.plot([0,200],[0,200], '--k', label = r'1:1 $\pm 40\%$')
    plt.plot([0,200],[0,1.4*200], '--k')
    plt.plot([0,200],[0,200/1.4], '--k')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title(r'$\lambda_{n}$')
    plt.xscale('log')
    plt.yscale('log')


    # plt.figure()
    # plt.plot(np.sign(np.array(params['Z_OSP']))[ind_L]*np.array(params['BT'])[ind_L], ks, 'o')
    # plt.figure()
    # plt.plot(np.sign(np.array(params_H['Z_OSP']))[ind_H]*np.array(params_H['BT'])[ind_H], ks_H, 'o')  
    
    
    fig1 = plt.figure()
    plt.subplot(221)
    (n, bins, patches) = plt.hist((ln_exp/lp_exp)**2, bins=10, label='H-mode', histtype='barstacked')
    plt.hist((ln_L_exp/lp_L_exp)**2, bins = 10, alpha=0.5,  label='L-mode', histtype='barstacked')
    plt.plot([gamma0,gamma0], [0.0,50], '-r', label=r'$\gamma=$'+str(gamma0))
    plt.text(15, 45, 'a)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    plt.xlabel(r'$\gamma$', fontsize = 12)
    plt.legend(fontsize= 11)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    
    plt.subplot(222)
    (n, bins, patches) = plt.hist(np.array(model_H['g'])[ind_H]/np.array(model_H['rhos'])[ind_H]*np.array(params_H['R0'])[ind_H], bins=10, label='H-mode', histtype='barstacked')
    plt.hist(np.array(model['g'])[ind_L]/np.array(model['rhos'])[ind_L]*np.array(params['R0'])[ind_L], bins = 10, alpha=0.5,  label='L-mode', histtype='barstacked')
    plt.xlabel(r'$G_{0}$', fontsize=12)
    plt.legend(fontsize=11)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)  
    plt.text(0.9, 27, 'b)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    
    plt.subplot(223)
    (n, bins, patches) = plt.hist(ks_H, bins=10,  label='H-mode', histtype='barstacked')
    plt.hist(ks, bins = 10, alpha=0.5, label='L-mode', histtype='barstacked')
    plt.xlabel(r'$\alpha_{s}$', fontsize = 12)
    plt.legend(fontsize=11)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.text(-0.9, 22, 'c)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    
    plt.subplot(224)
    (n, bins, patches) = plt.hist(kx_H, bins=10,  label='H-mode', histtype='barstacked')
    plt.hist(kx, bins = 10, alpha=0.5, label='L-mode', histtype='barstacked')
    plt.xlabel(r'$\alpha_{E{\times}B}$', fontsize=12)
    plt.legend(fontsize=11)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.text(-3.2, 17, 'd)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    
    plt.tight_layout()
    fig1.savefig('gamma_G0_alphas_alphaExB_hist_old_database.png', format='png', dpi=300)

    # plt.subplot(223)
    # (n, bins, patches) = plt.hist(np.array(exp_H['ne_U'])[ind_H], bins=20, label='H-mode', histtype='barstacked')
    # plt.hist(np.array(exp['ne_U'])[ind_L], bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    # plt.xlabel(r'$n_{e}$ [$\times 10^{19}m^{3}$]')
    # plt.legend()
    # plt.subplot(224)
    # (n, bins, patches) = plt.hist(np.array(exp_H['Te_U'])[ind_H], bins=20, label='H-mode', histtype='barstacked')
    # plt.hist(np.array(exp['Te_U'])[ind_L], bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    # plt.xlabel(r'$T_{e}$ [eV]')
    # plt.legend()
    
    
    fig2 = plt.figure()
    plt.subplot(221)
    plt.hist(ln_exp * np.array(model_H['rhos'])[ind_H]*1e3, bins=20, label='H-mode', histtype='barstacked')
    plt.hist(ln_L_exp* np.array(model['rhos'])[ind_L]*1e3, bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    plt.xlabel(r'$\lambda_{n}$ [mm]', fontsize = 12)
    plt.text(80, 5, 'a)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.legend(fontsize=11)
    
    
    plt.subplot(222)
    (n, bins, patches) = plt.hist(np.array(exp_H['lT'])[ind_H]*1e3, bins=20, label='H-mode', histtype='barstacked')
    plt.hist(np.array(exp['lT'])[ind_L]*1e3, bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    plt.xlabel(r'$\lambda_{T}$ [mm]', fontsize=12)
    plt.text(44, 5, 'b)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.legend(fontsize=11)
    
    plt.subplot(223)
    (n, bins, patches) = plt.hist(np.array(exp_H['ne_U'])[ind_H], bins=20, label='H-mode', histtype='barstacked')
    plt.hist(np.array(exp['ne_U'])[ind_L], bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    plt.xlabel(r'$n_{e}$ [$\times 10^{19}m^{3}$]',fontsize=12)
    plt.text(1.8, 8, 'c)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.legend(fontsize=11)
    
    plt.subplot(224)
    (n, bins, patches) = plt.hist(np.array(exp_H['Te_U'])[ind_H], bins=20, label='H-mode', histtype='barstacked')
    plt.hist(np.array(exp['Te_U'])[ind_L], bins = 20, alpha=0.4,  label='L-mode', histtype='barstacked')
    plt.xlabel(r'$T_{e}$ [eV]',fontsize=12)
    plt.text(210,8, 'd)', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    fig2.savefig('ln_lT_ne_Te_hist_old_database.png', format='png', dpi=300)
    
    
    fig3 = plt.figure()
    (n, bins, patches) = plt.hist((1.0+(ks_H+kx_H)**2)**(9/11), bins=30,  label='H-mode', histtype='barstacked')
    plt.hist((1.0+(ks+kx)**2)**(9/11), bins = bins, alpha=0.5, label='L-mode', histtype='barstacked')
    plt.xlabel(r'Shear correction', fontsize =12)
    plt.legend(fontsize=12)
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    fig3.set_figheight(4)
    fig3.set_figwidth(6)
    plt.tight_layout()
    fig3.savefig('shear_correction_hist_old_database.png', format='png', dpi=300)
    
    
    # cm2 = plt.cm.get_cmap('plasma')
    # z = np.array(params_H['Prad_div'])[ind_H]*1e-6
    # z2 = np.array(params['Prad_div'])[ind_L]*1e-6
    # plt.figure()
    # plt.subplot(121)
    # sc = plt.scatter(ln_exp, ln_sol, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    # plt.colorbar(sc).ax.set_title(r'$' + 'g [1e-4]' +'$')
    # plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    # plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    # plt.subplot(122)
    # sc = plt.scatter(ln_L_exp[indnan], ln_L_sol[indnan], c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    # plt.colorbar(sc).ax.set_title(r'$' + 'g[1e-4]' +'$')
    # plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    # plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    # plt.title('L-mode')
    
    n_bin = 25
    indnan = np.where(np.isnan(lp_L_sol)==0)[0]
    plt.figure()
    ha, a, b, c  = plt.hist2d(lp_sol, lp_exp, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha2, a2, b2, c2 = plt.hist2d(lp_L_sol[indnan], lp_L_exp[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    plt.close()

    plt.figure()
    ha3, a3, b3, c3  = plt.hist2d(ln_sol, ln_exp, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha4, a4, b4, c4 = plt.hist2d(ln_L_sol[indnan], ln_L_exp[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha5, a5, b5, c5 = plt.hist2d(lT_sol, lT_exp, bins=n_bin, label = r'$\lambda_{T}$ H-mode', cmap='hot')
    plt.close()
    
    plt.figure()
    ha6, a6, b6, c6 = plt.hist2d(lT_L_sol[indnan], lT_L_exp[indnan], bins=n_bin, label = r'$\lambda_{T}$ L-mode', cmap='hot')
    plt.close()
    
    
    # plt.figure()
    # plt.pcolormesh(a6, b6, np.transpose(ha6))
    
    # import tikzplotlib
    # tikzplotlib.clean_figure()
    # tikzplotlib.save('test_figure_lT.tex', externalize_tables=True)

    
    vmin = 0.0
    vmax = 3.0
    # fig = plt.figure()
    # ax = fig.add_subplot(2, 4, 2)
    
    fontsize = 14
    fontticks = 10
    fig, ((ax, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) =  plt.subplots(2,4, gridspec_kw={'width_ratios': [1, 1, 1, 0.01]})
    # h = ax.hist2d(lp_exp, lp_sol, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    h = ax.imshow(ha, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b), np.max(b),np.min(a), np.max(a)])
    # cbar = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(r'$\lambda_{p}$')
    ax.plot([0,35],[0,35], '--w', label = r'1:1 $\pm 40\%$')
    ax.plot([0,35],[0,1.4*35], '--w')
    ax.plot([0,35],[0,0.6*35], '--w')
    ax.set_xlim((0,35))
    ax.set_ylim((0,35))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax.set_title(r'H-mode $\lambda_{p}$', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontticks)
    ax.tick_params(axis='y', labelsize=fontticks)
    ax.legend()
    ax.set_facecolor("black")

    # ax2 = fig.add_subplot(2, 4, 6)
    # h2 = ax2.hist2d(lp_L_exp[indnan], lp_L_sol[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    h2 = ax5.imshow(ha2, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b2), np.max(b2),np.min(a2), np.max(a2)])
    # cbar2 = fig.colorbar(h2, ax=ax2, fraction=0.046, pad=0.04)
    # cbar2.set_label(r'$\lambda_{p}$')
    
    ax5.plot([0,60],[0,60], '--w', label = r'1:1 $\pm 40\%$')
    ax5.plot([0,60],[0,1.4*60], '--w')
    ax5.plot([0,60],[0,0.6*60], '--w')
    ax5.set_xlim((0,60))
    ax5.set_ylim((0,60))
    ax5.set_facecolor("black")
    ax5.set_aspect('equal', 'box')
    ax5.set_title(r'L-mode $\lambda_{p}$', fontsize=fontsize)
    ax5.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax5.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax5.tick_params(axis='x', labelsize=fontticks)
    ax5.tick_params(axis='y', labelsize=fontticks)
    ax5.legend()
    fig.tight_layout()
    
    # fig = plt.figure()
    # ax3 = fig.add_subplot(2, 4, 3)
    # h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    h3 = ax2.imshow(ha3, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b3), np.max(b3),np.min(a3), np.max(a3)])    
    # cbar3 = fig.colorbar(h3, ax=ax3, fraction=0.046, pad=0.04)
    # cbar3.set_label(r'$\lambda_{n}$')
    ax2.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax2.plot([0,80],[0,1.4*80], '--w')
    ax2.plot([0,80],[0,0.6*80], '--w')
    ax2.set_xlim((0,80))
    ax2.set_ylim((0,80))
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax2.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax2.set_title(r'H-mode $\lambda_{n}$', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontticks)
    ax2.tick_params(axis='y', labelsize=fontticks)
    ax2.legend()
    ax2.set_facecolor("black")

    # ax4 = fig.add_subplot(2, 4, 7)
    # h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    h4 = ax6.imshow(ha4, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b4), np.max(b4),np.min(a4), np.max(a4)]) 
    # cbar4 = fig.colorbar(h4, ax=ax4, fraction=0.046, pad=0.04)
    # cbar4.set_label(r'$\lambda_{n}$')
    ax6.plot([0,150],[0,150], '--w', label = r'1:1 $\pm 40\%$')
    ax6.plot([0,150],[0,1.4*150], '--w')
    ax6.plot([0,150],[0,0.6*150], '--w')
    ax6.set_xlim((0,150))
    ax6.set_ylim((0,150))
    ax6.set_facecolor("black")
    ax6.set_aspect('equal', 'box')
    ax6.set_title(r'L-mode $\lambda_{n}$', fontsize=fontsize)
    ax6.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax6.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax6.tick_params(axis='x', labelsize=fontticks)
    ax6.tick_params(axis='y', labelsize=fontticks)
    ax6.legend()

    # fig = plt.figure()
    # ax5 = fig.add_subplot(2, 4, 4)
    # h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    h5 = ax3.imshow(ha5, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b5), np.max(b5),np.min(a5), np.max(a5)])    
    # cbar5 = fig.colorbar(h5, ax=ax5, fraction=0.046, pad=0.04)
    # cbar5.set_label(r'$\lambda_{T}$')
    ax3.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax3.plot([0,80],[0,1.4*80], '--w')
    ax3.plot([0,80],[0,0.6*80], '--w')
    ax3.set_xlim((0,80))
    ax3.set_ylim((0,80))
    ax3.set_aspect('equal', 'box')
    ax3.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax3.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax3.set_title(r'H-mode $\lambda_{T}$', fontsize=fontsize)
    ax3.tick_params(axis='x', labelsize=fontticks)
    ax3.tick_params(axis='y', labelsize=fontticks)
    ax3.legend()
    ax3.set_facecolor("black")

    # ax6= fig.add_subplot(2, 4, 8)
    # h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    h6 = ax7.imshow(ha6, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b6), np.max(b6),np.min(a6), np.max(a6)]) 
    # cbar6 = fig.colorbar(h6, ax=ax6, fraction=0.04, pad=0.04)
    # cbar6.set_label('Point concentration (a.u.)')
    ax7.plot([0,150],[0,150], '--w', label = r'1:1 $\pm 40\%$')
    ax7.plot([0,150],[0,1.4*150], '--w')
    ax7.plot([0,150],[0,0.6*150], '--w')
    ax7.set_xlim((0,100))
    ax7.set_ylim((0,100))
    ax7.set_facecolor("black")
    ax7.set_aspect('equal', 'box')
    ax7.set_title(r'L-mode $\lambda_{T}$', fontsize=fontsize)
    ax7.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax7.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax7.tick_params(axis='x', labelsize=fontticks)
    ax7.tick_params(axis='y', labelsize=fontticks)
    ax7.legend()
    
    # ax8= fig.add_subplot(2, 4, 5)
    fig.delaxes(ax8)
    
    # ax7= fig.add_subplot(2, 4, 1)
    ax4.axis('off')
    cbar7 = fig.colorbar(h6, ax=ax4, fraction=20.0, pad=8, location='right')
    cbar7.set_label('Point concentration (a.u.)', fontsize=12)
    cbar7.ax.tick_params(labelsize=fontticks)
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.tight_layout()
    
    fig.savefig('compare_lambdas_SSF_exp.png', format='png', dpi=300)

    vmin = 0.0
    vmax = 3.0
    # fig = plt.figure()
    # ax = fig.add_subplot(2, 4, 2)
    
    fontsize = 14
    fontticks = 11
    fig, ((ax, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) =  plt.subplots(3,3, gridspec_kw={'width_ratios': [1, 1, 0.01]})
    # h = ax.hist2d(lp_exp, lp_sol, bins=n_bin, label = r'$\lambda_{p}$ H-mode', cmap='hot')
    h = ax.imshow(ha, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b), np.max(b),np.min(a), np.max(a)])
    # cbar = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(r'$\lambda_{p}$')
    ax.plot([0,35],[0,35], '--w', label = r'1:1 $\pm 40\%$')
    ax.plot([0,35],[0,1.4*35], '--w')
    ax.plot([0,35],[0,0.6*35], '--w')
    ax.set_xlim((0,35))
    ax.set_ylim((0,35))
    ax.set_xticks([0,10,20,30])
    ax.set_yticks([0,10,20,30])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax.set_title(r'H-mode $\lambda_{p}$', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontticks)
    ax.tick_params(axis='y', labelsize=fontticks)
    ax.legend()
    ax.set_facecolor("black")

    # ax2 = fig.add_subplot(2, 4, 6)
    # h2 = ax2.hist2d(lp_L_exp[indnan], lp_L_sol[indnan], bins=n_bin, label = r'$\lambda_{p}$ L-mode', cmap='hot')
    h2 = ax2.imshow(ha2, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b2), np.max(b2),np.min(a2), np.max(a2)])
    # cbar2 = fig.colorbar(h2, ax=ax2, fraction=0.046, pad=0.04)
    # cbar2.set_label(r'$\lambda_{p}$')
    
    ax2.plot([0,60],[0,60], '--w', label = r'1:1 $\pm 40\%$')
    ax2.plot([0,60],[0,1.4*60], '--w')
    ax2.plot([0,60],[0,0.6*60], '--w')
    ax2.set_xlim((0,60))
    ax2.set_ylim((0,60))
    ax2.set_facecolor("black")
    ax2.set_aspect('equal', 'box')
    ax2.set_title(r'L-mode $\lambda_{p}$', fontsize=fontsize)
    ax2.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax2.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontticks)
    ax2.tick_params(axis='y', labelsize=fontticks)
    ax2.legend()

    
    # fig = plt.figure()
    # ax3 = fig.add_subplot(2, 4, 3)
    # h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    h3 = ax4.imshow(ha3, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b3), np.max(b3),np.min(a3), np.max(a3)])    
    # cbar3 = fig.colorbar(h3, ax=ax3, fraction=0.046, pad=0.04)
    # cbar3.set_label(r'$\lambda_{n}$')
    ax4.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax4.plot([0,80],[0,1.4*80], '--w')
    ax4.plot([0,80],[0,0.6*80], '--w')
    ax4.set_xlim((0,80))
    ax4.set_ylim((0,80))
    ax4.set_xticks([0, 20, 40, 60, 80])
    ax4.set_yticks([0, 20, 40, 60, 80])
    ax4.set_aspect('equal', 'box')
    ax4.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax4.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax4.set_title(r'H-mode $\lambda_{n}$', fontsize=fontsize)
    ax4.tick_params(axis='x', labelsize=fontticks)
    ax4.tick_params(axis='y', labelsize=fontticks)
    ax4.legend()
    ax4.set_facecolor("black")

    # ax4 = fig.add_subplot(2, 4, 7)
    # h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    h4 = ax5.imshow(ha4, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b4), np.max(b4),np.min(a4), np.max(a4)]) 
    # cbar4 = fig.colorbar(h4, ax=ax4, fraction=0.046, pad=0.04)
    # cbar4.set_label(r'$\lambda_{n}$')
    ax5.plot([0,150],[0,150], '--w', label = r'1:1 $\pm 40\%$')
    ax5.plot([0,150],[0,1.4*150], '--w')
    ax5.plot([0,150],[0,0.6*150], '--w')
    ax5.set_xlim((0,150))
    ax5.set_ylim((0,150))
    ax5.set_facecolor("black")
    ax5.set_aspect('equal', 'box')
    ax5.set_title(r'L-mode $\lambda_{n}$', fontsize=fontsize)
    ax5.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax5.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax5.tick_params(axis='x', labelsize=fontticks)
    ax5.tick_params(axis='y', labelsize=fontticks)
    ax5.legend()

    # fig = plt.figure()
    # ax5 = fig.add_subplot(2, 4, 4)
    # h3 = ax3.hist2d(ln_exp, ln_sol, bins=n_bin, label = r'$\lambda_{n}$ H-mode', cmap='hot')
    h5 = ax7.imshow(ha5, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b5), np.max(b5),np.min(a5), np.max(a5)])    
    # cbar5 = fig.colorbar(h5, ax=ax5, fraction=0.046, pad=0.04)
    # cbar5.set_label(r'$\lambda_{T}$')
    ax7.plot([0,80],[0,80], '--w', label = r'1:1 $\pm 40\%$')
    ax7.plot([0,80],[0,1.4*80], '--w')
    ax7.plot([0,80],[0,0.6*80], '--w')
    ax7.set_xlim((0,80))
    ax7.set_ylim((0,80))
    ax7.set_aspect('equal', 'box')
    ax7.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax7.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax7.set_title(r'H-mode $\lambda_{T}$', fontsize=fontsize)
    ax7.tick_params(axis='x', labelsize=fontticks)
    ax7.tick_params(axis='y', labelsize=fontticks)
    ax7.set_xticks([0, 20, 40, 60, 80])
    ax7.set_yticks([0, 20, 40, 60, 80])
    ax7.legend()
    ax7.set_facecolor("black")

    # ax6= fig.add_subplot(2, 4, 8)
    # h4 = ax4.hist2d(ln_L_exp[indnan], ln_L_sol[indnan], bins=n_bin, label = r'$\lambda_{n}$ L-mode', cmap='hot')
    h6 = ax8.imshow(ha6, cmap ='hot', origin = "lower", interpolation = "gaussian", vmin=vmin, vmax = vmax, extent=[np.min(b6), np.max(b6),np.min(a6), np.max(a6)]) 
    # cbar6 = fig.colorbar(h6, ax=ax6, fraction=0.04, pad=0.04)
    # cbar6.set_label('Point concentration (a.u.)')
    ax8.plot([0,150],[0,150], '--w', label = r'1:1 $\pm 40\%$')
    ax8.plot([0,150],[0,1.4*150], '--w')
    ax8.plot([0,150],[0,0.6*150], '--w')
    ax8.set_xlim((0,100))
    ax8.set_ylim((0,100))
    ax8.set_facecolor("black")
    ax8.set_aspect('equal', 'box')
    ax8.set_title(r'L-mode $\lambda_{T}$', fontsize=fontsize)
    ax8.set_xlabel(r'Experiment [$\rho_{s}$]', fontsize=fontsize)
    ax8.set_ylabel(r'Model [$\rho_{s}$]', fontsize=fontsize)
    ax8.tick_params(axis='x', labelsize=fontticks)
    ax8.tick_params(axis='y', labelsize=fontticks)
    ax8.legend()
    
    # ax8= fig.add_subplot(2, 4, 5)
    fig.delaxes(ax6)
    fig.delaxes(ax9)
    
    # ax7= fig.add_subplot(2, 4, 1)
    ax3.axis('off')
    cbar7 = fig.colorbar(h6, ax=ax3, fraction=20.0, pad=8, location='right')
    cbar7.set_label('Point concentration (a.u.)', fontsize=fontsize)
    cbar7.ax.tick_params(labelsize=fontticks+1)
    fig.set_figheight(8)
    fig.set_figwidth(6.5)
    plt.tight_layout()
    fig.savefig('compare_lambdas_SSF_exp_vertical.png', format='png', dpi=300)
        
if 0:
    plt.figure()
    plt.plot((np.array(exp_H['ne_U'])/np.array(exp_H['Te_U'])**(3/2))[ind_H], gamma_sol, 'o', label = 'H-mode')
    plt.plot((np.array(exp['ne_U'])/np.array(exp['Te_U'])**(3/2))[ind_L], gamma_L_sol, 'o', label = 'L-mode')
    plt.legend()
    plt.xlabel(r'Collisionality proxy')
    plt.ylabel(r'$\gamma$')
    
    plt.figure()
    plt.hist2d((np.array(exp_H['ne_U'])/np.array(exp_H['Te_U'])**(3/2))[ind_H], gamma_sol, bins=20, label = 'H-mode')
    plt.figure()
    plt.hist2d((np.array(exp['ne_U'])/np.array(exp['Te_U'])**(3/2))[ind_L], gamma_L_sol, bins = 20, label = 'L-mode')
    # plt.legend()
    # plt.xlabel(r'Collisionality proxy')
    # plt.ylabel(r'$\gamma$')
    
    plt.figure()
    plt.loglog((np.array(exp_H['ne_U'])/np.array(exp_H['Te_U'])**(3/2))[ind_H], gamma_sol, 'o', label = 'H-mode')
    plt.loglog((np.array(exp['ne_U'])/np.array(exp['Te_U'])**(3/2))[ind_L], gamma_L_sol, 'o', label = 'L-mode')
    plt.legend()
    plt.xlabel(r'Collisionality proxy')
    plt.ylabel(r'$\gamma$')
    
    plt.figure()
    plt.plot(Te, lp_sol ,'o', label='H-mode')
    plt.plot(Te_L, lp_L_sol, ' o', label='L-mode')
    plt.legend()
    plt.xlabel(r'$T_{e}$ [eV]')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    plt.figure()
    plt.plot(Te, ln_sol ,'o', label='H-mode')
    plt.plot(Te_L, ln_L_sol, ' o', label='L-mode')
    plt.legend()
    plt.xlabel(r'$T_{e}$ [eV]')
    plt.ylabel(r'$\lambda_{n} [\rho_{s}]$')
    
    plt.figure()
    plt.plot(ks_H, kx_H, 'o', label = 'H-mode')
    plt.plot(ks, kx, 'o', label = 'L-mode')
    plt.legend()
    plt.xlabel(r'$\alpha_{s}$')
    plt.ylabel(r'$\alpha_{E{\times}B}$')
    
    
    
    plt.figure()
    plt.errorbar(lp_exp, lp_sol, None, lp_exp_err,  'o', label = 'H-mode')
    plt.errorbar(lp_L_exp, lp_L_sol, None, lp_L_exp_err,  'o', label = 'L-mode')
    # plt.plot(lp_L_exp, lp_L_sol, 'o', label = 'L-mode')
    plt.plot([0,100],[0,100], '--k', label = '1:1')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title(r'$\lambda_{p}$')
    
    plt.figure()
    plt.errorbar(ln_exp, ln_sol, None, ln_exp_err,  'o', label = 'H-mode')
    plt.errorbar(ln_L_exp, ln_L_sol, None, ln_L_exp_err,  'o', label = 'L-mode')
    plt.plot([0,200],[0,200], '--k', label = '1:1')
    plt.legend()
    plt.xlabel(r'Experiment [$\rho_{s}$]')
    plt.ylabel(r'Model [$\rho_{s}$]')
    plt.title(r'$\lambda_{n}$')
    
    plt.figure()
    plt.plot(gamma_sol, Te, 'o', label='H-mode')
    plt.plot(gamma_L_sol, Te_L, 'o', label='L-mode')
    plt.legend()
    plt.ylabel(r'$T_{e}$ [eV]')
    plt.xlabel(r'$\gamma$')
    
    plt.figure()
    plt.plot(np.array(params_H['H98'])[ind_H], lp_sol, 'o', label = 'H-mode')
    plt.plot(np.array(params['H98'])[ind_L], lp_L_sol, 'o', label = 'L-mode')
    plt.legend()
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.xlabel(r'$H_{98}$')
    
    plt.figure()
    plt.plot(np.array(params_H['H98'])[ind_H], lq*np.array(model_H['rhos'])[ind_H]*1e3, 'o', label = 'H-mode')
    plt.plot(np.array(params['H98'])[ind_L], lq_L*np.array(model['rhos'])[ind_L]*1e3, 'o', label = 'L-mode')
    plt.legend()
    plt.ylabel(r'$\lambda_{q} [mm]$')
    plt.xlabel(r'$H_{98}$')
    
    plt.figure()
    plt.plot(lp_exp*np.array(model_H['rhos'])[ind_H]*1e3, lp_sol*np.array(model_H['rhos'])[ind_H]*1e3, 'o', label = 'H-mode')
    plt.plot(lp_L_exp*np.array(model['rhos'])[ind_L]*1e3, lp_L_sol*np.array(model['rhos'])[ind_L]*1e3, 'o', label = 'L-mode')
    plt.plot([0,40],[0,40], '--k', label = '1:1')
    plt.legend()
    plt.xlabel(r'Experiment [$mm$]')
    plt.ylabel(r'Model [$mm$]')
    plt.title(r'$\lambda_{p}$')
    
    plt.figure()
    plt.plot(ln_exp*np.array(model_H['rhos'])[ind_H]*1e3, ln_sol*np.array(model_H['rhos'])[ind_H]*1e3, 'o', label = 'H-mode')
    plt.plot(ln_L_exp*np.array(model['rhos'])[ind_L]*1e3, ln_L_sol*np.array(model['rhos'])[ind_L]*1e3, 'o', label = 'L-mode')
    plt.plot([0,80],[0,80], '--k', label = '1:1')
    plt.legend()
    plt.xlabel(r'Experiment [$mm$]')
    plt.ylabel(r'Model [$mm$]')
    plt.title(r'$\lambda_{n}$')
    
    k = np.linspace(0,13,50)
    plt.figure()
    plt.subplot(121)
    plt.plot((kx_H+ks_H)**2, lp_exp, 'o', label = 'H-mode')
    plt.plot((kx+ks)**2, lp_L_exp, 'o', label = 'L-mode')
    plt.plot(k, 60*(1+k)**(-9/11), '--k', label='60*(1+k)**(-9/11)')
    plt.plot(k, 50*(1+k)**(-9/11), '--g', label='30*(1+k)**(-9/11)')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title(r'$\lambda_{p}$')
    plt.legend()
    plt.subplot(122)
    plt.plot((kx_H+ks_H)**2, ln_exp, 'o', label = 'H-mode')
    plt.plot((kx+ks)**2, ln_L_exp, 'o', label = 'L-mode')
    plt.plot(k, 60*(1+k)**(-9/11), '--k', label='100*(1+k)**(-9/11)')
    plt.plot(k, 50*(1+k)**(-9/11), '--g', label='140*(1+k)**(-9/11)')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.legend()
    plt.title(r'$\lambda_{n}$')
    
    # plt.figure()
    # plt.hist(lp_exp*(1.0+(kx_H+ks_H)**2)**(9/11)*gamma_sol**(4/11), 50, label='H-mode')
    # plt.hist(lp_L_exp*(1.0+(kx+ks)**2)**(9/11)*gamma_L_sol**(4/11), 50, label = 'L-mode')
    
    z = np.array(params_H['q95'])[ind_H]
    X = (kx_H+ks_H)**2
    # Y = np.array(model['ln'])/(1+(-0.5*np.sign(np.array(params['Z_OSP']))+model['kx_corr'])**2)**(9/11)/model['rhos']
    Y = lp_exp
    
    z2 = np.array(params['q95'])[ind_L]
    X2 = (kx+ks)**2
    Y2 = lp_L_exp
    
    cm2 = plt.cm.get_cmap('plasma')
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'q95' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'q95' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = np.array(model_H['g'])[ind_H]*1e4
    
    z2 = np.array(model['g'])[ind_L]*1e4
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'g [1e-4]' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'g[1e-4]' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = gamma_sol
    z2 = gamma_L_sol
    
    cm2 = plt.cm.get_cmap('plasma')
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + '\gamma' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + '\gamma' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    
    z = (kx_H/ks_H)
    z2 = kx/ks
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$\alpha_{E{\times}B}/\alpha_{s}$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$\alpha_{E{\times}B}/\alpha_{s}$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = Te
    z2 = Te_L
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'T_{e}' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'T_{e}' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    
    z = np.array(params_H['H98'])[ind_H]
    z2 = np.array(params['H98'])[ind_L]
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'H98' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'H98' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = np.array(model_H['sig_PHI'])[ind_H]*1e4
    
    z2 = np.array(model['sig_PHI'])[ind_L]*1e4
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + '\sigma [1e-4]' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + '\sigma [1e-4]' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = np.sign(np.array(params_H['BT'])[ind_H])
    z2 = np.sign(np.array(params['BT'])[ind_L])
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter((kx_H+ks_H)**2, lp_exp, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter((kx+ks)**2, lp_L_exp, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    z = np.sign(np.array(params_H['BT'])[ind_H])/ np.sign(np.array(params_H['Ip'])[ind_H])
    z2 = np.sign(np.array(params['BT'])[ind_L])/np.sign(np.array(params['Ip'])[ind_L])
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter((kx_H+ks_H)**2, lp_exp, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign / Ip sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter((kx+ks)**2, lp_L_exp, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign / Ip sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    
    z = np.array(exp_H['ne_U'])[ind_H]/np.array(exp_H['ne_U'])[ind_H]**(3/2)
    z2 = np.array(exp['ne_U'])[ind_L]/np.array(exp['Te_U'])[ind_L]**(3/2)
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(ln_exp, ln_sol, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.plot([0,100],[0,100], '--k', label = '1:1')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter(ln_L_exp, ln_L_sol, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.plot([0,100],[0,100], '--k', label = '1:1')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    
    z = np.array(params_H['H98'])[ind_H]
    z2 = np.array(params['H98'])[ind_L]
    
    X = ln_exp*np.array(model_H['rhos'])[ind_H]*1e3
    Y = ln_sol*np.array(model_H['rhos'])[ind_H]*1e3
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter(X, Y, c=z, vmin=0.3, vmax=1.7, s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'H98' +'$')
    plt.plot([0,80], [0,80], '--k')
    plt.xlabel(r'$\lambda_{n}  exp [mm]$')
    plt.ylabel(r'$\lambda_{n}  model [mm]$')
    plt.title('H-mode')
    
    X2 = ln_L_exp*np.array(model['rhos'])[ind_L]*1e3
    Y2 = ln_L_sol*np.array(model['rhos'])[ind_L]*1e3
    plt.subplot(122)
    sc = plt.scatter(X2, Y2, c=z2, vmin=0.3, vmax=1.7, s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'H98' +'$')
    plt.plot([0,80], [0,80], '--k')
    plt.xlabel(r'$\lambda_{n}  exp [mm]$')
    plt.ylabel(r'$\lambda_{n}  model [mm]$')
    plt.title('L-mode')
    
    
    X = lp_exp*np.array(model_H['rhos'])[ind_H]*1e3
    Y = lp_sol*np.array(model_H['rhos'])[ind_H]*1e3
    
    plt.figure()
    plt.subplot(121)
    (n, bins, patches) = plt.hist(X, bins=25)
    plt.hist(Y, bins = bins, alpha=0.3)
    plt.xlabel(r'$\lambda_{p}[mm]$')
    
    
    X2 = ln_L_exp*np.array(model['rhos'])[ind_L]*1e3
    Y2 = ln_L_sol*np.array(model['rhos'])[ind_L]*1e3
    plt.subplot(122)
    (n, bins2, patches) = plt.hist(X2, bins=25)
    plt.hist(Y2,bins = bins2, alpha=0.3)
    plt.xlabel(r'$\lambda_{p}[mm]$')
    plt.title('L-mode')
    
    z = np.sign(np.array(params_H['BT'])[ind_H])
    z2 = np.sign(np.array(params['BT'])[ind_L])
    
    plt.figure()
    plt.subplot(121)
    sc = plt.scatter((np.array(exp_H['ne_U'])/np.array(exp_H['Te_U'])**(3/2))[ind_H], gamma_sol, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('H-mode')
    plt.subplot(122)
    sc = plt.scatter((np.array(exp['ne_U'])/np.array(exp['Te_U'])**(3/2))[ind_L], gamma_L_sol, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    plt.title('L-mode')
    
    plt.figure()
    
    sc = plt.scatter((np.array(exp_H['ne_U'])/np.array(exp_H['Te_U'])**(3/2))[ind_H], gamma_sol, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
    # plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    sc = plt.scatter((np.array(exp['ne_U'])/np.array(exp['Te_U'])**(3/2))[ind_L], gamma_L_sol, c=z2, vmin=np.nanmin(z2), vmax=np.nanmax(z2), s=35, cmap=cm2)
    plt.colorbar(sc).ax.set_title(r'$' + 'BT sign' +'$')
    plt.xlabel(r'$(\alpha_{E{\times}B}+\alpha_{s})^{2}$')
    plt.ylabel(r'$\lambda_{p} [\rho_{s}]$')
    
    
    plt.figure()
    plt.plot(gamma_sol, np.array(exp_H['ne_U'])[ind_H], 'o', label='H-mode')
    plt.plot(gamma_L_sol, np.array(exp['ne_U'])[ind_L], 'o', label='L-mode')
    plt.legend()
    plt.ylabel(r'$T_{e}$ [eV]')
    plt.xlabel(r'$\gamma$')
