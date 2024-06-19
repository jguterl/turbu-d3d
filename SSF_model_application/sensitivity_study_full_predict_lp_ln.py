#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:40:28 2024

@author: peretm
"""

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
from matplotlib import colors

plt.close('all')
exp = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_exp_2.npy',allow_pickle=True).tolist()
model = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_model_2.npy',allow_pickle=True).tolist()
params = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/database_treatment/dataset_SSF/database_params_2.npy',allow_pickle=True).tolist()


# exp = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_exp_correc_no_shift_L_mode.npy',allow_pickle=True).tolist()
# model = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_model_correc_no_shift_L_mode.npy',allow_pickle=True).tolist()
# params = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/database_params_correc_no_shift_L_mode.npy',allow_pickle=True).tolist()


fudge0 = np.linspace(1.5,2.5, 10)
gammarange = np.linspace(4,8,10)


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
ind_H = ind_H[np.where(np.sqrt(1.602e-19/9.11e-31*1e3)*(np.array(exp_H['Te_U'])[ind_H]*1e-3)**(2)/(np.array(exp_H['ne_U'])[ind_H]*0.1)/9.11/1.36e5<0.8)[0]]

lp_exp = ((1.0/(1.0/np.array(exp_H['ln'])+1.0/np.array(exp_H['lT']))/np.array(model_H['rhos'])))[ind_H]
ln_exp = (np.array(exp_H['ln'])/np.array(model_H['rhos']))[ind_H]
lp_exp_err = 0.0 * lp_exp 
ln_exp_err = 0.0 * lp_exp 
ln_sol = 0.0 * lp_exp
lp_sol = 0.0 * lp_exp
lp_sol_or = 0.0 * lp_exp

n_sol = 0.0 * lp_exp
gamma_sol = 0.0 * lp_exp
kx_H = 0.0 * lp_exp
ks_H = 0.0 * lp_exp
Te = 0.0 * lp_exp
# i = 154
j = -1

err = np.zeros((len(fudge0), len(gammarange)))


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
        lp_sol_or[j] = lp[ind_sol[-1]]
      
        
        ln_sol[j] = np.sqrt(gamma) * lp_sol_or[j]

    except:
        # plt.figure()
        # plt.plot(lp, f, '-o')
        # breakpoint()
        lp_sol_or[j] = np.nan
        ln_sol[j] = np.nan
        []
    
k = -1
l = -1
for gamma in gammarange:
    k=k+1
    l=-1
    for fudge in fudge0:
        l=l+1
        j = -1
        for i in ind_H:
            j = j+1
            l0 = 3.9*model_H['g'][i]**(3/11)*(4*model_H['sig_PHI'][i])**(-2/11)*(4*model_H['sig_N'][i])**(-4/11)
            # gamma = (np.array(exp_H['ln'])[i]/lp_exp[j]/np.array(model_H['rhos'])[i])**2
            # gamma = gamma0
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
            
            err[l,k] = np.mean(100*(lp_sol/lp_sol_or-1))
        

fontsize = 13
fontticks = 11
# breakpoint()
# lq = lp_exp / (1.0+0.5*(np.sqrt(gamma_sol)-1.0)/np.sqrt(gamma_sol)) * np.array(model_H['rhos'])[ind_H]
divnorm=colors.TwoSlopeNorm(vmin=-45., vcenter=0., vmax=45)
fig = plt.figure()
ax = plt.subplot(121)
h = plt.imshow(err, extent=(gammarange.min(), gammarange.max(), fudge0.max(), fudge0.min()),
           interpolation='nearest', cmap='bwr', norm=divnorm)
plt.plot([5.3,5.3], [np.min(fudge0), np.max(fudge0)], '--k')
plt.plot([np.min(gammarange), np.max(gammarange)],[2.0,2.0], '--k')
# cbar = fig.colorbar(h, ax=ax, fraction=0.076, pad=0.04)
# cbar.set_label(r'mean $\epsilon$ [\%]')
ax.set_xlabel(r'$\gamma$', fontsize = fontsize)
ax.set_ylabel(r'$\beta$', fontsize = fontsize)
ax.tick_params(axis='x', labelsize=fontticks)
ax.tick_params(axis='y', labelsize=fontticks)
ax.set_aspect('auto', 'box')
ax.invert_yaxis()
plt.title('H-mode', fontsize = fontsize)

lq = 2/7 * np.array(exp_H['lT'])[ind_H]



ind_L = np.where(np.array(params['H98'])<0.6)[0]
ind_L = ind_L[np.where(np.array(params['H98'])[ind_L]>0.25)[0]]
ind_L = ind_L[np.where(np.array(exp['lT_err'])[ind_L]/np.array(exp['lT'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['ln_err'])[ind_L]/np.array(exp['ln'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['ne_U_err'])[ind_L]/np.array(exp['ne_U'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.array(exp['Te_U_err'])[ind_L]/np.array(exp['Te_U'])[ind_L]<=err_lim)[0]]
ind_L = ind_L[np.where(np.sqrt(1.602e-19/9.11e-31*1e3)*(np.array(exp['Te_U'])[ind_L]*1e-3)**(2)/(np.array(exp['ne_U'])[ind_L]*0.1)/9.11/1.36e5<0.8)[0]]


Bpol_L = mu0*np.abs(np.array(params['Ip'])[ind_L])/np.array(params['a'])[ind_L]/np.sqrt(0.5*(1.0+np.array(params['kappa'])[ind_L]**2))/2.0/math.pi

lp_L_exp = ((1.0/(1.0/np.array(exp['ln'])+1.0/np.array(exp['lT']))/np.array(model['rhos'])))[ind_L]
ln_L_exp = (np.array(exp['ln'])/np.array(model['rhos']))[ind_L]


lp_L_exp_err = 0.0 * lp_L_exp 
ln_L_exp_err = 0.0 * lp_L_exp 
lp_L_sol = 0.0 * lp_L_exp +0.0
lp_L_sol_or = 0.0 * lp_L_exp +0.0
ln_L_sol = 0.0 * lp_L_exp+0.0
n_sol_L = 0.0 * lp_L_exp+0.0
gamma_L_sol = 0.0 * lp_L_exp+0.0
kx = 0.0 * lp_L_exp+0.0
ks = 0.0 * lp_L_exp+0.0
Te_L = 0.0 * lp_L_exp+0.0
# i = 154
j = -1

err_L = np.zeros((len(fudge0), len(gammarange)))
gamma0 = 5.3
fudge = 2.0
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
        lp_L_sol_or[j] = lp[ind_sol[-1]]
        ln_L_sol[j] = lp_L_sol_or[j] * np.sqrt(gamma)
        kx[j] = -0.43*fudge*3.0/model['g'][i]**(0.5)*ll**(-3/2)/alpha_lT**2
        ks[j] = model['ks'][i]
        Te_L[j] = exp['Te_U'][i]
    except:
        lp_L_sol_or[j] = np.nan
        ln_L_sol[j] = np.nan
        kx[j] = np.nan
        ks[j] = np.nan
        Te_L[j] = np.nan
        []

k = -1
l = -1
for gamma in gammarange:
    k=k+1
    l=-1
    for fudge in fudge0:
        l=l+1
        j = -1
        for i in ind_L:
            j = j+1
            l0 = 3.9*model['g'][i]**(3/11)*(4*model['sig_PHI'][i])**(-2/11)*(4*model['sig_N'][i])**(-4/11)
            # gamma = (np.array(exp['ln'])[i]/lp_L_exp[j]/np.array(model['rhos'])[i])**2
            # gamma = gamma0
            
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

            
            err_L[l,k] = np.mean(100*(lp_L_sol/lp_L_sol_or-1))
 
divnorm2=colors.TwoSlopeNorm(vmin=-25., vcenter=0., vmax=25)
ax2 = plt.subplot(122)
h2 = plt.imshow(err_L, extent=(gammarange.min(), gammarange.max(), fudge0.max(), fudge0.min()),
           interpolation='nearest', cmap='bwr', norm=divnorm)

plt.plot([5.3,5.3], [np.min(fudge0), np.max(fudge0)], '--k')
plt.plot([np.min(gammarange), np.max(gammarange)],[2.0,2.0], '--k')
cbar = fig.colorbar(h2, ax=ax2, fraction=0.076, pad=0.04)
cbar.set_label(r'mean $\epsilon$ [\%]', fontsize = fontsize)
ax2.set_xlabel(r'$\gamma$', fontsize = fontsize)
ax2.set_ylabel(r'$\beta$', fontsize = fontsize)
ax2.tick_params(axis='x', labelsize=fontticks)
ax2.tick_params(axis='y', labelsize=fontticks)
ax2.set_aspect('auto', 'box')
ax2.invert_yaxis()
plt.title('L-mode', fontsize = fontsize)
fig.set_figheight(4)
fig.set_figwidth(8)
plt.tight_layout()

fig.savefig('sensitivity_ln_lp_Lmode_Hmode.png', format='png', dpi=300)
