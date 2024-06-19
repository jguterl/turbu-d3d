#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:17:22 2024

@author: peretm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp2d
import math

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

k = 60

shot = np.array(params_H['shot'])[ind_H[k]]
t0 = np.array(params_H['t0'])[ind_H[k]]
t1 = np.array(params_H['t1'])[ind_H[k]]

plt.close('all')
database = np.load('/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/saved_plateaus/no_shift/shot_' +str(shot) + '_t0_' +str(t0) + '_t1_' +str(t1) + 'auto.npy',allow_pickle=True).tolist()
psi= database['eq']['psin'][0,:,:]
R = database['eq']['r']
Z = database['eq']['z']

R0 = np.mean(database['eq']['rmaxis'])
Z0 = np.mean(database['eq']['zmaxis'])

# fig, ax =plt.subplots(1)
# im = ax.pcolormesh(R,Z,psi, cmap='hot', shading='auto')#, norm=colors.LogNorm(vmin=psi.min(), vmax=psi.max()))
# fig.colorbar(im,ax=ax)
# plt.title(r'Electron density $[m^{-3}]$')
# ax.set_xlabel('R [m]')
# ax.set_ylabel('Z [m]')

# R_sep = []
# Z_sep = []
# plt.figure()
# cs = plt.contour(R, Z, np.mean(database['eq']['psin'],axis=0), levels = [1])
# for item in cs.collections:
#    for i in item.get_paths():
#       v = i.vertices
#       R_sep = np.append(R_sep, v[:, 0])
#       Z_sep = np.append(Z_sep, v[:, 1])
# plt.close()



# dR_FT = 0e-2
# r_FT = 5e-3
# ind_OMP = np.where(R_sep>=R0)[0]
# ind_OMP = ind_OMP[np.argmin(np.abs(Z_sep[ind_OMP]-Z0))]

ind_R = np.where(R>=R0)[0]
ind_Z = np.argmin(np.abs(Z-Z0))

R_int = np.linspace(np.min(R[ind_R]), np.max(R[ind_R]), 200)
Z_int = 0.0*R_int + Z[ind_Z]
psi_out = 0.0*R_int + Z[ind_Z]

psi_int =  interp2d(R, Z, psi)

for i in range(len(psi_out)):
    psi_out[i] = psi_int(R_int[i],Z_int[i])

ind_sep = np.argmin(np.abs(psi_out-1))
R_sep = R_int[ind_sep]
database['ts_data']['R'] = database['ts_data']['psin_TS'] * 0.0 +0.0
fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [1, 0.4]})

neU = np.floor(database['ts_fit']['linear log']['ne_U']*100)/100
TeU = np.floor(database['ts_fit']['linear log']['Te_U']*10)/10
ln = np.floor(database['ts_fit']['linear log']['lne_U']*1e3)
lT = np.floor(database['ts_fit']['linear log']['lTe_U']*1e3)

dne = np.floor(database['ts_fit']['linear log']['ne_U_err']*100)/100
dTe = np.floor(database['ts_fit']['linear log']['Te_U_err']*10)/10

dln =  np.floor(database['ts_fit']['linear log']['lne_U_err']*1e3)
dlT =  np.floor(database['ts_fit']['linear log']['lTe_U_err']*1e3)

alpha = 0.1
fontsize = 14
for i in range(len(database['ts_data']['time'])):
    database['ts_data']['R'][:,i] = (np.interp(database['ts_data']['psin_TS'][:,i], psi_out, R_int))
    indmin = np.argmin(np.abs(database['ts_data']['psin_TS'][:,i]-0.95))
    indmax = np.argmin(np.abs(database['ts_data']['psin_TS'][:,i]-1.05))
    
    if i==0:
        ax1.semilogy(database['ts_data']['R'][indmax:indmin,i], database['ts_data']['density'][indmax:indmin,i]*1e-19, 'bo', alpha=alpha, label=r'experimental density [$\times 10^{19}m^{-3}$]')
        ax1.semilogy(database['ts_data']['R'][indmax:indmin,i], database['ts_data']['temp'][indmax:indmin,i], 'ro', alpha=alpha, label='experimental temperature [eV]')
    else:
        ax1.semilogy(database['ts_data']['R'][indmax:indmin,i], database['ts_data']['density'][indmax:indmin,i]*1e-19, 'bo', alpha=alpha)
        ax1.semilogy(database['ts_data']['R'][indmax:indmin,i], database['ts_data']['temp'][indmax:indmin,i], 'ro', alpha=alpha)

R_fit = (np.interp(database['ts_fit']['linear log']['psi'], psi_out, R_int))
ax1.semilogy(R_fit, database['ts_fit']['linear log']['ne'], '--b', label= r'fit $n_{e,sep}=$' + str(neU*10)+ r'$\pm$' +str(dne*10) + r' $\times 10^{18}m^{-3}$ / $\lambda_{n}=$' +str(ln) + r'$\pm$' +str(dln) +  'mm')
ax1.semilogy(R_fit, database['ts_fit']['linear log']['Te'], '--r', label= r'fit $T_{e,sep}=$' + str(TeU)+ r'$\pm$' +str(dTe) + r' $eV$ / $\lambda_{T}=$' +str(lT) + r'$\pm$' + str(dln) +'mm')

ax1.semilogy([R_sep, R_sep], [0.9e-1,2.0e3], '--k', label='separatrix')
ax1.set_title('DIII-D #'+str(shot) + r' $t\in[$' + str(t0/1e3) + ',' + str(t1/1e3) +r'$]$s', fontsize=fontsize)
ax1.set_xlim((np.interp(0.95, psi_out, R_int),np.interp(1.05, psi_out, R_int)))
ax1.set_xlabel('Major radius [m]', fontsize=fontsize)
ax1.set_ylabel('', fontsize=fontsize)
fig.delaxes(ax2)
ax1.legend(fontsize = fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.2))
fig.set_figheight(6.5)
fig.set_figwidth(8)
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize)
ax1.grid()
fig.savefig('fit_TS_data_DIIID.png', format='png', dpi=300)
# plt.tight_layout()
