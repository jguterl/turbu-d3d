#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:42:29 2024

@author: peretm
"""



import os
import numpy as np
import math
import matplotlib.pyplot as plt
from get_G0_ks import get_G0_ks
from get_G0_ks_2 import get_G0_ks_2

def calculate(database, model, names, powers, method):
    A = 1.0
    for j in range(len(names)):
        i = names[j]
        if i in list(database['mean'].keys()):
            A = A * np.abs(database['mean'][i])**(powers[j])
        elif i in list(database['ts_fit'][method].keys()):
            A = A *np.abs(database['ts_fit'][method][i])**(powers[j])
        elif i in list(model.keys()):
            A = A * np.abs(model[i][-1])**(powers[j])            
    return A

def calculate_err(database, model, names, powers, method):
    A = 0.0
    for j in range(len(names)):
        i = names[j]
        if i in list(database['mean'].keys()):
            A = A + np.abs(database['std'][i]) / np.abs(database['mean'][i])*np.abs(powers[j])
        elif i in list(database['ts_fit'][method].keys()):
            A = A + np.abs(database['ts_fit'][method][i+'_err']) / np.abs(database['ts_fit'][method][i])*np.abs(powers[j])
        elif i in list(model.keys()):
            A = A + np.abs(model[i+ '_err'][-1]) / np.abs(model[i][-1])*np.abs(powers[j])            
    return A

def plot_compare_ln(model, exp, params, names):
    cm2 = plt.cm.get_cmap('plasma')
    for i in names:
        if i in list(model.keys()):
            z = np.array(model[i])
        elif i in list(exp.keys()):
            z = np.array(exp[i])
        elif i in list(params.keys()):
            z = np.array(params[i])
        # X = np.array(exp['ln'])
        # Y = np.array(model['ln'])
        
        X = np.array(exp['ln'])/np.array(model['rhos']) 
        # Y = np.array(model['ln'])/(1+(-0.5*np.sign(np.array(params['Z_OSP']))+model['kx_corr'])**2)**(9/11)/model['rhos']
        Y = np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx_corr'])**2)**(9/11)/model['rhos']
        
        plt.figure()
        sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
        plt.colorbar(sc).ax.set_title(r'$' + i +'$')
        plt.plot([0,150],[0,150], '--k')
        plt.plot([0,150], [0,0.7*150], '--k')
        plt.plot([0,150], [0,1.3*150], '--k')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\lambda_{n}^{exp}$')
        plt.ylabel(r'$\lambda_{n}^{mod}$')
        plt.show()
        
def plot_compare_flux_ratio(model, exp, params, names):
    cm2 = plt.cm.get_cmap('plasma')
    for i in names:
        try:
            if i in list(model.keys()):
                z = np.array(model[i])
            elif i in list(exp.keys()):
                z = np.array(exp[i])
            elif i in list(params.keys()):
                z = np.array(params[i])
            # X = np.array(exp['ln'])
            # Y = np.array(model['ln'])
            
            X = params['H98']
            # Y = np.array(model['ln'])/(1+(-0.5*np.sign(np.array(params['Z_OSP']))+model['kx_corr'])**2)**(9/11)/model['rhos']
            Y = np.array(model['Gamma0'])/np.array(model['ratio'])/np.array(model['correc_shear'])/(np.array(model['D0'])/np.array(params['q95']))
            
            plt.figure()
            sc = plt.scatter(X, Y, c=z, vmin=np.nanmin(z), vmax=np.nanmax(z), s=35, cmap=cm2)
            plt.colorbar(sc).ax.set_title(r'$' + i +'$')
            plt.xlabel(r'$H98$')
            plt.ylabel(r'$Flux ratio$')
            plt.show()
        except: 
            []
            
path = '/fusion/projects/boundary/peretm/DIIID_analysis/GUI_fit/automated_fitting/saved_plateaus/L_mode_no_shift'
dir_list = os.listdir(path)

ind_rm = []
for i in range(len(dir_list)):
    if dir_list[i][0]!='s':
        ind_rm.append(i)
        
for i in range(len(ind_rm)):
    dir_list.remove(dir_list[ind_rm[-1-i]])
    

plt.close('all')

shot = []
model = {}
model['rhos'] = []
model['rhos_err'] = []
model['g'] = []
model['g_err'] = []
model['sig_N'] = []
model['sig_N_err'] = []
model['sig_PHI'] = []
model['sig_PHI_err'] = []
model['ln'] = []
model['ln_err'] = []
model['alphat'] = []
model['alphat_err'] = []
model['k00'] = []
model['k00_err'] = []
model['tau'] = []
model['tau_err'] = []
model['kx'] = []
model['kx_err'] = []
model['kx_corr'] = []
model['kx_corr_err'] = []
model['ks'] = []
model['ratio'] = []
model['Gamma0'] = []
model['alpha'] = []
model['q0'] = []
model['correc_shear'] = []
model['D0'] = []
model['chi0'] = []


method = 'linear log'

params = {}
params['shot'] = []
params['t0'] = []
params['t1'] = []

exp = {}
exp['ln'] = []
exp['ln_err'] = []
exp['lT'] = []
exp['lT_err'] = []
exp['ne_U'] = []
exp['ne_U_err'] = []
exp['Te_U'] = []
exp['Te_U_err'] = []
exp['H98'] = []


Z = 1
A = 2
qe = 1.602e-19
mp = 1.67e-27

rhos_names = ['Te_U', 'BT']
rhos_powers = [0.5, -1]

g_names = ['rhos', 'R0']
g_powers = [1, -1]
alphag = 0.9

sig_N_names = ['rhos', 'q95', 'R0']
sig_N_powers = [1.0, -1.0, -1.0]
alpha_sig_N = 1/math.pi

sig_PHI_names = ['rhos', 'q95', 'R0']
sig_PHI_powers = [1.0, -1.0, -1.0]
alpha_sig_PHI = 1/math.pi

alpha_ln = 3.9 
ln_names = ['g', 'sig_N', 'sig_PHI', 'rhos']
ln_powers = [3/11, -4/11, -2/11, 1]

alphat_names = ['ne_U', 'Te_U', 'q95', 'R0']
alphat_powers = [1, -2, 2, 1]
alpha_alphat = 30.0

k00_names = ['lpe_U', 'sig_PHI', 'g', 'rhos']
k00_powers = [1/4, 1/2, -1/4, -1/4]
alpha_k00 = (0.92/4.0)**(1/4)

tau_names = ['lpe_U', 'g', 'rhos']
tau_powers = [1/2, -1/2, -1/2]
alpha_tau = 0.43

kx_names = ['rhos', 'lTe_U', 'tau']
kx_powers = [2, -2, 1]
alpha_kx = 3
n_shot = len(dir_list)
# breakpoint()
# for i in range(len(dir_list)):
database = np.load(path + '/' + dir_list[0],allow_pickle=True).tolist()
for j in list(database['mean'].keys()):
    if j in ('eq', 'ts_data'):
        []
    else:
        params[j] = []
        params[j + '_err'] = []   
        

for i in range(n_shot):
    print(str(i+1)+'/'+ str(n_shot))
    database = np.load(path + '/' + dir_list[i],allow_pickle=True).tolist()
    if ((database['ts_fit'][method]['lne_U']==[]) or (database['ts_fit'][method]['lTe_U']==[]) or database['mean']['a']==np.nan):
        []   
    else:
        
        database['ts_fit'][method]['lpe_U'] = 1.0/(1.0/database['ts_fit'][method]['lne_U']+1.0/database['ts_fit'][method]['lTe_U'])
        database['ts_fit'][method]['lpe_U_err'] = database['ts_fit'][method]['lpe_U'] * (database['ts_fit'][method]['lne_U_err']/database['ts_fit'][method]['lne_U']/(1.0+database['ts_fit'][method]['lne_U']/database['ts_fit'][method]['lTe_U'])+database['ts_fit'][method]['lTe_U_err']/database['ts_fit'][method]['lTe_U']/(1.0+database['ts_fit'][method]['lTe_U']/database['ts_fit'][method]['lne_U']))
        if (database['ts_fit'][method]['lne_U_err']/database['ts_fit'][method]['lne_U']*100.0<=20):
            shot.append(int(dir_list[i][5:11]))
            print('shot: ' + dir_list[i][5:11])
    
            
            database['ts_fit'][method]['lpe_U'] = []
            database['ts_fit'][method]['lpe_U'] = database['ts_fit'][method]['lne_U']/(1.0+database['ts_fit'][method]['lne_U']/database['ts_fit'][method]['lTe_U'])
            database['ts_fit'][method]['lpe_U_err'] = []
            database['ts_fit'][method]['lpe_U_err'] = database['ts_fit'][method]['lpe_U']**2 * (database['ts_fit'][method]['lne_U_err']/database['ts_fit'][method]['lne_U']**2+database['ts_fit'][method]['lTe_U_err']/database['ts_fit'][method]['lTe_U']**2)
            
            model['rhos'].append(float(np.sqrt(A*mp/Z/qe)*calculate(database, model, rhos_names, rhos_powers, method)))
            model['g'].append(float(alphag * calculate(database, model, g_names, g_powers, method)))
            model['sig_N'].append(float(alpha_sig_N * calculate(database, model, sig_N_names, sig_N_powers, method)))
            model['sig_PHI'].append(float(alpha_sig_PHI * calculate(database, model, sig_PHI_names, sig_PHI_powers, method)))
            model['ln'].append(float(alpha_ln * calculate(database, model, ln_names, ln_powers, method)))
            model['alphat']. append(float(alpha_alphat * calculate(database, model, alphat_names, alphat_powers, method)))
            model['k00']. append(float(alpha_k00 * calculate(database, model, k00_names, k00_powers, method)))
            model['tau']. append(float(alpha_tau * calculate(database, model, tau_names, tau_powers, method)))
            
            params['shot'].append(int(dir_list[i][5:11]))
            params['t0'].append(database['t0'])
            params['t1'].append(database['t1'])
            for j in list(database['mean'].keys()):
                if j in ('eq', 'ts_data'):
                    []
                else:
                    params[j].append(database['mean'][j])
                    params[j + '_err'].append(database['std'][j])
                    
            if 'Prad_div' in database['mean']:
                []
            else:
                params['Prad_div'].append(np.nan)
                
            try:            
                # G0, ks = get_G0_ks(shot[-1], model['k00'][-1], database, path + '/' + dir_list[i])
                G0, ks = get_G0_ks_2(shot[-1], model['k00'][-1], database, path + '/' + dir_list[i])
                model['ks'].append(ks)
            except:
                G0 = alphag
                model['ks'].append(np.sign(database['mean']['Z_OSP'])*np.sign(database['mean']['BT'])*np.sign(database['mean']['BT'])/np.sign(database['mean']['Ip'])*0.5)
                
            model['g'][-1] = model['g'][-1] * G0/alphag
            model['rhos_err'].append(model['rhos'][-1]*calculate_err(database, model, rhos_names, rhos_powers, method))
            model['g_err'].append(float(model['g'][-1] * calculate_err(database, model, g_names, g_powers, method)))
            model['sig_N_err'].append(float(model['sig_N'][-1] * calculate_err(database, model, sig_N_names, sig_N_powers, method)))
            model['sig_PHI_err'].append(float(model['sig_PHI'][-1] * calculate_err(database, model, sig_PHI_names, sig_PHI_powers, method)))
            model['ln_err'].append(float(model['ln'][-1] * calculate_err(database, model, ln_names, ln_powers, method)))    
            model['alphat_err']. append(float(model['alphat'][-1] * calculate_err(database, model, alphat_names, alphat_powers, method)))
            model['k00_err']. append(float(model['k00'][-1] * calculate_err(database, model, k00_names, k00_powers, method)))
            model['tau_err']. append(float(model['tau'][-1] * calculate_err(database, model, tau_names, tau_powers, method)))
            model['g_err'][-1] = model['g_err'][-1] * G0/alphag
            model['kx']. append(float(alpha_kx * calculate(database, model, kx_names, kx_powers, method)))
            model['kx_err']. append(float(model['kx'][-1] * calculate_err(database, model, kx_names, kx_powers, method)))
                 
            
            
            exp['ln'].append(database['ts_fit'][method]['lne_U'])
            exp['ln_err'].append(database['ts_fit'][method]['lne_U_err'])
            
            exp['lT'].append(database['ts_fit'][method]['lTe_U'])
            exp['lT_err'].append(database['ts_fit'][method]['lTe_U_err'])
            
            exp['ne_U'].append(database['ts_fit'][method]['ne_U'])
            exp['ne_U_err'].append(database['ts_fit'][method]['ne_U_err'])
            
            exp['Te_U'].append(database['ts_fit'][method]['Te_U'])
            exp['Te_U_err'].append(database['ts_fit'][method]['Te_U_err'])       
            
            exp['H98'].append(database['mean']['H98'])
            
            
            ln = exp['ln'][-1]/model['rhos'][-1]
            lp = 1.0/(1.0/exp['ln'][-1]+1.0/exp['lT'][-1])/model['rhos'][-1]
            # model['ratio'].append(39*model['g'][-1]**(3/4)*model['sig_PHI'][-1]**(-1/2)/ln/lp**(3/4)/(1.0 + (model['ks'][-1]-0.43*lp**(-1/2)*ln**(-1)*model['g'][-1]**(-1/2))**2)**(9/4))
            model['ratio'].append((1.0 + (model['ks'][-1]-0.43*lp**(-1/2)*ln**(-1)*model['g'][-1]**(-1/2))**2)**(9/4))
            kx = -0.43*lp**(-1/2)*ln**(-1)*model['g'][-1]**(-1/2)
            model['Gamma0'].append(39*model['g'][-1]**(3/4)*model['sig_PHI'][-1]**(-1/2)/ln/lp**(3/4))
            model['q0'].append(39*model['g'][-1]**(3/4)*model['sig_PHI'][-1]**(-1/2)/lp**(7/4))
            model['alpha'].append(kx * (model['ks'][-1]+kx)/np.sqrt(1.0 + (kx + model['ks'][-1])**2))
            model['correc_shear'].append(1.0+1.13*model['alpha'][-1]-0.01*model['alpha'][-1]**2)
            model['D0'].append(database['mean']['q95']/ln)
            model['chi0'].append(database['mean']['q95']/lp)
            # breakpoint()

        
model['kx_corr'] = np.array(model['kx']) * np.array(model['k00'])*np.array(exp['lT'])/np.array(model['rhos'])/(1.0 + (np.array(model['k00'])*np.array(exp['lT'])/np.array(model['rhos']))**2)
alpha = np.array(model['k00'])*np.array(exp['lT'])/np.array(model['rhos'])
model['kx_corr_err'] = model['kx_corr']*(np.array(model['kx_err'])/np.array(model['kx'])+np.abs(1-alpha**2)/(1+alpha**2)*(np.array(model['k00_err'])/np.array(model['k00'])+np.array(exp['lT_err'])/np.array(exp['lT'])+np.array(model['rhos_err'])/np.array(model['rhos'])))

plt.figure()
plt.plot(np.array(exp['H98']), np.array(model['D0'])/np.array(params['q95']), 'o', label='Bohm particle flux')
plt.plot(np.array(exp['H98']), np.array(model['chi0'])/np.array(params['q95']), 'o', label='Bohm heat flux')
plt.plot(np.array(exp['H98']), np.array(model['Gamma0'])/np.array(model['ratio'])/np.array(model['correc_shear']), 'o', label='turbulent particle flux')
plt.plot(np.array(exp['H98']), np.array(model['q0'])/np.array(model['ratio'])/np.array(model['correc_shear']), 'o', label='turbulent heat flux')
plt.legend()
plt.title('with correc')
plt.xlabel('H98')
plt.ylabel('Normalised flux')

plt.figure()
plt.plot(np.array(exp['H98']), np.array(model['D0'])/np.array(params['q95']), 'o', label='Bohm particle flux')
plt.plot(np.array(exp['H98']), np.array(model['chi0'])/np.array(params['q95']), 'o', label='Bohm heat flux')
plt.plot(np.array(exp['H98']), np.array(model['Gamma0'])/np.array(model['ratio']), 'o', label='turbulent particle flux')
plt.plot(np.array(exp['H98']), np.array(model['q0'])/np.array(model['ratio']), 'o', label='turbulent heat flux')
plt.title('without correc')
plt.legend()
plt.xlabel('H98')
plt.ylabel('Normalised flux')


n = np.array(exp['ne_U'])*1e19
p =  np.array(exp['ne_U'])*1e19*qe * np.array(exp['Te_U'])
Cs = np.sqrt(Z*qe/A/mp*np.array(exp['Te_U']))

plt.figure()
plt.subplot(121)
plt.plot(np.array(exp['H98']), np.array(model['D0'])/np.array(params['q95'])*n*Cs, 'o', label='Bohm particle flux')
plt.plot(np.array(exp['H98']), np.array(model['Gamma0'])/np.array(model['ratio'])*n*Cs, 'o', label='turbulent particle flux')
plt.legend()
plt.xlabel('H98')
plt.ylabel('Flux')
plt.subplot(122)
plt.plot(np.array(exp['H98']), np.array(model['chi0'])/np.array(params['q95'])*p*Cs, 'o', label='Bohm heat flux')
plt.plot(np.array(exp['H98']), np.array(model['q0'])/np.array(model['ratio'])*p*Cs, 'o', label='turbulent heat flux')
plt.legend()
plt.xlabel('H98')
plt.ylabel('Flux')

plt.figure()
plt.subplot(121)
plt.plot(np.array(exp['H98']), np.array(model['D0'])/np.array(params['q95'])*n*Cs, 'o', label='Bohm particle flux')
plt.plot(np.array(exp['H98']), np.array(model['Gamma0'])/np.array(model['ratio'])*n*Cs/np.array(model['correc_shear']), 'o', label='turbulent particle flux')
plt.legend()
plt.xlabel('H98')
plt.ylabel('Flux')
plt.subplot(122)
plt.plot(np.array(exp['H98']), np.array(model['chi0'])/np.array(params['q95'])*p*Cs, 'o', label='Bohm heat flux')
plt.plot(np.array(exp['H98']), np.array(model['q0'])/np.array(model['ratio'])*p*Cs/np.array(model['correc_shear']), 'o', label='turbulent heat flux')
plt.legend()
plt.xlabel('H98')
plt.ylabel('Flux')


# plt.figure()
# plt.errorbar(np.array(exp['ln'])/np.array(model['rhos']), np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx_corr'])**2)**(9/11)/model['rhos'], np.array(model['ln'])/ np.array(model['rhos']) * (np.array(model['ln_err'])/np.array(model['ln'])+np.array(model['rhos_err'])/np.array(model['ln'])), np.array(exp['ln'])/ np.array(model['rhos']) * (np.array(exp['ln_err'])/np.array(exp['ln'])+np.array(model['rhos_err'])/np.array(model['ln'])), 'bo')
# plt.plot([0,150], [0,150], '--k')
# plt.plot([0,150], [0,0.7*150], '--k')
# plt.plot([0,150], [0,1.3*150], '--k')
# plt.yscale('log')
# plt.xscale('log')

# plt.axis('equal')

# plt.figure()
# plt.plot(np.array(model['alphat']), np.array(exp['ln'])/(np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx_corr'])**2)**(9/11)), 'ko')

# names_plot = ['Z_OSP', 'q95', 'sig_N', 'alphat', 'Pohm', 'nl', 'H98', 'Prad_div', 'Prad', 'ne_U', 'Te_U', 'kappa', 'tri_up', 'tri_bot', 'tri', 'ks', 'kx_corr', 'k00', 'ks', 'ln']
# plot_compare_flux_ratio(model, exp, params, names_plot)
# plot_compare_ln(model, exp, params, names_plot)

ln = np.array(exp['ln'])/np.array(model['rhos'])
lp = 1.0/(1.0/np.array(exp['ln'])+1.0/np.array(exp['lT']))/np.array(model['rhos'])
ind = np.where(np.array(params['H98'])<=1.0)[0]
ind_H = np.where(np.array(params['H98'])>1.0)[0]

ind = ind[np.where(np.array(exp['lT_err'])[ind]/np.array(exp['lT'])[ind]<=0.2)[0]]
ind = ind[np.where(np.array(exp['ln_err'])[ind]/np.array(exp['ln'])[ind]<=0.2)[0]]
ind = ind[np.where(np.array(exp['ne_U_err'])[ind]/np.array(exp['ne_U'])[ind]<=0.2)[0]]
ind = ind[np.where(np.array(exp['Te_U_err'])[ind]/np.array(exp['Te_U'])[ind]<=0.2)[0]]

ind_H = ind_H[np.where(np.array(exp['lT_err'])[ind_H]/np.array(exp['lT'])[ind_H]<=0.2)[0]]
ind_H = ind_H[np.where(np.array(exp['ln_err'])[ind_H]/np.array(exp['ln'])[ind_H]<=0.2)[0]]
ind_H = ind_H[np.where(np.array(exp['ne_U_err'])[ind_H]/np.array(exp['ne_U'])[ind_H]<=0.2)[0]]
ind_H = ind_H[np.where(np.array(exp['Te_U_err'])[ind_H]/np.array(exp['Te_U'])[ind_H]<=0.2)[0]]

plt.figure()
plt.subplot(121)
plt.title(r'\lambda_{n}')
plt.plot((np.array(model['rhos'])*ln)[ind]*1e3,(np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx'])**2)**(9/11)*2**(-6/11)*(lp/ln)**(-3/11))[ind]*1e3, 'o', label = 'ln L-mode')
plt.plot((np.array(model['rhos'])*ln)[ind_H]*1e3,(np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx'])**2)**(9/11)*2**(-6/11)*(lp/ln)**(-3/11))[ind_H]*1e3, 'o', label = 'ln H-mode')
plt.plot([0,90], [0,90], '--k', label ='1:1')
plt.xlabel('Experiment [mm]')
plt.ylabel('Model [mm]')
plt.legend()


plt.subplot(122)
plt.title(r'\lambda_{p}')
plt.plot((np.array(model['rhos'])*lp)[ind]*1e3,(np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx'])**2)**(9/11)*15**(-6/11))[ind]*1e3, 'o', label = 'lp L-mode')
plt.plot((np.array(model['rhos'])*lp)[ind_H]*1e3,(np.array(model['ln'])/(1+(-np.array(model['ks'])+model['kx'])**2)**(9/11)*15**(-6/11))[ind_H]*1e3, 'o', label = 'lp H-mode')
plt.plot([0,30], [0,30], '--k', label ='1:1')
plt.xlabel('Experiment [mm]')
plt.ylabel('Model [mm]')
plt.legend()