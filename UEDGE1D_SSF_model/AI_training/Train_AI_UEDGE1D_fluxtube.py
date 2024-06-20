#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:21:01 2023

@author: peretm
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
plt.close('all')
i_plot=0

ee = 1.602e-19
mi = 1.67e-27
A = 1
Z = 1


label = 'TeT'
scaler = MinMaxScaler()


database = np.load('/fusion/projects/boundary/peretm/AI_database/UEDGE1D_database_MP_fluxtube2.npy', allow_pickle=True).tolist()
taue = database['TeT']**(3/2)/database['neT']*2.33e10#/(sep['spar'][i_t1]-sep['spar'][i_x2])
database['Ldiv_T'] = (database['L_leg'])/1836/np.sqrt(ee/mi/A*database['TeT'])/taue
# database['Ldiv'] = database['Ldiv']*database['L_leg']

# database['alphat'] = (database['neU']/database['TeU']**2/1.03e16) * np.sqrt(2/A) * 6*6*3.1415*1.5/100
database['alphat'] = (database['neU']/database['TeU']**2)*3e-18 * np.sqrt(2/A) * 6*6*1.5

database['PgT'] = database['ngT']*ee*database['TgT']

database['Ldiv'] = database['Ldiv2']
# database['Ldiv'] = database['Ldiv_T']

test1 = (np.sqrt((np.array(database['TeX'])))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU'])))

# ind_filter = np.where(test1<1e-3)[0]

ind_filter = np.where(database['TeU'] <=0.5e3)[0]
ind_filter = ind_filter[np.where(database['fc'][ind_filter] >=0.03)[0]]
ind_filter = ind_filter[np.where(database['fc'][ind_filter] <=0.035)[0]]
# ind_filter = ind_filter[np.where(database['L_leg'][ind_filter] >=2)[0]]
#ind_filter = ind_filter[np.where((database['pcore']/database['PgT']/1e3)[ind_filter] <5)[0]]
# ind_filter = ind_filter[np.where(database['neU'][ind_filter] <=3e19)[0]]

log_n = np.log(np.array(database['ncore']))[ind_filter]/np.log(1e22)#/np.max(np.log(np.array(database['ncore'])))
log_p = np.log(np.array(database['pcore']))[ind_filter]/np.log(1e9)#np.max(np.log(np.array(database['pcore'])))
L_leg = np.array(database['fc'])[ind_filter]/0.15#/np.max(np.array(database['L_leg']))


Nsim = len(log_n)

i_train = random.sample(range(0, Nsim), int(np.floor(Nsim*0.95)))
i_test = [ x for x in np.arange(0,Nsim,1) if x not in i_train ]

i_train = np.array(i_train)
i_test = np.array(i_test)
#i_train = np.arange(0,int(np.floor(Nsim*0.8)),1)
#i_test = np.arange(int(np.floor(Nsim*0.8)), Nsim,1)
train = np.squeeze(np.dstack((log_n,log_p, L_leg)))[i_train,:]
test = np.squeeze(np.dstack((log_n,log_p, L_leg)))[i_test,:]

# train = scaler.fit_transform(train)
# test = scaler.fit_transform(test)
# L_leg = scaler.fit_transform(L_leg)
# train = np.zeros((len(database['i_u']),1,3))

# results_train = (np.array(database['TeT'])/np.array(database['TeU']))[i_train]
# results_test = (np.array(database['TeT'])/np.array(database['TeU']))[i_test]

# results = np.squeeze(np.dstack((np.array(database['TeT'])/np.array(database['TeU']),(np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU']))),1.0/(1.0+np.array(database['Ldiv'])/0.51))))[ind_filter]

# results = np.squeeze(np.dstack(((np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU']))),np.sqrt(np.array(database['TeU'])/np.array(database['TeT']))/(1.0+np.array(database['Ldiv'])/0.51))))[ind_filter]

# results = scaler.fit_transform(np.log(results))
# results = np.squeeze(np.dstack(((np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU']))),np.array(database['neT'])/np.array(database['neX'])*np.sqrt(np.array(database['TeU'])/np.array(database['TeT']))/(1.0+np.array(database['Ldiv'])/0.51))))[ind_filter]

# results = np.squeeze(np.dstack(((log_n + log_p + L_leg + 0.5 * log_n * log_p * L_leg)/20, (log_n + log_p + L_leg)/20*(np.exp(-log_p-log_n)))))

if label=='TeU':
    results = np.squeeze(np.dstack((np.log(database['TeU'])[ind_filter]/np.log(200)))) 
if label == 'TeT':
    results = np.squeeze(np.dstack((np.log(database['TeT']*100)[ind_filter]/np.log(30000))))                                                                     
if label == 'MX':
    results = np.squeeze(np.dstack((np.log(database['MX']*100)[ind_filter]/np.log(1000))))
                      
# results = scaler.fit_transform(results)
# results = (results - np.mean(results))/(np.max(results)-np.min(results))
ln_correc = ((database['neX'] / database['neT'] *np.sqrt(np.array(database['TeT'])/np.array(database['TeU']))*(1.0+np.array(database['Ldiv'])/0.51))**(2/11)*((np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU']))))**(-4/11))[ind_filter]

ind = ind_filter#np.where(database['fc']==0.0)[0]
plt.figure()
plt.semilogx(database['TeT'][ind],12*ln_correc,'o')
plt.figure()
plt.semilogx(database['Ldiv'][ind],12*ln_correc,'o')
plt.figure()
plt.semilogx(database['Ldiv_T'][ind],12*ln_correc,'o')
plt.figure()
plt.semilogx(database['Ldiv'][ind],database['Ldiv_T'][ind],'o')
plt.figure()
plt.semilogx(database['TeT'][ind],database['Ldiv'][ind],'o')


Ldiv = [9e-2, 3.5e-1, 1.8, 5.5, 5, 6, 6.5, 8 ]
ln = [16, 19, 23, 30, 34, 34, 40, 52]
alphat = np.linspace(np.min(database['alphat'][ind_filter]),np.max(database['alphat'][ind_filter]),1000)
    

    
cm = plt.cm.get_cmap('hot')
z = (database['Ldiv'])[ind_filter]
plt.style.use('dark_background')
plt.figure()
sc = plt.scatter(database['alphat'][ind_filter],2.5*ln_correc, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm)
plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
plt.plot(Ldiv,ln, 'rs')
plt.colorbar(sc).ax.set_title(r'$\Lambda_{div}$')
plt.xlabel(r'$\alpha_{t}$')
plt.ylabel(r'$\lambda_{n}$ [mm]')
plt.xlim((1e-3, 5))
plt.ylim((0,100))
# plt.xscale('log')
plt.show()

results_train = results[i_train]
results_test = results[i_test]

# results_train = scaler.fit_transform(results_train)
# results_test = scaler.fit_transform(results_test)
#np.max(np.array(database['TeT']))
# for i in range(len(database['i_u'])):
#     train[i,:,:] = np.log(np.array(database['ncore'][i]))/np.max(np.log(np.array(database['ncore']))), np.log(np.array(database['pcore'][i]))/np.max(np.log(np.array(database['pcore']))), np.array(database['L_leg'][i])/np.max(np.array(database['L_leg']))
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.01,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)
opt = keras.optimizers.Adam(learning_rate=0.0645)#learning_rate=0.0645

    
#model = keras.Sequential([keras.layers.Input(shape=(3,)), keras.layers.Dense(81,activation='relu'), keras.layers.Dense(9,activation='relu'), keras.layers.Dense(3,activation='linear')])
model = keras.Sequential([keras.layers.Input(shape=(3,)), keras.layers.Dense(32,activation='sigmoid'), keras.layers.Dense(32,activation='sigmoid'), keras.layers.Dense(32,activation='sigmoid'), keras.layers.Dense(1,activation='linear')])

model.compile(optimizer=opt, loss='mean_absolute_percentage_error')

model.fit(train, results_train, epochs=300)

# test_acc = np.sqrt(np.sum((model.predict(test) - results_test)**2))/np.sum(results_test)

loss = model.evaluate(test, results_test)
print('The loss is : ', loss)

# min0 = np.nanmin(database['TeU'][ind_filter])
# max0 = np.nanmax(database['TeU'][ind_filter])
# plt.figure()
# plt.plot(np.exp(model.predict(test)*np.log(500)), np.exp(results_test*np.log(500)), 'o')
# plt.plot([min0,max0],[min0,max0],'--k')
# plt.xlim((min0,max0))
# plt.ylim((min0, max0))
# plt.axis('square')
# plt.xlabel(r'$T_{e}^{U}$ NN [eV]')
# plt.ylabel(r'$T_{e}^{U}$ sims [eV]')
plt.figure()
plt.plot(results_test, model.predict(test), 'o')
plt.plot([0,1], [0,1], '--k')
plt.xlabel(r'$ln(' + label + ')$ NN [normalized]')
plt.ylabel(r'$ln(' + label + ')$ sims [normalized]')



# plt.plot([np.min(model.predict(test)[:]), np.max(model.predict(test)[:])],[np.min(model.predict(test)[:]), np.max(model.predict(test)[:])], '--k')
    
plt.figure()
plt.hist(results_train[:], bins=200)

# plt.figure()
# plt.hist(results_train[:,1], bins=200)   

plt.figure()
plt.plot(np.abs(model.predict(test) / results_test -1.0)*100,'o')


# plt.figure()
# plt.plot(results_train[:,0] , results_train[:,1],'o')

cm2 = plt.cm.get_cmap('RdYlBu')
z = (database['pcore'][ind_filter[i_train]])
plt.figure()
sc = plt.scatter(results_train[:,0],results_train[:,1], c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
plt.colorbar(sc).ax.set_title(r'$pcore$')
plt.xlabel(r'$results[0]$')
plt.ylabel(r'$results[1]$')

plt.show()

cm2 = plt.cm.get_cmap('RdYlBu')
z = (database['ncore'][ind_filter[i_train]])
plt.figure()
sc = plt.scatter(results_train[:,0],results_train[:,1], c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
plt.colorbar(sc).ax.set_title(r'$ncore$')
plt.xlabel(r'$results[0]$')
plt.ylabel(r'$results[1]$')

plt.show()    

cm2 = plt.cm.get_cmap('RdYlBu')
z = (database['ncore'])
plt.figure()
sc = plt.scatter(database['pcore'],database['TeU']*(database['qe0']+database['qi0']), c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
plt.colorbar(sc).ax.set_title(r'$ncore$')
plt.xlabel(r'$pcore$')
plt.ylabel(r'$q0$')


cm2 = plt.cm.get_cmap('RdYlBu')
z = (database['L_leg'])
plt.figure()
sc = plt.scatter(database['pcore'],database['TeU']*(database['qe0']+database['qi0']), c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
plt.colorbar(sc).ax.set_title(r'$L_{leg}$')
plt.xlabel(r'$pcore$')
plt.ylabel(r'$q0$')

    

    
if i_plot==1:
    

    
    plt.figure()
    plt.semilogx(database['Ldiv'][ind_filter],ln_correc, 'bo')
    
    
    alphat = np.linspace(np.min(database['alphat'][ind_filter]),np.max(database['alphat'][ind_filter]),100)
    plt.figure()
    plt.semilogx(database['alphat'][ind_filter],ln_correc/1.5, 'bo')
    plt.semilogx(alphat,1.0 + 10.4 * alphat**(2.5),'--r')
    plt.xlim((0.0000001, 1))
    plt.ylim((0,100))
    #Caralerro JET pts [Caralerro et al., PRL 2015]
    
    Ldiv = [9e-2, 3.5e-1, 1.8, 5.5, 5, 6, 6.5, 8 ]
    ln = [16, 19, 23, 30, 34, 34, 40, 52]
    
    plt.figure()
    plt.semilogx(database['Ldiv'][ind_filter],ln_correc*10, 'bo')
    plt.semilogx(Ldiv, ln, 'rs')
    plt.xlim((1e-3, 10))
    plt.ylim((0,100))
    plt.xlabel(r'$\Lambda_{div}$')
    plt.ylabel(r'$\lambda_{n} [mm]$')
    
    cm = plt.cm.get_cmap('RdYlBu')
    
    z = database['L_leg'][ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, vmin=np.min(z), vmax=np.max(z), s=35, cmap=cm)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$L_{leg} [m]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    z = database['taue'][ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, vmin=np.min(z), vmax=np.max(z)/10000.0, s=35, cmap=cm)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$\tau_{e} [s]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    z = database['TeT'][ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=10), s=35, cmap=cm)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$T_{e}^{T} [eV]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    z = (database['taue']/taue)[ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, vmin=np.min(z), vmax=np.max(z)/1000, s=35, cmap=cm)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r' Norm. $\tau_{e} [s]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    z = (database['Prad_div'])[ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$P_{rad,div} [a.u.]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    cm2 = plt.cm.get_cmap('RdYlBu')
    z = (database['MX'])[ind_filter]
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$M^{X}$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    cm2 = plt.cm.get_cmap('RdYlBu')
    z = (results[:,0])
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'Norm. $\sigma_{\parallel}^{N}$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
    
    cm2 = plt.cm.get_cmap('RdYlBu')
    z = (results[:,1])
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'Norm. $\sigma_{\parallel}^{\Phi}$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()

    cm2 = plt.cm.get_cmap('RdYlBu')
    z = (database['PgT'][ind_filter])
    plt.figure()
    sc = plt.scatter(database['alphat'][ind_filter],3*ln_correc/1.5, c=z, norm= colors.LogNorm(vmin=np.min(z), vmax=np.max(z)), s=35, cmap=cm2)
    # plt.xscale('log')
    plt.plot(alphat,3*(1.0 + 10.4 * alphat**(2.5)),'--b')
    plt.colorbar(sc).ax.set_title(r'$P_{n}^{T}[Pa]$')
    plt.xlabel(r'$\alpha_{t}$')
    plt.ylabel(r'$\lambda_{n}$ [mm]')
    plt.xlim((0, 1))
    plt.ylim((0,100))
    plt.show()
# test = train[100:106,:]
# prediction = np.squeeze(model.predict(test))

# print('Real value is : ', results[100:106] *np.array(database['TeU'][100:106]))# np.max(np.array(database['TeT'])))
# print('Predicted value is : ', prediction * np.array(database['TeU'][100:106]))# np.max(np.array(database['TeT'])))
# print('The asolute error is : ', 100.0*(prediction/results[100:106]-1))
