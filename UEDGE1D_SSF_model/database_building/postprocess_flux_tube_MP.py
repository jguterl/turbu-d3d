#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:12:22 2023

@author: peretm
"""


import os
import numpy as np
from scipy import integrate

# sim_name = 'slab_1D_scan_test_array'
# sim_name = 'slab_1D_scan_fixed_resolution'
# sim_name = 'slab_1D_scan_4_SSFmodel'
sim_name = '1D_flux_tube_test2'
data = np.load('/fusion/projects/boundary/peretm/simulations/{}/log.npy'.format(sim_name), allow_pickle=True).tolist()

casename0 = data['sims'][0]['casename'][0:-1]

#data_test = np.load('/fusion/projects/boundary/peretm/simulations/{}/log.npy'.format(sim_name), allow_pickle=True).tolist()

def get_data(file):
    return np.load(file, allow_pickle=True).tolist()
def postprocess(data):

    # i=0
    for sim in data['sims'].values():
       # i = i + 1
       # if i==4:
       #     break
       print(sim['casename'])
       directory = data['directory']
       path =  directory + '/SaveDir/{}'.format(sim['casename'])                                
       try:
           sim['postprocess'] = {"power":get_data(os.path.join(path,'power.npy')),"state":get_data(os.path.join(path,'final_state.npy')) }                                                                                              
           qi_target = sim['postprocess']['power'][3][0]['bbb.feix'][-2,1]
           qe_target = sim['postprocess']['power'][3][0]['bbb.feex'][-2,1]
           te_target = sim['postprocess']['state'][3][0]['bbb.te'][-2,1]
           qe_up = sim['postprocess']['power'][3][0]['bbb.feex'][1,1]
           qi_up = sim['postprocess']['power'][3][0]['bbb.feix'][1,1]
           sim['postprocess']['q_target'] = qi_target + qe_target
           sim['postprocess']['qe_0'] = qe_up
           sim['postprocess']['qi_0'] = qi_up
           sim['postprocess']['q_0'] = qi_up + qe_up
           sim['postprocess']['te_target'] = te_target/1.6e-19
           print("Loaded data from:",sim['casename'])
       except Exception as e:
           print(e)
postprocess(data)

#postprocess(data_test)
# np.save('data.npy',data)
# np.save('data_test.npy',data_test)
####%%

valid_sims = []
for s in data['sims'].values():
    if s.get('postprocess') is not None:
            valid_sims.append(s)

data = valid_sims     
del valid_sims     
# valid_test_sims = []
# for s in data_test['sims'].values():
#     if s.get('postprocess') is not None:
#             valid_test_sims.append(s)
            
# for s in  valid_sims:
#         s['uq']= {'q_target':s['postprocess']['q_target'],'te_target':s['postprocess']['te_target']}
# for s in  valid_test_sims:
#         s['uq']= {'q_target':s['postprocess']['q_target'],'te_target':s['postprocess']['te_target']}
# #%%    
def set_scale(y,scale=None):
    if scale is None or scale =='linear':
        return y
    elif scale=='log':
        return np.log(y)
    else:
        raise ValueError("unknwon scale:", scale)
        
def dump_uq_data(valid_sims, valid_test_sims=None, uq_directory='uq', y_scale=None, sim_name='data'):
    if not os.path.exists(uq_directory):
        os.mkdir(uq_directory)

    ytrain_filepath = os.path.join(uq_directory,'ytrain.dat')
    ptrain_filepath = os.path.join(uq_directory,'ptrain.dat') 
    qtrain_filepath = os.path.join(uq_directory,'qtrain.dat')

    nvalidsims = len(valid_sims)
    # number of param
    kw = list(data['sims'][0]['params'].keys())
    if y_scale is None:
        y_scale = dict((k,'linear') for k in data['sims'][0]['uq'].keys())
    elif type(y_scale)==str:
        y_scale = dict((k,y_scale) for k in data['sims'][0]['uq'].keys())
    print('y_scale =', y_scale)
    nparam = len(kw)
    
    np.save('param_name.npy',np.array(kw))
    # = sum([1 for s in sims.values() if s['out'] != 'fail'])
    ntot = 0
    noutputs = len(list(data['sims'][0]['uq'].keys()))
    ntot = len(valid_sims) 
    print("ntrain:",ntot, 'noutputs:', noutputs)
    array_ptrain = np.zeros((ntot,nparam))
    array_qtrain = np.zeros((ntot,nparam))
    array_ytrain = np.zeros((ntot, noutputs))
    
    params = dict((k,[]) for k in kw)
    for s in valid_sims:
        [params[k].append(s['params'][k]) for k in kw]
    param_val = dict((k, np.unique(v)) for k,v in params.items())
    param_min = dict((k,np.min(v)) for k,v in param_val.items())
    param_max = dict((k,np.max(v)) for k,v in param_val.items())



    kk=0
    for i,s in enumerate(valid_sims):
        ptrain = s['params']
        qtrain = dict((k,2*(v-param_min[k])/(param_max[k]-param_min[k])-1 ) for k,v in ptrain.items())
        ytrain = s['uq']
        array_qtrain[kk,0:nparam] = [q for q in qtrain.values()]
        array_ptrain[kk,0:nparam] = [p for p in ptrain.values()]
        array_ytrain[kk,0:noutputs] = [set_scale(y,y_scale.get(k)) for k,y in ytrain.items()] 
        kk += 1
    np.savetxt(ytrain_filepath,array_ytrain)
    np.savetxt(qtrain_filepath,array_qtrain)
    np.savetxt(ptrain_filepath,array_ptrain)
    pdata = {'ptrain':array_ptrain, 'ytrain':array_ytrain,'qtrain':array_qtrain}


  
    if valid_test_sims is not None: 
            yval_filepath = os.path.join(uq_directory,'yval.dat')
            pval_filepath = os.path.join(uq_directory,'pval.dat') 
            qval_filepath = os.path.join(uq_directory,'qval.dat')
            nvals = len(valid_test_sims)
            array_pval = np.zeros((nvals,nparam))
            array_qval = np.zeros((nvals,nparam))
            array_yval = np.zeros((nvals, noutputs))
            ntot = len(valid_test_sims) 
            print("ntest:",nvals, 'noutputs:', noutputs)
            kk=0
            for i,s in enumerate(valid_test_sims):
                pval = s['params']
                qval = dict((k,2*(v-param_min[k])/(param_max[k]-param_min[k])-1 ) for k,v in pval.items())
                yval = s['uq']
                array_qval[kk,0:nparam] = [q for q in qval.values()]
                array_pval[kk,0:nparam] = [p for p in pval.values()]
                array_yval[kk,0:nparam] = [set_scale(y,y_scale.get(k)) for k,y in yval.items()] 
                kk += 1
            pdata['yval'] = array_yval
            pdata['qval'] = array_qval
            pdata['pval'] = array_pval
            pdata['y_scale'] = y_scale
    np.savetxt(yval_filepath,array_yval)
    np.savetxt(qval_filepath,array_qval)
    np.savetxt(pval_filepath,array_pval)
    
    np.save(os.path.join(uq_directory,'postprocess_{}.npy'.format(sim_name)),pdata)

#dump_uq_data(valid_sims,valid_test_sims= valid_test_sims, y_scale="log", sim_name=sim_name)
#%%
# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots()
# for s in data['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['ncore'],s['params']['pcore'], color='blue', marker='o')
#     else:
#         ax.scatter(s['params']['ncore'],s['params']['pcore'], color='red', marker='o')
# for s in data_test['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['ncore'],s['params']['pcore'], color='green', marker='s')
#     else:
#         ax.scatter(s['params']['ncore'],s['params']['pcore'], color='purple', marker='s')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title(sim_name)
# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots()
# for s in data['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='blue', marker='o')
#     #else:
#         #ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='red', marker='o')
# for s in data_test['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='green', marker='s')
#     #else:
#         #ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='purple', marker='s')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title(sim_name)

# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots()
# for s in data['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['pcore'],s['uq']['te_target'], color='blue', marker='o')
#     #else:
#         #ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='red', marker='o')
# for s in data_test['sims'].values():
#     if s.get('postprocess') is not None: 
#         ax.scatter(s['params']['pcore'],s['uq']['te_target'], color='green', marker='s')
#     #else:
#         #ax.scatter(s['params']['pcore'],s['uq']['q_target'], color='purple', marker='s')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_title(sim_name)

#%%
def plot_profile(sim):
    if type(sim) != list: sim=[sim]
    fig, ax = plt.subplots(3,2)
    for s in sim:
        try:
            state = s['postprocess']['state'][3][0]
            power = s['postprocess']['power'][3][0]
            z = state['com.zm'][:,1,0]
            te = state['bbb.te'][:,1]
            ti = state['bbb.te'][:,1]
            ni = state['bbb.ni'][:,1,0]
            ng = state['bbb.ng'][:,1,0]
            up = state['bbb.up'][:,1,0]
            prad = power['bbb.pradcff'][:,1]
            ax[0,0].plot(z, te/1.602e-19)
            ax[0,1].plot(z, ni)
            ax[1,0].plot(z, ng)
            ax[1,1].plot(z, up)
            ax[2,0].plot(z,ti/1.602e-19)
            ax[2,1].plot(z, prad, label=s['params'])
            #ax[2,1].legend()
        except:
            pass

database = {}        
Nsim = len(data)

database['i_u'] = [0] * Nsim
database['i_t'] = [-1] * Nsim
database['i_x'] = [18] * Nsim
database['neU'] = np.zeros(Nsim)
database['neT'] = np.zeros(Nsim)
database['neX'] = np.zeros(Nsim)
database['MU'] = np.zeros(Nsim)
database['MX'] = np.zeros(Nsim)
database['MT'] = np.zeros(Nsim)
database['CsU'] = np.zeros(Nsim)
database['CsX'] = np.zeros(Nsim)
database['CsT'] = np.zeros(Nsim)
database['TeU'] = np.zeros(Nsim)
database['TeX'] = np.zeros(Nsim)
database['TeT'] = np.zeros(Nsim)
database['TiU'] = np.zeros(Nsim)
database['TiX'] = np.zeros(Nsim)
database['TiT'] = np.zeros(Nsim)
database['TgU'] = np.zeros(Nsim)
database['TgX'] = np.zeros(Nsim)
database['TgT'] = np.zeros(Nsim)
database['ngU'] = np.zeros(Nsim)
database['ngX'] = np.zeros(Nsim)
database['ngT'] = np.zeros(Nsim)
database['vU'] = np.zeros(Nsim)
database['vX'] = np.zeros(Nsim)
database['vT'] = np.zeros(Nsim)
database['ncore'] = np.zeros(Nsim)
database['fc'] = np.zeros(Nsim)
database['pcore'] = np.zeros(Nsim)
database['L_leg'] = np.zeros(Nsim)
database['Prad_div'] = np.zeros(Nsim)
database['taue'] = np.zeros(Nsim)
database['nue'] = np.zeros(Nsim)
database['Ldiv'] = np.zeros(Nsim)
database['Ldiv2'] = np.zeros(Nsim)
database['sim_num'] = (np.zeros(Nsim)).astype(int)
database['qe0'] = np.zeros(Nsim)
database['qi0'] = np.zeros(Nsim)
ee = 1.602e-19
mi = 1.67e-27
Cs0 = np.sqrt(ee/mi)
A = 1
for k in range(len(data)):
    # database['i_x'][k] = np.argmin(np.abs(data[k]['postprocess']['state'][3][0]['com.zm'][:,1,0]-zxpt))
    database['neU'][k] = data[k]['postprocess']['state'][3][0]['bbb.ni'][database['i_u'][k],1,0]
    database['neX'][k] = data[k]['postprocess']['state'][3][0]['bbb.ni'][database['i_x'][k],1,0]
    database['neT'][k] = data[k]['postprocess']['state'][3][0]['bbb.ni'][database['i_t'][k],1,0]
    database['ngU'][k] = data[k]['postprocess']['state'][3][0]['bbb.ng'][database['i_u'][k],1,0]
    database['ngX'][k] = data[k]['postprocess']['state'][3][0]['bbb.ng'][database['i_x'][k],1,0]
    database['ngT'][k] = data[k]['postprocess']['state'][3][0]['bbb.ng'][database['i_t'][k],1,0]
    database['TeU'][k] = data[k]['postprocess']['state'][3][0]['bbb.te'][database['i_u'][k],1]/ee
    database['TeX'][k] = data[k]['postprocess']['state'][3][0]['bbb.te'][database['i_x'][k],1]/ee
    database['TeT'][k] = data[k]['postprocess']['state'][3][0]['bbb.te'][database['i_t'][k],1]/ee
    database['TiU'][k] = data[k]['postprocess']['state'][3][0]['bbb.ti'][database['i_u'][k],1]/ee
    database['TiX'][k] = data[k]['postprocess']['state'][3][0]['bbb.ti'][database['i_x'][k],1]/ee
    database['TiT'][k] = data[k]['postprocess']['state'][3][0]['bbb.ti'][database['i_t'][k],1]/ee
    database['TgU'][k] = data[k]['postprocess']['state'][3][0]['bbb.tg'][database['i_u'][k],1,0]/ee
    database['TgX'][k] = data[k]['postprocess']['state'][3][0]['bbb.tg'][database['i_x'][k],1,0]/ee
    database['TgT'][k] = data[k]['postprocess']['state'][3][0]['bbb.tg'][database['i_t'][k],1,0]/ee
    database['vU'][k] = data[k]['postprocess']['state'][3][0]['bbb.up'][database['i_u'][k],1,0]
    database['vX'][k] = data[k]['postprocess']['state'][3][0]['bbb.up'][database['i_x'][k],1,0]
    database['vT'][k] = data[k]['postprocess']['state'][3][0]['bbb.up'][database['i_t'][k],1,0]
    database['CsU'][k] = Cs0 * np.sqrt(database['TeU'][k]+database['TiU'][k])
    database['CsX'][k] = Cs0 * np.sqrt(database['TeX'][k]+database['TiT'][k])
    database['CsT'][k] = Cs0 * np.sqrt(database['TeT'][k]+database['TiT'][k])   
    database['MX'][k] = database['vX'][k]/database['CsX'][k]
    database['MT'][k] = database['vT'][k]/database['CsT'][k]
    database['ncore'][k] = data[k]['params']['ncore']
    database['pcore'][k] = data[k]['params']['pcore']
    database['fc'][k] = data[k]['params']['fc']#/ee
    te = data[k]['postprocess']['state'][3][0]['bbb.te'][database['i_x'][k]:database['i_t'][k],1]/ee
    ne = data[k]['postprocess']['state'][3][0]['bbb.ni'][database['i_x'][k]:database['i_t'][k],1,0]
    z = data[k]['postprocess']['state'][3][0]['com.zm'][-len(ne)-1:database['i_t'][k], 1,0]
    r = data[k]['postprocess']['state'][3][0]['com.rm'][-len(ne)-1:database['i_t'][k], 1,0]
    Lpar = np.append(0,np.cumsum(np.sqrt(np.diff(r)**2+np.diff(z)**2)))
    database['L_leg'][k] = Lpar[-1]
    database['Prad_div'][k] = integrate.simps(data[k]['postprocess']['power'][3][0]['bbb.prad'][database['i_x'][k]:database['i_t'][k],1],Lpar)
    # Lpar = data[k]['postprocess']['state'][3][0]['com.zm'][database['i_x'][k]:database['i_t'][k],1,0]
    database['taue'][k] = integrate.simps(te**(3/2)/ne, Lpar)*2.33e10#/(sep['spar'][i_t1]-sep['spar'][i_x2])
    database['nue'][k] = integrate.simps(te**(-3/2)*ne, Lpar)/2.33e10/database['L_leg'][k] #/(sep['spar'][i_t1]-sep['spar'][i_x2])
    database['Ldiv'][k] = (database['L_leg'][k])/1836/A/np.sqrt(ee/mi/A*database['TeT'][k])/database['taue'][k] 
    database['Ldiv2'][k] = (database['L_leg'][k])/1836/A/np.sqrt(ee/mi/A*database['TeT'][k])*database['nue'][k] 
    database['sim_num'][k] = int(data[k]['casename'].replace(casename0, ''))
    database['qe0'][k] = data[k]['postprocess']['qe_0']
    database['qi0'][k] = data[k]['postprocess']['qi_0']
np.save('/fusion/projects/boundary/peretm/AI_database/UEDGE1D_database_MP_fluxtube2.npy', database, allow_pickle=True)
#plot_profile(list(data['sims'].values()))
#sim = list(data['sims'].values())
#state = sim[0]['postprocess']['state'][3][0]
#power  = sim[0]['postprocess']['power'][3][0]
#plt.plot(state['com.zm'][:,1,0], state['bbb.te'][:,1])
#plt.figure();plt.plot(state['com.zm'][:,1,0], power['bbb.feix'][:,1]+power['bbb.feex'][:,1])
