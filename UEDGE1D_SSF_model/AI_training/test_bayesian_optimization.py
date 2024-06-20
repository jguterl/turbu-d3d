#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:12:59 2023

@author: peretm
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from bayes_opt import BayesianOptimization
import time


def training(N_lay, N_node, i_act, rate):
    database = np.load('/fusion/projects/boundary/peretm/AI_database/UEDGE1D_database_MP_2.npy', allow_pickle=True).tolist()
    database['Ldiv'] = database['Ldiv2']
    
    ee = 1.602e-19
    mi = 1.67e-27
    A = 1
    Z = 1
    alpha = (ee)**(3/2) * 2.61**(4/7) / 2000**(3/7) *Z**(1/2)/A**(1/2)/mi**(1/2)

# ind_filter = np.where(test1<1e-3)[0]
    param = alpha * database['ncore']/database['pcore']**(4/7)*database['L_leg']**(3/7)
    ind_filter = np.where(database['TeU'] <=0.5e3)[0]
    ind_filter = ind_filter[np.where(database['L_leg'][ind_filter] >=2)[0]]
    # ind_filter = ind_filter[np.where(param[ind_filter] >=6)[0]]
    # ind_filter = ind_filter[np.where(param[ind_filter] >=15)[0]]

    test1 = (np.sqrt((np.array(database['TeX'])))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU'])))
    test2 = np.sqrt(np.array(database['TeU'])/np.array(database['TeT']))*np.array(database['neT'])/np.array(database['neX'])/(1.0+np.array(database['Ldiv'])/0.51)


    log_n = np.log(np.array(database['ncore']))[ind_filter]/np.log(3e21)#/np.max(np.log(np.array(database['ncore'])))
    log_p = np.log(np.array(database['pcore']))[ind_filter]/np.log(3e8)#np.max(np.log(np.array(database['pcore'])))
    L_leg = np.array(database['L_leg'])[ind_filter]/300#/np.max(np.array(database['L_leg']))

    Nsim = len(log_n)

    i_train = random.sample(range(0, Nsim), int(np.floor(Nsim*0.95)))
    i_test = [ x for x in np.arange(0,Nsim,1) if x not in i_train ]

    i_train = np.array(i_train)
    i_test = np.array(i_test)

    train = np.squeeze(np.dstack((log_n,log_p, L_leg)))[i_train]
    test = np.squeeze(np.dstack((log_n,log_p, L_leg)))[i_test]

    # results = np.log(database['TeU'])[ind_filter]/np.log(500) #test1[ind_filter] # np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU'])))[ind_filter]
    results = np.squeeze(np.dstack((np.log(database['TeT']*1e5)[ind_filter]/np.log(1e8))))                                                                      

    # results = (np.log(test1)[ind_filter]/np.log(1e-5) -0.1 )* 0.8
    # ln_correc = (2**(-6/11) * (database['neX'] / database['neT'] *np.sqrt(np.array(database['TeT'])/np.array(database['TeU']))*(1.0+np.array(database['Ldiv'])/0.51))**(2/11)*((np.sqrt(np.array(database['TeX']))*np.array(database['MX'])*np.array(database['neX']))/(np.array(database['neU'])*np.sqrt(np.array(database['TeU']))))**(-4/11))[ind_filter]


    results_train = results[i_train]
    results_test = results[i_test]
    
    act = ['sigmoid', 'linear', 'elu', 'relu']
    opt = keras.optimizers.Adam(learning_rate=rate)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(3,)))
    for i in range(int(N_lay)):
        model.add(keras.layers.Dense(int(N_node),activation=act[int(i_act)]))
        
    model.add(keras.layers.Dense(1,activation='linear'))

    model.compile(optimizer=opt, loss='mean_absolute_percentage_error')

    model.fit(train, results_train, epochs=1000, verbose=0)

    test_acc = np.sqrt(np.sum((model.predict(test) - results_test)**2))/np.sum(results_test)

    loss2 = np.array(model.history.history['loss'])[-1]
    # loss2 = np.sum((model.predict(test)-results_test)**2)/np.sum(results_test**2)
    plt.close('all')
    plt.figure()
    plt.plot(model.predict(test), results_test, 'o')
    plt.plot([0,1],[0,1],'--k')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.axis('square')
    plt.xlabel(r'$T_{e}^{U}$ NN ')
    plt.ylabel(r'$T_{e}^{U}$ sims ')
    plt.pause(1)
    return loss2
# plt.figure()
# plt.plot(loss2,'bo')


# loss = model.evaluate(test, results_test)



# model = keras.Sequential(name="my_sequential")
# model.add(layers.Dense(2, activation="relu", name="layer1"))
# model.add(layers.Dense(3, activation="relu", name="layer2"))
# model.add(layers.Dense(4, name="layer3"))


pbounds = {
    'N_lay': (1, 5),
    'N_node': (10, 1000),
    'i_act': (0,2),
    'rate': (1e-3, 1e-1)
    }


optimizer = BayesianOptimization(
    f=training,
    pbounds=pbounds,
    verbose=2,  
    random_state=1,
)

start = time.time()
optimizer.maximize(init_points=20, n_iter=30)
end = time.time()
print('Bayes optimization takes {:.2f} seconds to tune'.format(end - start))
print(optimizer.max)
