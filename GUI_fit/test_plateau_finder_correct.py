#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:03:06 2023

@author: peretm
"""

from get_DIIID_data import *
import matplotlib.pyplot as plt

def find_plateaus(F, min_length=200, tolerance = 0.75, smoothing=25):
    '''
    Finds plateaus of signal using second derivative of F.

    Parameters
    ----------
    F : Signal.
    min_length: Minimum length of plateau.
    tolerance: Number between 0 and 1 indicating how tolerant
        the requirement of constant slope of the plateau is.
    smoothing: Size of uniform filter 1D applied to F and its derivatives.
    
    Returns
    -------
    plateaus: array of plateau left and right edges pairs
    dF: (smoothed) derivative of F
    d2F: (smoothed) Second Derivative of F
    '''
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    
    # calculate smooth gradients
    smoothF = uniform_filter1d(F, size = smoothing)
    dF = uniform_filter1d(np.gradient(smoothF),size = smoothing)/smoothF
    d2F = uniform_filter1d(np.gradient(dF),size = smoothing)
    
    def zero_runs(x):
        '''
        Helper function for finding sequences of 0s in a signal
        https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
        '''
        iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges
    
    # Find ranges where second derivative is zero
    # Values under eps are assumed to be zero.
    # eps = np.quantile(abs(d2F),tolerance) 
    smalld2F = (abs(dF) <= tolerance)
    # Find repititions in the mask "smalld2F" (i.e. ranges where d2F is constantly zero)
    p = zero_runs(np.diff(smalld2F))
    
    
    # np.diff(p) gives the length of each range found.
    # only accept plateaus of min_length
    
    plat =[]
    try:
        plat = p[(np.diff(p) > min_length).flatten()]
    except:
        []
    plateaus = []
    
    if plat!=[]:
        for i in range(len(plat)):
            if np.abs(np.mean(dF[plat[i,0]:plat[i,1]]))<tolerance:
                plateaus.append(plat[i,:])
    
    return (plateaus, dF, d2F, smoothF)
plt.close('all')
shotnumber = 180334
database = fetch_data(shotnumber)



time = database['tauth']['time']
data = database['tauth']['data']

# tol = 4e-5 #Good for nl and for Ip
# tol = 0.006 # Good for Pohm
tol = 0.009 #Good for tauth



dt_min = 400
di_min = int(np.floor(dt_min/np.mean(np.diff(time))))
dt_smooth = 250
di_smooth = int(np.floor(dt_smooth/np.mean(np.diff(time))))
plt.figure()
plt.plot(time, data, 'b')
plateaus, dF, d2F, smoothdata = find_plateaus(data, min_length=di_min, tolerance = tol, smoothing=di_smooth)
plt.plot(time, smoothdata, '--k')
for i in range(len(plateaus)):
    plt.plot([time[plateaus[i][0]],time[plateaus[i][1]]], [np.mean(data[plateaus[i][0]:plateaus[i][1]]),np.mean(data[plateaus[i][0]:plateaus[i][1]])],'r')

plt.figure()
plt.plot(time, abs(dF), 'b')