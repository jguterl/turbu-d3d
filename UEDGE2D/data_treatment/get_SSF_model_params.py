#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:07:35 2023

@author: peretm
"""

import numpy as np
import matplotlib.pyplot as plt
import get_separatrix_data as get_sep
from scipy import integrate
from get_profiles import get_upstream_2target_2Xpt_profiles

ee = 1.602e-19
mi = 1.67e-27
A = 1.0

#file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/final_state_100723_135235.npy'
# file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc1.40e+19_pcore2.00e+06/final_state_120723_093600.npy'
#file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc1.40e+19_pcore2.00e+06/final_state_120723_105248.npy'
# file2load = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc1.60e+19_pcore2.00e+06/final_state_120723_120155.npy'
file2load ='/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc1.60e+19_pcore2.00e+06/final_state_120723_143115.npy'
rrfile = '/fusion/projects/boundary/peretm/simulations/Hplasma_2/SaveDir/rd_newgrid2_nc7.00e+18_pcore2.00e+06/rr.npy'
sep = get_sep.get_separatrix_par_data(file2load, rrfile)
i_u, i_x1, i_x2, i_t1, i_t2 = get_sep.get_u2x2t_indices(sep['R'],sep['Z'])
Rmin, Rmax, Zmin, Zmax, R0, Z0, a, kappa, delta_up, delta_low, delta, Rxpt, Zxpt = get_sep.get_sep_geom_params(sep['R'], sep['Z'])

teU = sep['te'][i_u]
teX = sep['te'][i_x2]
teT = sep['te'][i_t1]
tiU = sep['ti'][i_u]
tiX = sep['ti'][i_x2]
tiT = sep['ti'][i_t1]
neU = sep['ne'][i_u]
neX = sep['ne'][i_x2]
neT = sep['ne'][i_t1]
MX = sep['M'][i_x2]
MT = sep['M'][i_t1]

rhos = np.sqrt(ee/mi/A*teU)/(ee*3.0/A/mi)


taue = integrate.simps(sep['te'][i_x2:i_t1]**(3/2)/sep['ne'][i_x2:i_t1], sep['spar'][i_x2:i_t1])*2.33e10#/(sep['spar'][i_t1]-sep['spar'][i_x2])
Ldiv = (sep['spar'][i_t1]-sep['spar'][i_x2])/1836/np.sqrt(ee/mi/A*teT)/taue
g = 1.5 * rhos / R0 
sig_0 = 2 * rhos / (sep['spar'][i_x2]-sep['spar'][i_u])
sig_N = 2 * sig_0 *np.sqrt((teX+tiX)/(teU+tiU))*neX/neU*MX
sig_PHI = sig_0 * np.sqrt((teU+tiU)/(teT+tiT))/(1+Ldiv/0.51)
ln_mod = 3.9 * g**(3/11) * sig_N**(-4/11) * sig_PHI**(-2/11) * rhos 

prof = get_upstream_2target_2Xpt_profiles(file2load)

n = prof['upstream']['ne'][prof['upstream']['rsep']>=0]
r = prof['upstream']['rsep'][prof['upstream']['rsep']>=0]
T = prof['upstream']['te'][prof['upstream']['rsep']>=0]

ln = -0.5 * (n[0:-1]+n[1:]) * np.diff(r) / np.diff(n)
lt = -0.5 * (T[0:-1]+T[1:]) * np.diff(r) / np.diff(T)

plt.figure()
plt.plot(r, n, 'ko')
plt.plot(r, n[0]*np.exp(-r/ln[0]),'ro')
plt.plot(r,n[0]*np.exp(-r/ln_mod), 'bo')

plt.figure()
plt.plot(r, T, 'ko')
plt.plot(r, T[0]*np.exp(-r/lt[0]),'ro')

# plt.figure()
# plt.plot(sep['spar'], sep['up'],'k')
# plt.plot(sep['spar'][i_u], sep['up'][i_u],'ro')

# fig, ax = plt.subplots(1)
# ax.plot(sep['R'], sep['Z'], 'ko')
# ax.plot(Rxpt, Zxpt, 'ro')
# ax.plot(R0, Z0, 'bo')
# ax.plot(sep['R'][i_u], sep['Z'][i_u], 'go')
# ax.plot(sep['R'][i_x2], sep['Z'][i_x2], 'go')
# ax.plot(sep['R'][i_t1], sep['Z'][i_t1], 'go')
# ax.plot(sep['R'][i_x1], sep['Z'][i_x1], 'mo')
# ax.plot(sep['R'][i_t2], sep['Z'][i_t2], 'mo')
# ax.axis('equal')