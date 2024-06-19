#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:29:13 2024

@author: peretm
"""


import numpy as np
import matplotlib.pyplot as plt
import math

plt.close('all')

Psep = np.linspace(5e6, 5e10, 100)
Te = 0.0*Psep+0.0

g = 0.0 * Te + 0.0
sig_N = 0.0 * Te + 0.0
sig_PHI = 0.0 * Te + 0.0
l0 = 0.0 * Te + 0.0
rho0 = 0.0 * Te + 0.0
alpha_ExB = 0.0 * Te + 0.0
gamma = 5.3
alpha_lT = np.sqrt(gamma)/(np.sqrt(gamma)-1)
fudge = 2.0



lp_sol = 0.0 * Te + 0.0

lp = np.linspace(1.0, 200, 1000)

f = np.zeros((len(Te), len(lp)))

BT = 2.0
R = 1.6
a = 0.5
kappa = 1.6

q = 5.0
alphas = -0.5



for i in range(len(Psep)):
    Te0 = 0.13 * (q**2*R/a*Psep[i]/2*7/alpha_lT/lp)**(2/7)
    rhos0 = np.sqrt(Te0*2.0*1.67e-27/1.602e-19/BT**2)
    g0 = rhos0/R
    sig_N0 = 4/math.pi/q/R *rhos0
    sig_PHI0 = 4/math.pi/q/R *rhos0
    l00 = 3.9 * g0**(3/11) * sig_PHI0**(-2/11) * sig_N0**(-4/11)
    f[i,:] = lp*(1+(alphas- fudge*0.43*3.0/g0**(0.5)*lp**(-3/2)/alpha_lT**2)**2)**(9/11) - l00
    
    
    ind_sol = np.where(np.sign(f[i,0:-1]*f[i,1:])<0.0)[0]
    
    if len(ind_sol)==0:
        lp_sol[i] = np.nan
        alpha_ExB[i] = np.nan
        g[i] = np.nan
        sig_N[i] = np.nan
        sig_PHI[i] = np.nan
        l0 = np.nan
        rho0[i] = np.nan
        
    else:
        lp_sol[i] = lp[ind_sol[-1]]
        g[i] = rhos0[ind_sol[-1]]/R
        alpha_ExB[i] = -fudge*0.43*3.0/g[i]**(0.5)*lp_sol[i]**(-3/2)/alpha_lT**2
        sig_N[i] = 4/math.pi/q/R *rhos0[ind_sol[-1]]
        sig_PHI[i] = 4/math.pi/q/R *rhos0[ind_sol[-1]]
        l0[i] = 3.9 * g0[ind_sol[-1]]**(3/11) * sig_PHI0[ind_sol[-1]]**(-2/11) * sig_N0[ind_sol[-1]]**(-4/11)
        Te[i] = Te0[ind_sol[-1]]
        rho0[i] = rhos0[ind_sol[-1]]
        
    # plt.plot(lp, f[i,:], '-o')
plt.figure()    
plt.plot(Te, lp_sol, 'o')
plt.xlabel('Separatrix temperature [eV]')
plt.ylabel(r'$\lambda_{p} [\rho_s]$')

plt.figure()
plt.plot(Te, lp_sol*rho0*1e3, 'o')
plt.xlabel('Separatrix temperature [eV]')
plt.ylabel(r'$\lambda_{p} [mm]$')

plt.figure()
plt.plot(Te, alpha_ExB, 'o')
plt.xlabel('Separatrix temperature [eV]')
plt.ylabel(r'$\alpha_{ExB}$')

plt.figure()    
plt.plot(Psep*1e-6, lp_sol*rho0*1e3, 'o')
plt.xlabel('Power crossing separatrix [MW]')
plt.ylabel(r'$\lambda_{p} [\rho_s]$')
