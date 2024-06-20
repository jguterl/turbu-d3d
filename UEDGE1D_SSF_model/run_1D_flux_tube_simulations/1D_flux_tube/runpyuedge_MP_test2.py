#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:26:06 2023

@author: peretm
"""

from uedge import *
from uedge.hdf5 import *
import os
import shutil
from UEDGEToolBox.Launcher import *

import rd_1D_outer_sol_2

import time
startT = time.time()
# fixed parameters
#pin = 1.e6
#frc = 1e-10
#alb = 0.995
#swit = 13.47263*1.3
#bbb.pcoree = pin/2./swit
#bbb.pcorei = pin/2./swit
#bbb.afracs = frc/100.
#bbb.albdsi[0] = 0.995

# Do density scan

n_sample = [5e18]

for isam in range(len(n_sample)):
  # bbb.restart = 0	#Begin from savefile, not estimated profiles
  # bbb.allocate()		#allocate space for savevariables
  # UBox.Initialize()
  # bbb.restart = 1
  # UBox.InitPlasma()
  bbb.ncore[0] = n_sample[isam]
  bbb.pcoree = 2.0e6		#core elec power 
  bbb.pcorei = 2.0e6		#core ion power
  bbb.isimpon = 2
  bbb.afracs = 0.015
  fn = 'n'+str(round(n_sample[isam]/1e18,0))
  # h5n = 'savedt.hdf5' + '_' + fn

  # run uedge and save files
  bbb.dtreal = 1e-9
  bbb.itermx = 30
  #bbb.nis = 0.999*bbb.nis
  #bbb.icntnunk = 0
  bbb.exmain()
    # bbb.ncore[0] = 5e18
    # bbb.pcoree = 2.0e6		#core elec power 
    # bbb.pcorei = 2.0e6
    # bbb.isimpon = 2
    # bbb.afracs = 0.015
    # bbb.dtreal = 1e-9
    # bbb.itermx = 30
    # bbb.exmain()
# # loop
# for isam in range(len(n_sample)):
#   # bbb.restart = 0	#Begin from savefile, not estimated profiles
#   # bbb.allocate()		#allocate space for savevariables
#   # UBox.Initialize()
#   # bbb.restart = 1
#   # UBox.InitPlasma()
#    n_sample[isam]
#   		#core ion power
  
#   fn = 'n'+str(round(n_sample[isam]/1e18,0))
#   # h5n = 'savedt.hdf5' + '_' + fn

#   # run uedge and save files
  
#   #bbb.nis = 0.999*bbb.nis
#   #bbb.icntnunk = 0
  
#   if bbb.iterm == 1:
#       print ('before rdcontdt')
#       exec(open('rdcontdt_exec.py').read())
#       fnrm_old = np.sqrt(sum((bbb.yldot[0:bbb.neq-1]*bbb.sfscal[0:bbb.neq-1])**2))
#       if (fnrm_old<bbb.ftol_min):
#           UBox.SaveData('test_' + fn + '_imp2.npy')
#           # exec(open('/home/zml/uedge_new_github/UEDGE/pylib/savedata.py').read())
#           # savedata()
#           # #if not os.path.exists(fn):
#           # #    os.makedirs(fn)
#           # if os.path.exists(fn):
#           #     shutil.rmtree(fn, ignore_errors=True)
#           #     os.rename('data',fn)
#           # else:
#           #     os.rename('data',fn)
#           # os.rename('savedt.hdf5',h5n)
#   else:
#       print('change ncore')

  

# runT = time.time() - startT
# open(str(nmin)+'-'+str(nmax)+'-'+str(nsa)+'_'+str(runT),'a').close()
# exit()
