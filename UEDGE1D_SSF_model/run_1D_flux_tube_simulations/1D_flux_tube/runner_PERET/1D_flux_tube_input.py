#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:13:19 2023

@author: peretm
"""

from UEDGEToolBox.Launcher import *
from uedge import *
import argparse
#%% ----------------------------------------------------------------------- %%#
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--project",     dest="project", type=str,   default=None, help="Parameter domain file")
parser.add_argument("-c", "--casename",   dest="casename",    type=str,   default=None, help="Input PC coef. file")
parser.add_argument("-pc", "--pcore",    dest="pcore",     type=float,   default=4e6, help="power_injected")
parser.add_argument("-n", "--ncore",   dest="ncore",      type=float,   default=0.8e19,    help="PC type")#,     choices=['HG','LU','LU_N','LG','JB','SW'])
parser.add_argument("-bt", "--btor",    dest="btor",     type=float,   default=3.0,       help="Input PC order")
parser.add_argument("-bp", "--bpol",   dest="bpol",   type=float,   default=0.3,  help="Surrogate construction method")
parser.add_argument("-rp", "--rpol",   dest="rpol",   type=float,   default=0.01,  help="Surrogate construction method")
parser.add_argument("-i", "--i",   action='store_true',  help="Surrogate construction method")
parser.add_argument("-zx", "--zxpt",   dest="zxpt",   type=float,   default=2.61,  help="Surrogate construction method")
parser.add_argument("-zd", "--zdiv",   dest="zdiv",   type=float,   default=4.18,  help="Surrogate construction method")
parser.add_argument("-nc", "--nxcore",   dest="nxcore",   type=int,   default=25,  help="Surrogate construction method")
parser.add_argument("-nl", "--nxleg",   dest="nxleg",   type=int,   default=75,  help="Surrogate construction method")
parser.add_argument("-na", "--alfxt",   dest="alfxt",   type=float,   default=6.0,  help="Surrogate construction method")
parser.add_argument("-fc", "--fc",   dest="fc",   type=float,   default=0.015,  help="Surrogate construction method")
parser.add_argument("-fx", "--fxc",   dest="fxc",   type=float,   default=4,  help="Surrogate construction method")
parser.add_argument("-t", "--tau",   dest="tau",   type=float,   default=1e-4,  help="Surrogate construction method")
args = parser.parse_args()

if args.project is not None: UBox.SetCurrentProject(args.project)
if args.casename is not None: UBox.SetCaseName(args.casename)


# # iota=np.array([np.arange(1,i+1) for i in range(1,300)],dtype=np.ndarray)
api.apidir=Source('api',Folder='/fusion/projects/boundary/peretm/database')
aph.aphdir=Source('aph',Folder='/fusion/projects/boundary/peretm/database')
com.coronalimpfname = Source('api/mist.dat', Folder='/fusion/projects/boundary/peretm/database')

# bbb.restart = 0	#Begin from savefile, not estimated profiles
# bbb.allocate()		#allocate space for savevariables
# UBox.Initialize()
# bbb.restart = 1
# UBox.InitPlasma()

bbb.mhdgeo = 1 		#use MHD equilibrium
bbb.gengrid = 0
bbb.GridFileName = "/fusion/projects/boundary/peretm/1D_flux_tube/gridue"
#!rm -f aeqdsk neqdsk		#change names of MHD eqil. files 
#!cp aeqdskd3d aeqdsk		# (Cannot tab or indent these 3 lines)
#!cp neqdskd3d neqdsk
flx.psi0min1 = 0.9999		#normalized flux on core bndry
flx.psi0min2 = 0.9999		#normalized flux on pf bndry
flx.psi0sep = 1.00001	#normalized flux at separatrix
flx.psi0max = 1.015		#normalized flux on outer wall bndry
bbb.ngrid = 1		#number of meshes (always set to 1)
com.nxleg[0,0] = 16		#pol. mesh pts from inner plate to x-point
com.nxcore[0,0] = 30	#pol. mesh pts from x-point to top on inside
com.nxcore[0,1] = 30	#pol. mesh pts from top to x-point on outside
com.nxleg[0,1] = 32		#pol. mesh pts from x-point to outer plate
com.nysol[0] = 1		#rad. mesh pts in SOL
com.nycore[0] = 0		#rad. mesh pts in core
#	alfcy = 2.5		#factor concentrating mesh near separatrix
com.nxomit = 59		#only consider half-space btwn outer mp & plate

# Finite-difference algorithms (upwind, central diff, etc.)
bbb.methn = 33		#ion continuty eqn
bbb.methu = 33		#ion parallel momentum eqn
bbb.methe = 33		#electron energy eqn
bbb.methi = 33		#ion energy eqn
bbb.methg = 33		#neutral gas continuity eqn

# Boundary conditions
bbb.isnicore = 1		#=1 sets density = ncore
bbb.ncore[0] = args.ncore	#hydrogen ion density on core
bbb.iflcore = 1		#set power to pcoree,i
bbb.pcoree = args.pcore / 2.0		#core elec power 
bbb.pcorei = args.pcore / 2.0		#core ion power
bbb.tedge = 2.		#fixed wall,pf Te,i if istewcon=1, etc
bbb.recycp[0] = 0.99 	#hydrogen recycling coeff at plates
bbb.isnwconi = 0		#=0 for fniy=0
bbb.isnwcono = 0
bbb.isupwo = 2
bbb.isupwi = 2
bbb.lyup = 1.e5
bbb.isupcore = 1		#gives fmiy=0 on core bndry
bbb.istewc = 3
bbb.istiwc = 3
bbb.istepfc = 3
bbb.istipfc = 3
bbb.lyte = 1.e5
bbb.lyti = 1.e5
bbb.matwso = 1
bbb.matwsi = 1
bbb.recycw = 1e-10
bbb.recycm = -0.95		#up(nx,,2) = -recycm*up(nx,,1)
bbb.isupss = 1		#=1 allows supersonic BC
bbb.isfixlb = 2		#=1 sets symmetry BC at outer midplane (ix=0)
##..pumping
#bbb.albdsi[0] = 0.995
#..pumping
bbb.albdsi[0] = 0.998
bbb.albdso[0] = 0.998
bbb.ckinfl = 0.

# Localized Gaussian particle source
bbb.allocate()		#need to generate storage space
bbb.ivolcur[0] = 0. 
bbb.z0ni = 1.69		#outer midplane
bbb.r0ni = 2.32		#outer midplane
bbb.zwni = 0.2		#vertical Gaussian half-width
bbb.rwni = 0.2		#radial Gaussian half-width

# Transport coefficients
bbb.difni[0] = 1.		#D for radial hydrogen diffusion
bbb.kye = 1.		#chi_e for radial elec energy diffusion
bbb.kyi = 1.		#chi_i for radial ion energy diffusion
bbb.travis[0] = 1.e-4	#eta_a for radial ion momentum diffusion
bbb.parvis = 1.0		#parallel visc coefficent

# Flux limits
bbb.flalfe = 0.21		#electron parallel thermal conduct. coeff
bbb.flalfi = 1.#0.21		#ion parallel thermal conduct. coeff
bbb.flalfv = 0.5		#ion parallel viscosity coeff
bbb.flalfgx = 1.		#neut. dens in poloidal direction
bbb.flalfgy = 1.		#neut. dens in radial direction
bbb.flalfvgx = 1.		#neut. momentum dens in poloidal direction
bbb.flalfvgy = 1.		#neut. momentum dens in radial direction
bbb.flalftgx = 1.		#neut. energy dens in poloidal direction
bbb.flalftgy = 1.		#neut. energy dens in radial direction
bbb.isplflxl = 0		#=0 turns off Te,i flux limiting at plates

# Background neutral source at very low density
bbb.ngbackg[0] = 1.e11

# Ion-electron recombination
bbb.isrecmon = 1

# Solver package
bbb.svrpkg = "nksol"	#Newton solver using Krylov method
bbb.premeth = "ilut"	#Solution method for precond. Jacobian matrix
bbb.mfnksol = 3

# Parallel neutral momentum equation
bbb.isupgon[0]=1

bbb.isngon[0]=0
com.ngsp=1
com.nhsp=2
bbb.ziin[com.nhsp-1]=0
bbb.ineudif = 2

#..Carbon impurity with fixed fraction
bbb.isimpon = 2
bbb.afracs = args.fc

'''
#..zml difference from singe version
bbb.iswflxlvgy = 1
bbb.iswflxltgy = 1
bbb.cfvgpy[1] = 1.
bbb.isfdiax2 = 1.
'''

# Restart from a pfb savefile
# bbb.restart = 0	#Begin from savefile, not estimated profiles
# bbb.allocate()		#allocate space for savevariables
# UBox.InitPlasma()
# UBox.Initialize()
# bbb.restart = 0
# # 
bbb.restart = 0	#Begin from savefile, not estimated profiles
bbb.allocate()		#allocate space for savevariables
UBox.Initialize()
bbb.restart = 1
UBox.InitPlasma()

bbb.ncore[0] = args.ncore	#hydrogen ion density on core
bbb.pcoree = args.pcore / 2.0		#core elec power 
bbb.pcorei = args.pcore / 2.0
bbb.isimpon = 2
bbb.afracs = args.fc

bbb.dtreal = 1e-9
bbb.itermx = 30
bbb.exmain()

if bbb.iterm == 1:
   print ('before rdcontdt')
   exec(open('/fusion/projects/boundary/peretm/1D_flux_tube/runner_PERET/rdcontdt_exec.py').read())
   fnrm_old = np.sqrt(sum((bbb.yldot[0:bbb.neq-1]*bbb.sfscal[0:bbb.neq-1])**2))
   if (fnrm_old<bbb.ftol_min):
       bbb.pradpltwl()
       UBox.Save('final_state.npy')
       UBox.Save('fc.npy',DataSet = [('afracs','atau')],DataType=['UEDGE'])
       UBox.Save('power.npy',DataSet = [('prad','pradcff','pradc','pradhyd','feex','feey','feix','feiy','pwr_pltz','pwr_plth','pwr_wallh','pwr_wallz')],DataType=['UEDGE'],OverWrite=True)

# UBox.Run(mult_dt_fwd=3.4,dtreal = 1e-7, t_stop=1e3)

# bbb.pwr_wallh
# bbb.pwr_wallz
# bbb.pwr_pltz
# bbb.pwr_plth
# UBox.Run(mult_dt_fwd=3.4,dtreal = 1e-9, t_stop=1e3)


#UBox.Init()
#hdf5_restore('pf_1D_outer_sol.hdf5')

#grd.fuzz=2