#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:55:57 2024

@author: peretm
"""
from omfit_classes.omfit_thomson import OMFITthomson
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import matplotlib.pyplot as plt
import numpy as np
from intersect import intersection
from random import seed
from random import random
import math
import matplotlib.colors as colors
from scipy.interpolate import interp2d
from omfit_classes.omfit_mds import OMFITmdsValue

plt.close('all')

def find_Xpt(R, Z, Bp, Rwall, Zwall):
    ind0 = np.where(Z>=0)[0]
    ind1 = np.where(Z<0)[0]
    RXpt0 = np.zeros(len(R))
    RXpt1 = np.zeros(len(R))
    ZXpt0 = np.zeros(len(R))
    ZXpt1 = np.zeros(len(R))   
    Bp0 = np.zeros(len(R))  
    Bp1 = np.zeros(len(R))  
    Rout = np.zeros(2)
    Zout = np.zeros(2)
    for i in range(len(R)):
        ZXpt0[i]=Z[ind0[np.nanargmin(Bp[ind0,i])]]
        Bp0[i] = Bp[ind0[np.nanargmin(Bp[ind0,i])],i]
        ZXpt1[i]=Z[ind1[np.nanargmin(Bp[ind1,i])]]
        Bp1[i] = Bp[ind1[np.nanargmin(Bp[ind1,i])],i]
    for i in range(len(R)):
        Rpt = np.arange(R[i], 3.0, 1e-3)
        Zpt0 = Rpt * 0.0 +ZXpt0[i]
        Zpt1 = Rpt * 0.0 +ZXpt1[i]
        x0, y0 = intersection(Rpt, Zpt0,Rwall, Zwall)
        x0 = np.unique(x0)
        y0 = np.unique(y0)
        if x==[]:
            ZXpt0[i] = math.nan
            Bp0[i] = math.nan
                
        elif np.floor(len(x0)/2)==len(x0)/2:
            ZXpt0[i] = math.nan
            Bp0[i] = math.nan
    
        x1, y1 = intersection(Rpt, Zpt1,Rwall, Zwall)
        x1 = np.unique(x1)
        y1 = np.unique(y1)
        if x==[]:
            ZXpt1[i] = math.nan
            Bp1[i] = math.nan
                
        elif np.floor(len(x1)/2)==len(x1)/2:
            ZXpt1[i] = math.nan
            Bp1[i] = math.nan
    Rout[0] = R[np.nanargmin(Bp0)]
    Rout[1] = R[np.nanargmin(Bp1)]
    Zout[0] = ZXpt0[np.nanargmin(Bp0)]
    Zout[1] = ZXpt1[np.nanargmin(Bp1)]    
    return Rout, Zout

def read_data(file_path):
    f = open(file_path,'r')
    lines = f.readlines()
    f.close()
    return lines

def write_data(file_path, lines):
    f = open(file_path, 'w')
    for line in lines:
        f.write(line)
        f.write('\n')
    f.close()
def get_nbdry(lines):
    data_txt = []
    for i in range(len(lines)):
        if 'NBDRY' in  lines[i]:
            i_nbdry = i
    data_txt.append(lines[i_nbdry][8:]) 
    NBDRY = int([num for num in data_txt[0].split()][0])
    return NBDRY

def get_wall(lines):
    for i in range(len(lines)):
        if 'XLIM = ' in lines[i]:
            i_xlim = i
        
        if 'YLIM = ' in lines[i]:
            i_ylim = i
            
        if 'NBDRY' in  lines[i]:
            i_nbdry = i
    
    data_txt = []
    R = []
    Z = []
    for i in range(i_xlim,i_ylim):
        if i==i_xlim:
            data_txt.append(lines[i][8:])  
        else:
            data_txt.append(lines[i])
        
    for i in range(len(data_txt)):
        data0 = [num for num in data_txt[i].split()]
        for j in range(len(data0)):
            if '*' in data0[j]:
                ind = data0[j].find('*')
                for k in range(int(data0[j][0:ind])):
                    R.append(float(data0[j][ind+1:]))
            else:
                R.append(float(data0[j]))
                
    data_txt = []
    for i in range(i_ylim,i_nbdry):
        if i==i_ylim:
            data_txt.append(lines[i][8:])  
        else:
            data_txt.append(lines[i])
        
    for i in range(len(data_txt)):
        data0 = [num for num in data_txt[i].split()]
        for j in range(len(data0)):
            if '*' in data0[j]:
                ind = data0[j].find('*')
                for k in range(int(data0[j][0:ind])):
                    Z.append(float(data0[j][ind+1:]))
            else:
                Z.append(float(data0[j]))

    return np.array(R), np.array(Z)
def get_nwall(lines):
    Rwall, Zwall = get_wall(lines)
    return len(Rwall)

def get_sep(lines):
    for i in range(len(lines)):
        if 'RBDRY = ' in lines[i]:
            i_xlim = i
        
        if 'ZBDRY = ' in lines[i]:
            i_ylim = i
            
        if 'IERCHK' in  lines[i]:
            i_nbdry = i
    
    data_txt = []
    R = []
    Z = []
    for i in range(i_xlim,i_ylim):
        if i==i_xlim:
            data_txt.append(lines[i][8:])  
        else:
            data_txt.append(lines[i])
        
    for i in range(len(data_txt)):
        data0 = [num for num in data_txt[i].split()]
        for j in range(len(data0)):
            if '*' in data0[j]:
                ind = data0[j].find('*')
                for k in range(int(data0[j][0:ind])):
                    R.append(float(data0[j][ind+1:]))
            else:
                R.append(float(data0[j]))
                
    data_txt = []
    for i in range(i_ylim,i_nbdry):
        if i==i_ylim:
            data_txt.append(lines[i][8:])  
        else:
            data_txt.append(lines[i])
        
    for i in range(len(data_txt)):
        data0 = [num for num in data_txt[i].split()]
        for j in range(len(data0)):
            if '*' in data0[j]:
                ind = data0[j].find('*')
                for k in range(int(data0[j][0:ind])):
                    Z.append(float(data0[j][ind+1:]))
            else:
                Z.append(float(data0[j]))

    return np.array(R), np.array(Z)

def get_li(lines):
    for i in range(len(lines)):
        if 'FLI' in lines[i]:
            i_xlim = i
    data_txt = lines[i_xlim]
    ind = data_txt.find('=')
    li = float(data_txt[ind+1:])      
    return li

def get_betan(lines):
    for i in range(len(lines)):
        if 'FBETAN' in lines[i]:
            i_xlim = i
    data_txt = lines[i_xlim]
    ind = data_txt.find('=')
    betan = float(data_txt[ind+1:])      
    return betan  

def flip_sep(lines):
    lines2 = lines
    R_wall, Z_wall = get_wall(lines)
    R_sep, Z_sep = get_sep(lines)
    R_sym = 0.5 * (np.min(R_wall[R_wall>0.0]) + np.max(R_wall[R_wall>0.0]))
    R_sep_flip = R_sym - (R_sep-R_sym)
    for i in range(len(lines)):
        if 'RBDRY = ' in lines[i]:
            i_xlim = i
    
    for i in range(len(R_sep_flip)):
        if i==0:
            lines2[i_xlim] =  ' RBDRY =  ' + "{:.4f}".format(R_sep_flip[i])

        else:
            lines2[i_xlim+i] = '    ' + "{:.4f}".format(R_sep_flip[i])
    return lines2

def change_sep(lines, R_sep, Z_sep):
    lines2 = []
    # lines2 = lines
    for i in range(len(lines)):
        lines2.append(lines[i])
        if 'RBDRY = ' in lines[i]:
            i_xlim = i
        if 'ZBDRY = ' in lines[i]:
            i_ylim = i

    for i in range(len(R_sep)):
        if i==0:
            lines2[i_xlim] =  ' RBDRY =  ' + "{:.4f}".format(R_sep[i])

        else:
            lines2[i_xlim+i] = '    ' + "{:.4f}".format(R_sep[i])
    for i in range(len(Z_sep)):
        if i==0:
            lines2[i_ylim] =  ' ZBDRY =  ' + "{:.4f}".format(Z_sep[i])
        else:
            lines2[i_ylim+i] =  '    ' + "{:.4f}".format(Z_sep[i])
    return lines2

def change_sep_weights(lines, R_sep, Z_sep):
    lines2 = []
    
    Weights = 1.0 + 0.0 * R_sep
    Weights[R_sep<np.min(R_sep)+0.05*(np.max(R_sep)-np.min(R_sep))] = 100.0
    Weights[Z_sep<np.min(Z_sep)+0.05*(np.max(Z_sep)-np.min(Z_sep))] = 100.0
    Weights[R_sep>np.min(R_sep)+0.95*(np.max(R_sep)-np.min(R_sep))] = 100.0
    Weights[Z_sep>np.min(Z_sep)+0.95*(np.max(Z_sep)-np.min(Z_sep))] = 100.0    
    # lines2 = lines
    for i in range(len(lines)):
        lines2.append(lines[i])
        if  'FWTBDRY = ' in lines[i]:
            i_xlim = i
    
    for i in range(len(R_sep)):
        if i==0:
            lines2[i_xlim] =  ' FWTBDRY = ' + "{:.1f}".format(Weights[i])

        else:
            lines2[i_xlim+i] = '   ' + "{:.1f}".format(Weights[i])
 
    return lines2


def change_wall(lines, R_wall, Z_wall):
    
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
        if 'XLIM = ' in lines[i]:
            i_xlim = i
        
        if 'YLIM = ' in lines[i]:
            i_ylim = i
            
        if 'NBDRY' in  lines[i]:
            i_nbdry = i
    
    nperline_R = np.ceil(len(R_wall)/(i_ylim-i_xlim))
    nperline_Z = np.ceil(len(Z_wall)/(i_nbdry-i_ylim))
    
    n = 0
    j = 0
    text = ' XLIM = '
    for i in range(len(R_wall)):
        if n<nperline_R-1:
            text = text +  ' ' + "{:.4f}".format(R_wall[i])
            n = n+1
            
        elif (n==nperline_R-1):
            text = text + ' ' + "{:.4f}".format(R_wall[i])
            lines2[i_xlim+j] = text
            j = j+1
            n = 0
            text = ' '
            
        if (i==len(R_wall)-1):
            text = text + ' ' + "{:.4f}".format(R_wall[i])
            lines2[i_xlim+j] = text
            
    n = 0
    j = 0
    text = ' YLIM = '
    for i in range(len(Z_wall)):
        if n<nperline_Z-1:
            text = text +  ' ' + "{:.4f}".format(Z_wall[i])
            n = n+1
            
        elif (n==nperline_Z-1):
            text = text +  ' ' + "{:.4f}".format(Z_wall[i])
            lines2[i_ylim+j] = text
            
            j = j+1
            n = 0
            text = ' '
            
        if (i==len(Z_wall)-1):
            text = text + ' ' + "{:.4f}".format(Z_wall[i])
            lines2[i_ylim+j] = text  
        
        lines2[i_xlim-1] = ' LIMITR =                ' + str(len(R_wall))

    return lines2

def change_shot(lines, shot):
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
    
    lines2[1] = ' ISHOT	= ' +str(shot) 
    return lines2

def change_time(lines, time):
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
    
    lines2[2] = ' ITIME	= ' +str(time) 
    return lines2

def change_li_betan(lines,shot,time):
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
        if 'FLI' in lines[i]:
            i_li = i
        
        if 'FBETAN' in lines[i]:
            i_betan = i
            
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='BETAN')
    if (tmp2.dim_of(0)) is not None:
        
        betan_time = tmp2.dim_of(0)
        index = np.min(np.nanargmin(np.abs(betan_time-time)))
        betan = tmp2.data()[index.astype(int)]  
        
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='LI')
    if (tmp2.dim_of(0)) is not None:
        
        li_time = tmp2.dim_of(0)
        index = np.min(np.nanargmin(np.abs(li_time-time)))
        li = tmp2.data()[index.astype(int)] 
    
    lines2[i_li] =  ' FLI	= ' + "{:.1f}".format(li)
    lines2[i_betan] =  ' FBETAN	= ' + "{:.1f}".format(betan)
    return lines2

def change_Ip_Bt(lines,shot,time):
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
        if 'BTOR' in lines[i]:
            i_Bt = i
        
        if 'PLASMA' in lines[i]:
            i_Ip = i
            
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='BT')
    if (tmp2.dim_of(0)) is not None:
        
        betan_time = tmp2.dim_of(0)
        index = np.min(np.nanargmin(np.abs(betan_time-time)))
        BT = tmp2.data()[index.astype(int)]  
        
    tmp2 = OMFITmdsValue('DIII-D', treename=None, shot=shot, TDI='IP')
    if (tmp2.dim_of(0)) is not None:
        
        li_time = tmp2.dim_of(0)
        index = np.min(np.nanargmin(np.abs(li_time-time)))
        Ip = tmp2.data()[index.astype(int)] 

    lines2[i_Ip] =  ' PLASMA	= ' + "{:.1f}".format(Ip)
    lines2[i_Bt] =  ' BTOR	= ' + "{:.1f}".format(BT)
    return lines2           
            
shot = 188855 #194371 #119919
# shot = 194661
time = 3000

plt.close('all')
path = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input.txt'
path_write = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input_test_' + str(shot) +'_t_' +str(time) + '.txt'

lines = read_data(path)

NBDRY = get_nbdry(lines)

NWALL = get_nwall(lines)


Filter = {'core': {'redchisq_limit': 10.0,'frac_temp_err_hot_max': 0.3,
                      'frac_temp_err_cold_max': 0.95,
                      'frac_dens_err_max': 0.3}}
ts = OMFITthomson('DIII-D', shot, 'EFIT01', -1, ['core', 'tangential'], quality_filters=Filter)
ts()
eq = ts['efit_data']

ind = np.min(np.nanargmin(np.abs(eq['atime']-time)))

R = np.arange(np.nanmin(eq['r']), np.nanmax(eq['r']), 2e-2)
Z = np.arange(np.nanmin(eq['z']), np.nanmax(eq['z']), 2e-2)

psi_interp = interp2d(eq['r'], eq['z'], eq['psin'][ind,:,:])
psin = psi_interp(R, Z)

R_2D, Z_2D = np.meshgrid(R, Z)

for i in range(len(R)):
    for j in range(len(Z)):
        Rpt = np.arange(R[i], 3.0, 1e-3)
        Zpt = Rpt * 0.0 +Z[j]
        x, y = intersection(Rpt, Zpt,eq['lim'][:,0], eq['lim'][:,1])
        x = np.unique(x)
        y = np.unique(y)
        if x==[]:
            # psi[j,i] = 2.0+random()
            psin[j,i] = 2.0+random()
            
        elif np.floor(len(x)/2)==len(x)/2:
            # psi[j,i] = 2.0+random()
            psin[j,i] = 2.0+random()
            
plt.figure()
fig, ax =plt.subplots(1)
im = ax.pcolormesh(R,Z,psin, cmap='hot', shading='auto', norm=colors.Normalize(0.0, 2.0))
fig.colorbar(im,ax=ax)

# plt.figure()
cs = plt.contour(R, Z, psin, levels = [1])
# breakpoint()
# print('test')
# plt.plot(eq['lim'][:,0], eq['lim'][:,1], 'k')
# plt.axis('equal')
plt.close()
for item in cs.collections:
   for i in item.get_paths():
      v = i.vertices
      R_sep = v[:, 0]
      Z_sep = v[:, 1]

BR = -1/R_2D[0:-1,:] * np.diff(psin, axis = 0)/np.diff(Z_2D, axis = 0)
BZ =  1/R_2D[:,0:-1] * np.diff(psin, axis = 1)/np.diff(R_2D, axis = 1)



BR = np.delete(BR, -1, 1)

BZ = np.delete(BZ, -1, 0)

psin = np.delete(psin, -1, 1)
psin = np.delete(psin, -1, 0)
R_int = np.delete(R,-1,0)
Z_int = np.delete(Z, -1, 0)


Rxpt, Zxpt = find_Xpt(R_int, Z_int, np.sqrt(BR**2+BZ**2), eq['lim'][:,0], eq['lim'][:,1])

Rxpt = Rxpt[-1]
Zxpt = Zxpt[-1]

if Zxpt<0:
    ind = np.where(Z_sep> 0.95 * Zxpt)[0]
    R_sep = R_sep[ind]
    Z_sep = Z_sep[ind]
    
if Zxpt>0:
    ind = np.where(Z_sep< 0.95 * Zxpt)[0]
    R_sep = R_sep[ind]
    Z_sep = Z_sep[ind]
    
# plt.plot(R_sep, Z_sep, '-ok')

ind_cut = np.unique(np.floor(np.linspace(0, len(R_sep)-1, NBDRY))).astype(int)

# R_sep2, Z_sep2 = get_sep(lines)
# R_wall2, Z_wall2 = get_wall(lines)
# R_wall = eq['lim'][:,0]
# Z_wall = eq['lim'][:,1]
# plt.figure()
# plt.plot(R_sep, Z_sep, 'o')
# plt.plot(R_sep2, Z_sep2, 'o')
# plt.figure()
# plt.plot(R_wall, Z_wall, 'o')
# plt.plot(R_wall2, Z_wall2, 'o')

lines2 = change_sep(lines, R_sep[ind_cut], Z_sep[ind_cut])
lines2 = change_sep_weights(lines2, R_sep[ind_cut], Z_sep[ind_cut])
lines2 = change_wall(lines2, eq['lim'][:,0], eq['lim'][:,1])
lines2 = change_shot(lines2, shot)
lines2 = change_time(lines2, time)
lines2 = change_li_betan(lines2,shot,time)
lines2 = change_Ip_Bt(lines2,shot,time)

write_data(path_write, lines2)
# eq = read_basic_eq_from_mds(device='DIII-D', shot=shot, tree='EFIT01', quiet=False, toksearch_mds=None)

# efit_type = 'EFIT01'

# time = 2500
# device = 'DIII-D'

# eq = OMFITgeqdsk('g%06d.%05d' % (shot, time)).from_mdsplus(
#     device=device, shot=shot, time=time, SNAPfile=efit_type)
# a = eq['fluxSurfaces'].load()

# # eq = eq.to_omas()

