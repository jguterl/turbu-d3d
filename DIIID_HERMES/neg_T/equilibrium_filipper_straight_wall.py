#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:12:00 2024

@author: peretm
"""


import matplotlib.pyplot as plt
import numpy as np
from random import seed
from random import random

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
def change_li_betan(lines,li,betan):
    lines2 = []
    for i in range(len(lines)):
        lines2.append(lines[i])
        if 'FLI' in lines[i]:
            i_li = i
        
        if 'FBETAN' in lines[i]:
            i_betan = i
            
    
    lines2[i_li] =  ' FLI	= ' + "{:.1f}".format(li)
    lines2[i_betan] =  ' FBETAN	= ' + "{:.1f}".format(betan)
    return lines2

plt.close('all')
path = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input.txt'
path_write = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input_flip_straight_wall.txt'
path_write2 = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input_straight_wall.txt'

lines = read_data(path)
lines3 = read_data(path)
R_wall, Z_wall = get_wall(lines)

N_side = 25
dx = 1e-8
Rmin = 0.96
Rmax = 2.45

Zmin =-1.25
Zmax = 1.05

R_wall2 = np.concatenate((np.linspace(Rmin, Rmax, N_side), np.ones(N_side)*(Rmax)+random()*dx, np.linspace(Rmax, Rmin, N_side), np.ones(N_side)*(Rmin)+random()*dx))
Z_wall2 = np.concatenate((np.ones(N_side)*(Zmin)+random()*dx, np.linspace(Zmin, Zmax, N_side), np.ones(N_side)*(Zmax)+random()*dx, np.linspace(Zmax, Zmin, N_side)))

# R_wall2.append(np.ones(50)*(2.50))
# Z_wall2.append(np.linspace(-1.25, 1.10, 50))

# R_wall2.append(np.linspace(2.50, 0.96, 50))
# Z_wall2.append(np.ones(50)*(1.10))

# R_wall2.append(np.ones(50)*(0.96))
# Z_wall2.append(np.linspace(1.1, -1.25, 50))

# R_wall2 = np.array(R_wall2)
# Z_wall2 = np.array(Z_wall2)

R_sep, Z_sep = get_sep(lines)

li = get_li(lines)
betan = get_betan(lines)

plt.figure()
plt.plot(R_wall, Z_wall, 'k')
plt.plot(R_wall2, Z_wall2, 'r')
plt.plot(R_sep, Z_sep, '-ob')

print(np.min(R_wall[R_wall>0.0]))
print(np.max(R_wall[R_wall>0.0]))

R_sym = 0.5 * (np.min(R_wall[R_wall>0.0]) + np.max(R_wall[R_wall>0.0]))

R_sep_flip = R_sym - (R_sep-R_sym)

plt.plot(R_sep_flip, Z_sep, '-or')


lines2 = flip_sep(lines)
lines = change_li_betan(lines2, 1.1, 2.6)
lines3 = flip_sep(flip_sep(lines3))
lines2 = change_wall(lines2, R_wall2, Z_wall2)
lines3 = change_wall(lines3, R_wall2, Z_wall2)
lines3 = change_li_betan(lines3, 1.1, 2.6)
write_data(path_write, lines2)
write_data(path_write2, lines3)