#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:53:14 2024

@author: peretm
"""
import matplotlib.pyplot as plt
import numpy as np

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
    
plt.close('all')
path = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input.txt'
path_write = '/home/peretm/HERMES/DIIID_HERMES/neg_T/input_flip.txt'

lines = read_data(path)

R_wall, Z_wall = get_wall(lines)

R_sep, Z_sep = get_sep(lines)

li = get_li(lines)
betan = get_betan(lines)

plt.figure()
plt.plot(R_wall, Z_wall, 'k')
plt.plot(R_sep, Z_sep, '-ob')

print(np.min(R_wall[R_wall>0.0]))
print(np.max(R_wall[R_wall>0.0]))

R_sym = 0.5 * (np.min(R_wall[R_wall>0.0]) + np.max(R_wall[R_wall>0.0]))

R_sep_flip = R_sym - (R_sep-R_sym)

plt.plot(R_sep_flip, Z_sep, '-or')

lines2 = flip_sep(lines)
write_data(path_write, lines2)