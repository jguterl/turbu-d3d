# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
warnings.filterwarnings("ignore")

def plot_raw_data(file2load, **kwargs):
    '''  

    Parameters
    ----------
    file2load : string
        path of file to load
    **kwargs : field = list
        list of the field to load (example field = ['ne','te']) 
        if empty all fields are loaded : ne, ni, te, ti, ng, up, M
               plot = True/False
               if 0 no plot (by default plot=True)

    Returns
    -------
    output : dictionary
             Containing the fields mentioned in field input

    '''
    sim = np.load(file2load, allow_pickle=True).tolist()

    ee = 1.602e-19
    mi = 1.67e-27

    R = sim[3][0]['com.rm'][:,:,0]
    Z = sim[3][0]['com.zm'][:,:,0]

    plot = True
    for key, value in kwargs.items():
        if (key=='field'):
            field=value
        if (key=='plot'):
            plot=value
    output = {}
    try:
        
        for i in range(len(field)):
            count=0
            if (field[i]=="ne"):
                ne = sim[3][0]['bbb.ni'][:,:,0]
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,ne, cmap='hot', shading='auto', norm=colors.LogNorm(vmin=ne.min(), vmax=ne.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Electron density $[m^{-3}]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                count = count + 1
                output['ne'] = ne
            if (field[i]=="ne"):
                ni = sim[3][0]['bbb.ni'][:,:,0]
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,ni, cmap='hot', shading='auto', norm=colors.LogNorm(vmin=ne.min(), vmax=ne.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Electron density $[m^{-3}]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                    count = count + 1
                    output['ni'] = ni
            elif (field[i]=="ng"):
                ng = sim[3][0]['bbb.ng'][:,:,0]
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,ng, cmap='hot', shading='auto', norm=colors.LogNorm(vmin=ng.min(), vmax=ng.max()))
                    fig.colorbar(pcm, ax=ax[0], extend='max')
                
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Neutral density $[m^{-3}]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                    count = count + 1
                output['ng'] = ng
            elif (field[i]=='te'):
                te = sim[3][0]['bbb.te']/ee  
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,te, cmap='hot', shading='auto', norm=colors.Normalize(vmin=te.min(), vmax=te.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Electron temperature $[eV]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                count = count + 1
                output['te'] = te
            elif (field[i]=='ti'):
                ti = sim[3][0]['bbb.ti']/ee
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,ti, cmap='hot', shading='auto', norm=colors.Normalize(vmin=ti.min(), vmax=ti.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Ion temperature $[eV]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                count = count + 1
                output['ti'] = ti
            elif (field[i]=='M'):
                up = sim[3][0]['bbb.up'][:,:,0]
                M = up/np.sqrt(ee/mi*(te+ti))
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,M, cmap='hot', shading='auto', norm=colors.Normalize(vmin=M.min(), vmax=M.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Mach number')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                count = count + 1
                output['up'] = up
                output['M'] = M
            elif (field[i]=='up'):
                up = sim[3][0]['bbb.up'][:,:,0]
                if (plot==1):
                    fig, ax =plt.subplots(1)
                    im = ax.pcolormesh(R,Z,up, cmap='hot', shading='auto', norm=colors.Normalize(vmin=up.min(), vmax=up.max()))
                    fig.colorbar(im,ax=ax)
                    plt.title(r'Parallel plasma velocity $[m/s]$')
                    ax.set_xlabel('R [m]')
                    ax.set_ylabel('Z [m]')
                count = count + 1
                output['up'] = up
            if (count==0):
                print('Field name number', i, 'is invalid.')
    except:
        ne = sim[3][0]['bbb.ni'][:,:,0]
        output['ne'] = ne
        ni = sim[3][0]['bbb.ni'][:,:,0]
        output['ni'] = ni        
        ng = sim[3][0]['bbb.ng'][:,:,0]
        output['ng'] = ng
        te = sim[3][0]['bbb.te']/ee 
        output['te'] = te
        ti = sim[3][0]['bbb.ti']/ee
        output['ti'] = ti
        up = sim[3][0]['bbb.up'][:,:,0]
        output['up'] = up
        M = up/np.sqrt(ee/mi*(te+ti))
        output['M'] = M
        
        if (plot==1):
            fig1, ax = plt.subplots(1,2)
            im1 = ax[0].pcolormesh(R,Z,te,cmap='hot', shading='auto', norm=colors.Normalize(vmin=te.min(), vmax=te.max()))
            fig1.colorbar(im1, ax=ax[0])
            im2 = ax[1].pcolormesh(R,Z,ti,cmap='hot', shading='auto', norm=colors.Normalize(vmin=ti.min(), vmax=ti.max()))
            fig1.colorbar(im2, ax=ax[1])
            ax[0].title.set_text(r'Electron temperature [eV]')
            ax[1].title.set_text(r'Ion temperature [eV]')
            ax[0].set_xlabel('R [m]')
            ax[0].set_ylabel('Z [m]')
            ax[1].set_xlabel('R [m]')
            ax[1].set_ylabel('Z [m]')
            
            fig, ax = plt.subplots(1,2)
            im3 = ax[0].pcolormesh(R,Z,ne, cmap='hot', shading='auto', norm=colors.LogNorm(vmin=ne.min(), vmax=ne.max()))
            fig.colorbar(im3, ax=ax[0])
            im4 = ax[1].pcolormesh(R,Z,ng, cmap='hot', shading='auto', norm=colors.LogNorm(vmin=ng.min(), vmax=ng.max()))
            fig.colorbar(im4, ax=ax[1])
            ax[0].title.set_text(r'Electron density $[m^{-3}]$')
            ax[1].title.set_text(r'Neutral density $[m^{-3}]$')
            ax[0].set_xlabel('R [m]')
            ax[0].set_ylabel('Z [m]')
            ax[1].set_xlabel('R [m]')
            ax[1].set_ylabel('Z [m]')
        
            fig2, ax3 = plt.subplots(1)
            im5 = ax3.pcolormesh(R, Z, M, cmap='hot', shading='auto', norm=colors.Normalize(vmin=M.min(), vmax=M.max()))
            fig2.colorbar(im5, ax=ax3)
            ax3.title.set_text('Mach number')
            ax3.set_xlabel('R [m]')
            ax3.set_ylabel('Z [m]')
    output['R'] = R
    output['Z'] = Z
    return output

