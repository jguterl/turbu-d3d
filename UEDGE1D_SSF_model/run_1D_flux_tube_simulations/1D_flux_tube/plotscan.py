import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("/home/zml/uedge_new_github/UEDGE/pylib"))
import data
execfile('/home/zml/uedge_new_github/UEDGE/pylib/readpara.py')

read_geo(fn='./n0.5/')
ev = 1.6022e-19
nxo = True

# read in case
case = []
cdir = []
iyt = 1
for d in os.listdir('./'):
    link = './'+d
    if d[0]=='n' and os.path.isdir(link):
        cdir.append(link)
        case1 = data.data()
        read_all(case1,nx,ny,fn=link+'/',nxomi=nxo)
        calc(case1,nx,ny,nxomi=nxo)
        case1.ncore = float(d[1:])*1e19
        case1.tet = case1.te[nx,iyt]/ev
        case1.jsat = case1.ni[nx,iyt]*case1.ui[nx,iyt,0]*ev
        case.append(case1)

# reorder case
for ic in range(len(case)):
    for ic1 in range(ic+1,len(case)):
        if case[ic].ncore > case[ic1].ncore:
            casein = case[ic]
            case[ic] = case[ic1]
            case[ic1] = casein

# a typical jsat with increasing ncore
import matplotlib.pyplot as plt
ms = '5'
color = 'b'
# density scan
fig,ax = plt.subplots(2,1,figsize=(4,5))
ax1 = ax[0]
ax2 = ax[1]
ax2t = ax2.twinx()
for ic in range(len(case)):
    xn = case[ic].ncore
    tet = case[ic].tet
    jsat = case[ic].jsat
    ma = case[ic].ui[nx,iyt,0]/case[ic].cs[nx,iyt]
    ax1.plot(xn,jsat,marker = '.',markersize=ms,color = color)
    ax2.plot(xn,tet,marker = '.',markersize=ms,color = color)
    ax2t.plot(xn,ma,marker = '.',markersize=ms,color = 'C1')
ax1.set_ylabel('jsat')
ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax2.set_ylabel('Te',color=color)
ax2t.set_ylabel('M',color='C1')

# density profiles
xs = 1
xe = nx+1
lw = 1
lsty1 = '--'
lsty2 = 'dotted'
fig,ax = plt.subplots(4,2,figsize=(8,10),gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
fign,axn = plt.subplots(4,2,figsize=(8,10),gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[1,0]
ax4 = ax[1,1]
ax5 = ax[2,0]
ax6 = ax[2,1]
ax7 = ax[3,0]
ax8 = ax[3,1]
axn1 = axn[0,0]
axn2 = axn[0,1]
axn3 = axn[1,1]
axn4 = axn[1,0]
axn5 = axn[2,0]
axn6 = axn[2,1]
telmin = 100.
termax = 0.
termin = 2.5
mlmin = 100.
mrmax = 1.
eps = 1.e-3
mi = 1.6726e-27*2
#case.ix_sonic = np.zeros(len(case)) + nx+1
for ic in range(len(case)):
    xcs = case[ic].xcs[xs:xe]
    ne = case[ic].ni[xs:xe,iyt]
    te = case[ic].te[xs:xe,iyt]/ev
    tet = case[ic].tet
    up = (case[ic].ui[xs:xe,iyt,0]+case[ic].ui[xs-1:xe-1,iyt,0])/2
    #ma = case[ic].ui[xs:xe,iyt,0]/case[ic].cs[xs:xe,iyt]
    ma = case[ic].Mach[xs:xe,iyt]
    mt = case[ic].Mach[nx,iyt]
    fl = case[ic].fni[xs:xe,iyt,0]/case[ic].sx[xs:xe,iyt]
    sion = case[ic].sion[xs:xe,iyt]
    srec = case[ic].srec[xs:xe,iyt]
    #scx = case[ic].scx[xs:xe,iyt]
    scx = case[ic].smp2[xs:xe,iyt]
    vol = case[ic].vol[xs:xe,iyt]
    ti = case[ic].ti[xs:xe,iyt]/ev
    ptot = ne*(te*ev+ti*ev+mi*up**2)
    if tet > 2.5 and tet < telmin:
        icl = ic
        telmin = tet
    #if tet < 2.5 and tet > termax:
    if tet < 2.5 and termax == 0.:
        icr = ic
        termax = tet
    if mt > 1. and tet < termin:
        icr2 = ic
        termin = tet
    #if tet < 2.5 and mt <= 1.: continue
    ax1.plot(xcs,ne,marker = '',lw = lw,label=str(ic))
    ax2.plot(xcs,te,marker = '',lw = lw,label=str(ic))
    ax3.plot(xcs,up,marker = '',lw = lw,label=str(ic))
    ax4.plot(xcs,ma,marker = '',lw = lw,label=str(ic))
    ax5.plot(xcs,fl,marker = '',lw = lw,label=str(ic))
    ax6.plot(xcs,sion,marker = '',lw = lw,label=str(ic))
    ax7.plot(xcs,ptot,marker = '',lw = lw,label=str(ic))
    ax8.plot(xcs,scx,marker = '',lw = lw,label=str(ic))
    case[ic].ix_sonic = nx
    ix_sonic = nx
    for ix in range(1,nx-1):
        if (ma[ix] <=1. and ma[ix+1]>1.):
            case[ic].ix_sonic = ix
            ix_sonic = ix
    if ix_sonic != nx:
        ax2.axvline(x = 0.5*(xcs[ix_sonic]+xcs[ix_sonic+1]),lw = 0.5,linestyle=lsty1,color='k')
        ax4.axvline(x = 0.5*(xcs[ix_sonic]+xcs[ix_sonic+1]),lw = 0.5,linestyle=lsty1,color='k')
        ax6.axvline(x = 0.5*(xcs[ix_sonic]+xcs[ix_sonic+1]),lw = 0.5,linestyle=lsty1,color='k')
    # fign and axn
    axn1.plot(case[ic].ncore,sum(sion*vol)/sum(vol),marker = '.',markersize=ms)
    print('mach = ', case[ic].Mach[nx,iyt],case[ic].ug[nx,iyt,0]/case[ic].ui[nx,iyt,0])
    axn4.plot(xcs,case[ic].ng[xs:xe,iyt],lw = lw)
    axn5.plot(xcs,case[ic].ug[xs:xe,iyt,0],lw = lw)
axn1.set_ylabel(r'total ionization source')
ax1.plot(case[icl].xcs[xs:xe],case[icl].ni[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax1.plot(case[icr].xcs[xs:xe],case[icr].ni[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax1.set_ylabel(r'$n_i$')
ax2.plot(case[icl].xcs[xs:xe],case[icl].te[xs:xe,iyt]/ev,lw = 3, linestyle=lsty1,color = 'C0')
ax2.plot(case[icr].xcs[xs:xe],case[icr].te[xs:xe,iyt]/ev,lw = 3, linestyle=lsty1,color = 'C1')
ax2.set_ylabel(r'$T_e$')
ax3.plot(case[icl].xcs[xs:xe],case[icl].ui[xs:xe,iyt,0],lw = 3, linestyle=lsty1,color = 'C0')
ax3.plot(case[icr].xcs[xs:xe],case[icr].ui[xs:xe,iyt,0],lw = 3, linestyle=lsty1,color = 'C1')
ax3.set_ylabel(r'$u_\parallel$')
ax3.plot(case[icl].xcs[xs:xe],case[icl].ug[xs:xe,iyt,0],lw = 2, linestyle=lsty2,color = 'C0')
ax3.plot(case[icr].xcs[xs:xe],case[icr].ug[xs:xe,iyt,0],lw = 2, linestyle=lsty2,color = 'C1')
ax4.plot(case[icl].xcs[xs:xe],case[icl].Mach[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax4.plot(case[icr].xcs[xs:xe],case[icr].Mach[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax4.set_ylabel(r'$M$')
ax4.axhline(y=1.0,linestyle = '--', color = 'k')
ax5.plot(case[icl].xcs[xs:xe],case[icl].fni[xs:xe,iyt,0]/case[icl].sx[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax5.plot(case[icr].xcs[xs:xe],case[icr].fni[xs:xe,iyt,0]/case[icr].sx[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax5.set_ylabel(r'fni')
ax5.set_yscale('log')
ax6.plot(case[icl].xcs[xs:xe],case[icl].sion[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax6.plot(case[icr].xcs[xs:xe],case[icr].sion[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax6.set_ylabel(r'S_ion')
ax1.set_ylabel(r'$n_i$')
ax6.set_yscale('log')
#ax6.plot(case[icl].xcs[xs:xe],case[icl].srec[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
#ax6.plot(case[icr].xcs[xs:xe],case[icr].srec[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax7.plot(case[icl].xcs[xs:xe],case[icl].ni[xs:xe,iyt]*(case[icl].te[xs:xe,iyt]+case[icl].ti[xs:xe,iyt]+mi*case[icl].ui[xs:xe,iyt,0]**2),lw = 3, linestyle=lsty1,color = 'C0')
ax7.plot(case[icr].xcs[xs:xe],case[icr].ni[xs:xe,iyt]*(case[icr].te[xs:xe,iyt]+case[icr].ti[xs:xe,iyt]+mi*case[icr].ui[xs:xe,iyt,0]**2),lw = 3, linestyle=lsty1,color = 'C1')
ax7.set_ylabel(r'P tot')
'''
ax8.plot(case[icl].xcs[xs:xe],case[icl].scx[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax8.plot(case[icr].xcs[xs:xe],case[icr].scx[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax8.set_ylabel(r'S_cx')
ax8.set_yscale('log')
'''
ax8.plot(case[icl].xcs[xs:xe],case[icl].smp2[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
ax8.plot(case[icr].xcs[xs:xe],case[icr].smp2[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
ax8.set_ylabel(r'S_m')
#ax8.set_yscale('log')
axn1.axvline(x = case[icr].ncore,linestyle=lsty1,color = 'k')
#axn1.axvline(x = case[icr2].ncore,linestyle=lsty1,color = 'k')

# sub-super transition
  # various lambda
  #.. right case
lamp = case[icr].lambp[xs:xe,iyt]
lamt = case[icr].lambtei[xs:xe,iyt]/2.
lamv = -case[icr].lambvis[xs:xe,iyt]*case[icr].fM[xs:xe,iyt]
lamm = -case[icr].lambm[xs:xe,iyt]*case[icr].fM[xs:xe,iyt]
lamb = case[icr].lambb[xs:xe,iyt]*case[icr].fM[xs:xe,iyt]
lam_tot = lamp+lamt+lamv+lamm+lamb
P0,=axn2.plot(case[icr].xcs[xs:xe],lamp,lw = 1,label='lamb_Sp')
P1,=axn2.plot(case[icr].xcs[xs:xe],lamt,lw = 1,label='lamb_T')
P2,=axn2.plot(case[icr].xcs[xs:xe],lamv,lw = 1,label='lamb_vis')
P3,=axn2.plot(case[icr].xcs[xs:xe],lamm,lw = 1,label='lamb_Sm')
P4,=axn2.plot(case[icr].xcs[xs:xe],lamb,lw = 1,label='lamb_B')
P5,=axn2.plot(case[icr].xcs[xs:xe],lam_tot,lw = 3,linestyle=lsty1,color = 'C1',label='lamb_tot')
axn2.axvline(x = 0.5*(case[icr].xcs[case[icr].ix_sonic]+case[icr].xcs[case[icr].ix_sonic+1]),linestyle=lsty1,color = 'k')
axn2.legend()
  #.. left case
lamp = case[icl].lambp[xs:xe,iyt]
lamt = case[icl].lambtei[xs:xe,iyt]/2.
lamv = -case[icl].lambvis[xs:xe,iyt]*case[icl].fM[xs:xe,iyt]
lamm = -case[icl].lambm[xs:xe,iyt]*case[icl].fM[xs:xe,iyt]
lamb = case[icl].lambb[xs:xe,iyt]*case[icl].fM[xs:xe,iyt]
lam_tot = lamp+lamt+lamv+lamm+lamb
axn2.plot(case[icl].xcs[xs:xe],lamp,lw = 1,label='lamb_Sp',linestyle=lsty1,color= P0.get_color())
axn2.plot(case[icl].xcs[xs:xe],lamt,lw = 1,label='lamb_T',linestyle=lsty1,color= P1.get_color())
axn2.plot(case[icl].xcs[xs:xe],lamv,lw = 1,label='lamb_vis',linestyle=lsty1,color= P2.get_color())
axn2.plot(case[icl].xcs[xs:xe],lamm,lw = 1,label='lamb_Sm',linestyle=lsty1,color= P3.get_color())
axn2.plot(case[icl].xcs[xs:xe],lamb,lw = 1,label='lamb_B',linestyle=lsty1,color= P4.get_color())
axn2.plot(case[icl].xcs[xs:xe],lam_tot,lw = 3,linestyle=lsty1,color = 'C0',label='lamb_tot')

ic1 = icr
lamm = -case[ic1].lambm[xs:xe,iyt]*case[ic1].fM[xs:xe,iyt]
lamm_ion = -case[ic1].lambm_ion[xs:xe,iyt]*case[ic1].fM[xs:xe,iyt]
lamm_rec = -case[ic1].lambm_rec[xs:xe,iyt]*case[ic1].fM[xs:xe,iyt]
lamm_cx = -case[ic1].lambm_cx[xs:xe,iyt]*case[ic1].fM[xs:xe,iyt]
axn3.plot(case[ic1].xcs[xs:xe],lamm,lw = 3,label='lamb_Sm',linestyle=lsty1,color= P3.get_color())
axn3.plot(case[ic1].xcs[xs:xe],lamm_ion,lw = 1,label='lamb_Sm^ion',linestyle=lsty1)
axn3.plot(case[ic1].xcs[xs:xe],lamm_rec,lw = 1,label='lamb_Sm^rec',linestyle=lsty1)
axn3.plot(case[ic1].xcs[xs:xe],lamm_cx,lw = 1,label='lamb_Sm^cx',linestyle=lsty1)
axn3.legend()

axn4.plot(case[icl].xcs[xs:xe],case[icl].ng[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C0')
axn4.plot(case[icr].xcs[xs:xe],case[icr].ng[xs:xe,iyt],lw = 3, linestyle=lsty1,color = 'C1')
axn4.set_yscale('log')

axn5.plot(case[icl].xcs[xs:xe],case[icl].ug[xs:xe,iyt,0],lw = 3, linestyle=lsty1,color = 'C0')
axn5.plot(case[icr].xcs[xs:xe],case[icr].ug[xs:xe,iyt,0],lw = 3, linestyle=lsty1,color = 'C1')

plt.show()
