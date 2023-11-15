import sys,os
import numpy as np
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text.latex',preamble=r'\usepackage{amssymb}')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import griddata

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

from qcdlib.aux import AUX

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

import warnings
warnings.filterwarnings("ignore")


cmap = matplotlib.cm.get_cmap('plasma')

def gen_D(wdir,Q2=None):
    
    FLAV=[]
    FLAV.append('u')
    FLAV.append('s')
    FLAV.append('c')
    FLAV.append('b')
    FLAV.append('g')

    print('\ngenerating diff at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if Q2==None: Q2 = conf['Q20']

    conf['SofferBound'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False,load_lhapdf=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    diff=conf['diffpippim']
    #--setup kinematics
    NZ,NM = 20,50
    Z=np.linspace(0.19,0.99,NZ)
    M=np.linspace(0.28,2.05,NM)
    shape = (NM,NZ)
    L = NM*NZ

    Zgrid, Mgrid = np.meshgrid(Z,M)
    Zgrid = Zgrid.flatten()
    Mgrid = Mgrid.flatten()


    D = {}

    cnt = 0
    for par in replicas:
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])
        lprint('%d/%d'%(cnt+1,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in D: D[flav] = []
            func = diff.get_D(Zgrid,Mgrid,Q2*np.ones(L),flav,evolve=True)
            D[flav].append(func)
 
        cnt+=1

   

    print() 
    checkdir('%s/data'%wdir)

    filename='%s/data/diff-Q2=%3.5f.dat'%(wdir,Q2)

    save({'Zgrid':Zgrid,'Mgrid':Mgrid,'Q2':Q2,'D':D},filename)
    print ('Saving data to %s'%filename)

def plot_D(wdir,Q2=None,mode=1):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=2,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*6,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax13 = py.subplot(nrows,ncols,3)
    ax21 = py.subplot(nrows,ncols,4)
    ax22 = py.subplot(nrows,ncols,5)
    ax23 = py.subplot(nrows,ncols,6)

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']

    conf['SofferBound'] = False
    RESMAN(datasets=False,load_lhapdf=False)
    diff = conf['diffpippim']

    M0 = diff.M0['u']

    #--load data if it exists
    filename='%s/data/diff-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_D(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)

    Zfixed = [0.25,0.45,0.65]
    Mfixed = [0.40,1.00,1.60]

    N = 100
    Z = np.linspace(0.20,0.99,N)
    M = np.linspace(0.28,2.05,N)

    D     = data['D']
    Zgrid = data['Zgrid']
    Mgrid = data['Mgrid']

    for flav in D:
        diff = D[flav]
        if flav=='u': color = 'purple'
        if flav=='s': color = 'green'
        if flav=='c': color = 'orange'
        if flav=='b': color = 'pink'
        if flav=='g': color = 'blue'

        #if flav!='b': alpha = 0.0
        #else: alpha = 0.3

        #--plot with z fixed
        for i in range(len(Zfixed)):
            if i == 0: ax = ax11
            if i == 1: ax = ax12
            if i == 2: ax = ax13
            zfixed = Zfixed[i]*np.ones(N)

            if mode==0:
                for j in range(len(D[flav])):
                    result = griddata((Mgrid,Zgrid),D[flav][j],(M,zfixed),fill_value=0,method='cubic',rescale=True)
                    hand[flav] ,= ax.plot(M,result,color=color,alpha=0.3,zorder=1)
              

            if mode==1:
                mean = np.mean(diff,axis=0)
                std  = np.std (diff,axis=0)
                mean = griddata((Mgrid,Zgrid),mean,(M,zfixed),fill_value=0,method='cubic',rescale=True)
                std  = griddata((Mgrid,Zgrid),std ,(M,zfixed),fill_value=0,method='cubic',rescale=True)
                hand[flav]  = ax.fill_between(M,(mean-std),(mean+std),color=color,alpha=0.8,zorder=1)

        #--plot with M fixed
        for i in range(len(Mfixed)):
            if i == 0: ax = ax21
            if i == 1: ax = ax22
            if i == 2: ax = ax23
            mfixed = Mfixed[i]*np.ones(N)

            if mode==0:
                for j in range(len(D[flav])):
                    result = griddata((Mgrid,Zgrid),D[flav][j],(mfixed,Z),fill_value=0,method='cubic',rescale=True)
                    hand[flav] ,= ax.plot(Z,result,color=color,alpha=0.3,zorder=1)
              

            if mode==1:
                mean = np.mean(diff,axis=0)
                std  = np.std (diff,axis=0)
                mean = griddata((Mgrid,Zgrid),mean,(mfixed,Z),fill_value=0,method='cubic',rescale=True)
                std  = griddata((Mgrid,Zgrid),std ,(mfixed,Z),fill_value=0,method='cubic',rescale=True)

                hand[flav]  = ax.fill_between(Z,(mean-std),(mean+std),color=color,alpha=0.8,zorder=1)










    ##########################
    #--SET UP PLOT
    ##########################

    for ax in [ax11,ax12,ax13]:

          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=8)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=4)

          minorLocator = MultipleLocator(0.1)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(1.0)
          #ax.yaxis.set_minor_locator(minorLocator)

          ax.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=25)
          ax.axhline(0,0,1,color='black',alpha=1.0)


    ax11.set_xlim(0.28,1.3)
    ax11.set_xticks([0.4,0.6,0.8,1.0,1.2])

    ax12.set_xlim(0.28,2.0)
    ax12.set_xticks([0.4,0.8,1.2,1.6])

    ax13.set_xlim(0.28,2.0)
    ax13.set_xticks([0.4,0.8,1.2,1.6])

    ax21.set_xlim(0.2,1.0)
    ax21.set_xticks([0.4,0.6,0.8])

    ax22.set_xlim(0.2,1.0)
    ax22.set_xticks([0.4,0.6,0.8])

    ax23.set_xlim(0.4,1.0)
    ax23.set_xticks([0.6,0.8])


    for ax in [ax21,ax22,ax23]:

          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=8)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=4)

          minorLocator = MultipleLocator(0.1)
          ax.xaxis.set_minor_locator(minorLocator)

          ax.set_xlabel(r'\boldmath$z$',size=25)
          ax.axhline(0,0,1,color='black',alpha=1.0)


    #--plot grid points
    for m in M0:
        for ax in [ax11,ax12,ax13]:
            ax.axvline(m,0,1,alpha=0.1,ls='--',color='black')

    ax11.set_ylabel(r'\boldmath$D_1^q(z,M_h) [{\rm GeV}^{-1}]$',size=25)
    ax21.set_ylabel(r'\boldmath$D_1^q(z,M_h) [{\rm GeV}^{-1}]$',size=25)

    #ax12.tick_params(labelleft=False)
    #ax13.tick_params(labelleft=False)
    #ax22.tick_params(labelleft=False)
    #ax23.tick_params(labelleft=False)

    ax11.set_ylim(-1.0,5.0)
    ax12.set_ylim(-1.0,3.1)
    ax13.set_ylim(-0.5,1.3)
    ax21.set_ylim(-1.0,5.0)
    ax22.set_ylim(-0.5,2.5)
    ax23.set_ylim(-0.5,0.3)

    #ax11.set_yticks([0,2,4,6,8,10])
    #ax12.set_yticks([0,2,4,6,8,10])
    #ax13.set_yticks([0,2,4,6,8,10])
    #ax21.set_yticks([0,2,4,6,8,10])
    #ax22.set_yticks([0,2,4,6,8,10])
    #ax23.set_yticks([0,2,4,6,8,10])

    minorLocator = MultipleLocator(0.25)
    ax11.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax12.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax13.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax21.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax22.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.10)
    ax23.yaxis.set_minor_locator(minorLocator)




    ax22.text(0.35,0.75,r'\boldmath$Q^2=%s~{\rm GeV}^2$'%Q2, transform=ax22.transAxes,size=20)

    ax11.text(0.40,0.85,r'$z=0.25$', transform=ax11.transAxes,size=20)
    ax12.text(0.40,0.85,r'$z=0.45$', transform=ax12.transAxes,size=20)
    ax13.text(0.40,0.85,r'$z=0.65$', transform=ax13.transAxes,size=20)

    ax21.text(0.35,0.85,r'$M_h=0.4~{\rm GeV}$', transform=ax21.transAxes,size=20)
    ax22.text(0.35,0.85,r'$M_h=1.0 ~{\rm GeV}$', transform=ax22.transAxes,size=20)
    ax23.text(0.35,0.85,r'$M_h=1.6 ~{\rm GeV}$', transform=ax23.transAxes,size=20)

    #ax13.text(0.68,0.75,r'\boldmath$5 \times D_1^q$', transform=ax13.transAxes,size=30)
    #ax23.text(0.68,0.75,r'\boldmath$5 \times D_1^q$', transform=ax23.transAxes,size=30)

    handles = []
    handles.append(hand['u'])
    handles.append(hand['s'])
    handles.append(hand['c'])

    labels = []
    labels.append(r'\boldmath$u$')
    labels.append(r'\boldmath$s$')
    labels.append(r'\boldmath$c$')

    ax11.legend(handles,labels,loc='upper right',fontsize=30,frameon=0,handletextpad=0.3,handlelength=0.9,ncol=1,columnspacing=1.0)

    handles = []
    handles.append(hand['b'])
    handles.append(hand['g'])

    labels = []
    labels.append(r'\boldmath$b$')
    labels.append(r'\boldmath$g$')

    ax12.legend(handles,labels,loc='upper right',fontsize=30,frameon=0,handletextpad=0.3,handlelength=0.9,ncol=1,columnspacing=1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0.3,wspace=0.2)

    filename = '%s/gallery/diffs-%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax13.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    ax23.set_rasterized(True)

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)


def gen_H(wdir,Q2=None):
    
    FLAV=[]
    FLAV.append('u')

    print('\ngenerating tdiff at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if Q2==None: Q2 = conf['Q20']

    conf['SofferBound'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False,load_lhapdf=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    diff=conf['tdiffpippim']
    #--setup kinematics
    NZ,NM = 20,50
    Z=np.linspace(0.19,0.99,NZ)
    M=np.linspace(0.28,2.05,NM)
    shape = (NM,NZ)
    L = NM*NZ

    Zgrid, Mgrid = np.meshgrid(Z,M)
    Zgrid = Zgrid.flatten()
    Mgrid = Mgrid.flatten()


    H = {}

    cnt = 0
    for par in replicas:
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])
        lprint('%d/%d'%(cnt+1,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in H: H[flav] = []
            func = diff.get_H(Zgrid,Mgrid,Q2*np.ones(L),flav,evolve=True)
            H[flav].append(func)
 
        cnt+=1

   

    print() 
    checkdir('%s/data'%wdir)

    filename='%s/data/tdiff-Q2=%3.5f.dat'%(wdir,Q2)

    save({'Zgrid':Zgrid,'Mgrid':Mgrid,'Q2':Q2,'H':H},filename)
    print ('Saving data to %s'%filename)

def plot_H(wdir,Q2=None,mode=1):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']

    conf['SofferBound'] = False
    RESMAN(datasets=False,load_lhapdf=False)
    tdiff = conf['tdiffpippim']

    M0 = tdiff.M0['u']
    nrows,ncols=2,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*6,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax13 = py.subplot(nrows,ncols,3)
    ax21 = py.subplot(nrows,ncols,4)
    ax22 = py.subplot(nrows,ncols,5)
    ax23 = py.subplot(nrows,ncols,6)

    #--load data if it exists
    filename='%s/data/tdiff-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_H(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)

    Zfixed = [0.25,0.45,0.65]
    Mfixed = [0.40,1.00,1.60]

    N = 100
    Z = np.linspace(0.20,0.99,N)
    M = np.linspace(0.28,2.05,N)

    H     = data['H']
    Zgrid = data['Zgrid']
    Mgrid = data['Mgrid']

    for flav in H:
        diff = H[flav]
        if flav=='u': color = 'purple'

        #--plot with z fixed
        for i in range(len(Zfixed)):
            if i == 0: ax = ax11
            if i == 1: ax = ax12
            if i == 2: ax = ax13
            zfixed = Zfixed[i]*np.ones(N)

            if mode==0:
                for j in range(len(H[flav])):
                    result = griddata((Mgrid,Zgrid),H[flav][j],(M,zfixed),fill_value=0,method='cubic',rescale=True)
                    hand[flav] ,= ax.plot(M,result,color=color,alpha=0.3,zorder=1)
              

            if mode==1:
                mean = np.mean(diff,axis=0)
                std  = np.std (diff,axis=0)
                mean = griddata((Mgrid,Zgrid),mean,(M,zfixed),fill_value=0,method='cubic',rescale=True)
                std  = griddata((Mgrid,Zgrid),std ,(M,zfixed),fill_value=0,method='cubic',rescale=True)
                hand[flav]  = ax.fill_between(M,(mean-std),(mean+std),color=color,alpha=0.8,zorder=1)

        #--plot with M fixed
        for i in range(len(Mfixed)):
            if i == 0: ax = ax21
            if i == 1: ax = ax22
            if i == 2: ax = ax23
            mfixed = Mfixed[i]*np.ones(N)

            if mode==0:
                for j in range(len(H[flav])):
                    result = griddata((Mgrid,Zgrid),H[flav][j],(mfixed,Z),fill_value=0,method='cubic',rescale=True)
                    hand[flav] ,= ax.plot(Z,result,color=color,alpha=0.3,zorder=1)
              

            if mode==1:
                mean = np.mean(diff,axis=0)
                std  = np.std (diff,axis=0)
                mean = griddata((Mgrid,Zgrid),mean,(mfixed,Z),fill_value=0,method='cubic',rescale=True)
                std  = griddata((Mgrid,Zgrid),std ,(mfixed,Z),fill_value=0,method='cubic',rescale=True)

                hand[flav]  = ax.fill_between(Z,(mean-std),(mean+std),color=color,alpha=0.8,zorder=1)



    #--plot D1 boundary
    #--load data if it exists
    filename='%s/data/diff-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_D(wdir,Q2,force=True)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    D     = data['D']
    Zgrid = data['Zgrid']
    Mgrid = data['Mgrid']

    for flav in D:
        diff = D[flav]
        if flav!='u': continue

        #--plot with z fixed
        for i in range(len(Zfixed)):
            if i == 0: ax = ax11
            if i == 1: ax = ax12
            if i == 2: ax = ax13
            zfixed = Zfixed[i]*np.ones(N)

            mean = np.mean(diff,axis=0)
            mean = griddata((Mgrid,Zgrid),mean,(M,zfixed),fill_value=0,method='cubic',rescale=True)

            bound = ax.plot(M,-mean,color='black',alpha=0.8,zorder=1)

        #--plot with M fixed
        for i in range(len(Mfixed)):
            if i == 0: ax = ax21
            if i == 1: ax = ax22
            if i == 2: ax = ax23
            mfixed = Mfixed[i]*np.ones(N)

            mean = np.mean(diff,axis=0)
            mean = griddata((Mgrid,Zgrid),mean,(mfixed,Z),fill_value=0,method='cubic',rescale=True)

            bound = ax.plot(Z,-mean,color='black',alpha=0.8,zorder=1)



    ##########################
    #--SET UP PLOT
    ##########################

    for ax in [ax11,ax12,ax13]:
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=8)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=4)

          minorLocator = MultipleLocator(0.1)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(1.0)

          ax.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=25)
    
    ax11.set_xlim(0.28,1.3)
    ax11.set_xticks([0.4,0.6,0.8,1.0,1.2])

    ax12.set_xlim(0.28,2.0)
    ax12.set_xticks([0.4,0.8,1.2,1.6])

    ax13.set_xlim(0.28,2.0)
    ax13.set_xticks([0.4,0.8,1.2,1.6])

    ax21.set_xlim(0.2,1.0)
    ax21.set_xticks([0.4,0.6,0.8])

    ax22.set_xlim(0.2,1.0)
    ax22.set_xticks([0.4,0.6,0.8])

    ax23.set_xlim(0.4,1.0)
    ax23.set_xticks([0.6,0.8])


    for ax in [ax21,ax22,ax23]:

          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=8)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=4)

          minorLocator = MultipleLocator(0.1)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(1.0)
          #ax.yaxis.set_minor_locator(minorLocator)

          ax.set_xlabel(r'\boldmath$z$',size=25)

    #--plot grid points
    for m in M0:
        for ax in [ax11,ax12,ax13]:
            ax.axvline(m,0,1,alpha=0.1,ls='--',color='black')

    ax11.set_ylabel(r'\boldmath$H_1^{q}(z,M_h) [{\rm GeV}^{-1}]$',size=25)
    ax21.set_ylabel(r'\boldmath$H_1^{q}(z,M_h) [{\rm GeV}^{-1}]$',size=25)

    #ax11.set_ylim(0,2.2)
    #ax12.set_ylim(0,1.1)
    #ax13.set_ylim(0,0.5)
    #ax21.set_ylim(0,1.2)
    #ax22.set_ylim(0,2.1)
    #ax23.set_ylim(0,1.5)
    ax11.set_ylim(-1.5,0.0)
    ax12.set_ylim(-0.8,0.0)
    ax13.set_ylim(-0.4,0.0)
    ax21.set_ylim(-0.8,0.0)
    ax22.set_ylim(-1.1,0.0)
    ax23.set_ylim(-0.2,0.0)


    minorLocator = MultipleLocator(0.25)
    ax11.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax12.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.05)
    ax13.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax21.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.25)
    ax22.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.05)
    ax23.yaxis.set_minor_locator(minorLocator)


    ax22.text(0.30,0.20,r'\boldmath$Q^2=%s~{\rm GeV}^2$'%Q2, transform=ax22.transAxes,size=20)

    ax11.text(0.40,0.05,r'$z=0.25$', transform=ax11.transAxes,size=20)
    ax12.text(0.40,0.05,r'$z=0.45$', transform=ax12.transAxes,size=20)
    ax13.text(0.40,0.05,r'$z=0.65$', transform=ax13.transAxes,size=20)

    ax21.text(0.35,0.05,r'$M_h=0.4~{\rm GeV}$', transform=ax21.transAxes,size=20)
    ax22.text(0.35,0.05,r'$M_h=1.0~{\rm GeV}$', transform=ax22.transAxes,size=20)
    ax23.text(0.35,0.05,r'$M_h=1.6~{\rm GeV}$', transform=ax23.transAxes,size=20)

    #ax13.text(0.68,0.75,r'\boldmath$5 \times D_1^q$', transform=ax13.transAxes,size=30)
    #ax23.text(0.68,0.75,r'\boldmath$5 \times D_1^q$', transform=ax23.transAxes,size=30)

    handles = []
    handles.append(hand['u'])

    labels = []
    labels.append(r'\boldmath$u$')

    ax11.legend(handles,labels,loc='lower right',fontsize=30,frameon=0,handletextpad=0.3,handlelength=0.9,ncol=1,columnspacing=1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0.3,wspace=0.2)

    filename = '%s/gallery/tdiffs-%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax13.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    ax23.set_rasterized(True)

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)




#--conditional probabilities
def gen_D_Mh_given_z(wdir,Q2=None,force=False):
    
    FLAV=[]
    FLAV.append('u')
    FLAV.append('s')
    FLAV.append('c')
    FLAV.append('b')
    FLAV.append('g')

    print('\ngenerating diff expectation (Mh given z) at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if Q2==None: Q2 = conf['Q20']

    if force==False:
        if 'diffpippim' not in conf['steps'][istep]['active distributions']:
            #if 'diffpippim' not in conf['steps'][istep]['passive distributions']:
            print('diffpippim is not an active or passive distribution')
            return 

    conf['SofferBound'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False,load_lhapdf=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    diff=conf['diffpippim']
    #--setup kinematics
    NZ,NM = 40,50
    #--Gaussian quadrature integration for Mh
    xg, wg = np.polynomial.legendre.leggauss(NM)
    Mmin,Mmax = 0.28,2.00
    Z=np.linspace(0.19,0.99,NZ)
    #M=np.linspace(0.28,2.05,NM)
    M = 0.5*(Mmax-Mmin)*xg + 0.5*(Mmax + Mmin)
    shape = (NM,NZ)
    L = NM*NZ

    Zgrid, Mgrid = np.meshgrid(Z,M)
    Zgrid = Zgrid.flatten()
    Mgrid = Mgrid.flatten()

    Mjac = 0.5*(Mmax - Mmin)
    Mbin = Mmax - Mmin


    D = {}

    cnt = 0
    for par in replicas:
        #if cnt>5: continue
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])
        lprint('%d/%d'%(cnt+1,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in D: D[flav] = []
            func = diff.get_D(Zgrid,Mgrid,Q2*np.ones(L),flav,evolve=True)

            num = Mgrid*func
            den = func

            num = np.reshape(num, shape).T
            den = np.reshape(den, shape).T

            num = np.sum(wg*num,axis=1)*Mjac/Mbin
            den = np.sum(wg*den,axis=1)*Mjac/Mbin

            result = num/den

            #print(flav)
            #print(num)

            D[flav].append(result)

        cnt+=1

   

    print() 
    checkdir('%s/data'%wdir)

    filename='%s/data/diff-Mh-given-z-Q2=%3.5f.dat'%(wdir,Q2)

    save({'Z':Z,'Mmin':Mmin,'Mmax':Mmax,'Q2':Q2,'D':D},filename)
    print ('Saving data to %s'%filename)

def gen_D_z_given_Mh(wdir,Q2=None,force=False):
    
    FLAV=[]
    FLAV.append('u')
    FLAV.append('s')
    FLAV.append('c')
    FLAV.append('b')
    FLAV.append('g')

    print('\ngenerating diff expectation (z given Mh) at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if Q2==None: Q2 = conf['Q20']

    if force==False:
        if 'diffpippim' not in conf['steps'][istep]['active distributions']:
            #if 'diffpippim' not in conf['steps'][istep]['passive distributions']:
            print('diffpippim is not an active or passive distribution')
            return 

    conf['SofferBound'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False,load_lhapdf=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    diff=conf['diffpippim']
    #--setup kinematics
    NZ,NM = 40,50
    #--Gaussian quadrature integration for Mh
    xg, wg = np.polynomial.legendre.leggauss(NZ)
    Zmin,Zmax = 0.25,1.00
    M=np.linspace(0.28,2.05,NM)
    Z = 0.5*(Zmax-Zmin)*xg + 0.5*(Zmax + Zmin)
    shape = (NM,NZ)
    L = NM*NZ

    Zgrid, Mgrid = np.meshgrid(Z,M)
    Zgrid = Zgrid.flatten()
    Mgrid = Mgrid.flatten()

    Zjac = 0.5*(Zmax - Zmin)
    Zbin = Zmax - Zmin

    D = {}

    cnt = 0
    for par in replicas:
        #if cnt>5: continue
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])
        lprint('%d/%d'%(cnt+1,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in D: D[flav] = []
            func = diff.get_D(Zgrid,Mgrid,Q2*np.ones(L),flav,evolve=True)

            num = Zgrid*func
            den = func


            num = np.reshape(num, shape)
            den = np.reshape(den, shape)

            num = np.sum(wg*num,axis=1)*Zjac/Zbin
            den = np.sum(wg*den,axis=1)*Zjac/Zbin

            result = num/den

            D[flav].append(result)

        cnt+=1

   

    print() 
    checkdir('%s/data'%wdir)

    filename='%s/data/diff-z-given-Mh-Q2=%3.5f.dat'%(wdir,Q2)

    save({'M':M,'Zmin':Zmin,'Zmax':Zmax,'Q2':Q2,'D':D},filename)
    print ('Saving data to %s'%filename)























