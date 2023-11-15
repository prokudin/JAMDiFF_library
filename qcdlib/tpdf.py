#!/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from tools.tools import checkdir
import pylab as py
from scipy.special import gamma
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import numpy as np
import qcdlib.alphaS
from qcdlib.aux import AUX
from qcdlib.dglap import DGLAP
from qcdlib.kernels import KERNELS
from qcdlib.mellin import MELLIN
from tools.config import conf
#import lhapdf

class TPDF():

    def __init__(self, func, mellin=None, shape = 'nderiv'):

        self.func = func
        if func in ['tpdf']:        spl = 'trans'
        if func in ['twist3pdf3']:  spl = 'upol'
        if func in ['sivers']:      spl = 'sivers'

        self.spl=spl
        self.shape = shape
        self.Q20=conf['Q20']
        self.mc2=conf['aux'].mc2
        self.mb2=conf['aux'].mb2

        if 'no_evolve' in conf: self.no_evolve = conf['no_evolve']
        else:                   self.no_evolve = False

        if func in ['tpdf']:
            if 'lhapdf_pdf'   in conf: self.lhapdf_pdf   = conf['lhapdf_pdf']
            else:                      self.lhapdf_pdf   = 'JAM22-PDF_proton_nlo'

            if 'lhapdf_ppdf'  in conf: self.lhapdf_ppdf  = conf['lhapdf_ppdf']
            else:                      self.lhapdf_ppdf  = 'JAM22-PPDF_proton_nlo'

            #--setup PDFs and PPDFs for SB
            if 'SofferBound' in conf and conf['SofferBound']:
                if mellin==None:
                    self.X = np.linspace(0.001,0.99,100)
                    if 'LHAPDF:PDF'  in conf: self.PDF   = conf['LHAPDF:PDF'] 
                    if 'LHAPDF:PPDF' in conf: self.PPDF  = conf['LHAPDF:PPDF'] 

                    #self.setup_SB()


        if mellin==None:
            self.kernel=KERNELS(conf['mellin'],self.spl)
            if 'mode' in conf: mode=conf['mode']
            else: mode='truncated'
            self.dglap=DGLAP(conf['mellin'],conf['alphaS'],self.kernel,mode,conf['order'])
            self.mellin=conf['mellin']
        else:
            self.kernel=KERNELS(mellin,self.spl)
            if 'mode' in conf: mode=conf['mode']
            else: mode='truncated'
            self.dglap=DGLAP(mellin,conf['alphaS'],self.kernel,mode,conf['order'])
            self.mellin=mellin

        if '%s_choice'%func in conf:
            self.choice = conf['%s_choice'%func]
        else:
            self.choice = 'valence'

        if func in ['tpdf']:
            if 'db factor' in conf:
                self.dbfactor = conf['db factor']
            else:
                self.dbfactor = None
        else:
            self.dbfactor = None


        self.set_params()
        self.setup()

    def setup_SB(self,Q2=None):

        print('Setting up Soffer Bound...')

        flavs = ['u','d','s','c','ub','db','sb','cb','g','uv','dv']

        PDF  = self.PDF
        PPDF = self.PPDF

        X   = self.X
        if Q2==None: Q2  = conf['Q20']
        self.SB = {}
        self.SB['mean'] = {}
        self.SB['std']  = {}
        self.SB['mean'] = {_:np.zeros(len(X)) for _ in flavs}
        self.SB['std']  = {_:np.zeros(len(X)) for _ in flavs}

        for i in range(len(X)):
            self.SB['mean']['u'] [i] = 0.5*np.mean([PDF[k].xfxQ2( 2,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['mean']['d'] [i] = 0.5*np.mean([PDF[k].xfxQ2( 1,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['mean']['ub'][i] = 0.5*np.mean([PDF[k].xfxQ2(-2,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['mean']['db'][i] = 0.5*np.mean([PDF[k].xfxQ2(-1,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['mean']['uv'][i] = 0.5*np.mean([PDF[k].xfxQ2( 2,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 2,X[i],Q2)/X[i]\
                                                   +PDF[k].xfxQ2(-2,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['mean']['dv'][i] = 0.5*np.mean([PDF[k].xfxQ2( 1,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 1,X[i],Q2)/X[i]\
                                                   +PDF[k].xfxQ2(-1,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['u'] [i] = 0.5*np.std ([PDF[k].xfxQ2( 2,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['d'] [i] = 0.5*np.std ([PDF[k].xfxQ2( 1,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['ub'][i] = 0.5*np.std ([PDF[k].xfxQ2(-2,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['db'][i] = 0.5*np.std ([PDF[k].xfxQ2(-1,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['uv'][i] = 0.5*np.std ([PDF[k].xfxQ2( 2,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 2,X[i],Q2)/X[i]\
                                                   +PDF[k].xfxQ2(-2,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-2,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            self.SB['std'] ['dv'][i] = 0.5*np.std ([PDF[k].xfxQ2( 1,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 1,X[i],Q2)/X[i]\
                                                   +PDF[k].xfxQ2(-1,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-1,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['mean']['g'] [i] = 0.5*np.mean([PDF[k].xfxQ2(21,X[i],Q2)/X[i] + PPDF[k].xfxQ2(21,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['mean']['s'] [i] = 0.5*np.mean([PDF[k].xfxQ2( 3,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 3,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['mean']['c'] [i] = 0.5*np.mean([PDF[k].xfxQ2( 4,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 4,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['mean']['sb'][i] = 0.5*np.mean([PDF[k].xfxQ2(-3,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-3,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['mean']['cb'][i] = 0.5*np.mean([PDF[k].xfxQ2(-4,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-4,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['std'] ['g'] [i] = 0.5*np.std ([PDF[k].xfxQ2(21,X[i],Q2)/X[i] + PPDF[k].xfxQ2(21,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['std'] ['s'] [i] = 0.5*np.std ([PDF[k].xfxQ2( 3,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 3,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['std'] ['c'] [i] = 0.5*np.std ([PDF[k].xfxQ2( 4,X[i],Q2)/X[i] + PPDF[k].xfxQ2( 4,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['std'] ['sb'][i] = 0.5*np.std ([PDF[k].xfxQ2(-3,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-3,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)
            #self.SB['std'] ['cb'][i] = 0.5*np.std ([PDF[k].xfxQ2(-4,X[i],Q2)/X[i] + PPDF[k].xfxQ2(-4,X[i],Q2)/X[i] for k in range(len(PDF))],axis=0)

    def set_params(self):

        if self.choice=='valence':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}

            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['uv']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['dv']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['c']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['b']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['cb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['bb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','uv','dv','s','c','b','ub','db','sb','cb','bb']
            self.PAR  = ['N', 'a', 'b', 'c', 'd','N1', 'a1', 'b1', 'c1', 'd1','N2', 'a2', 'b2', 'c2', 'd2']



        if self.choice=='basic':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}
            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['u']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['d']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','u','d','s','ub','db','sb']
            self.PAR  = ['N0', 'a0', 'b0', 'c0', 'd0','N1', 'a1', 'b1', 'c1', 'd1','N2', 'a2', 'b2', 'c2', 'd2']

        if self.choice=='JAM3D':
            """
            f(x) = N0 * x**a0 * (1-x)**b0 * (1 + N1 * x*a1 * (1-x)**b1)
            """
            self.params = {}
            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['u']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['d']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0])

            self.FLAV = ['g','u','d','s','ub','db','sb']
            self.PAR  = ['N0', 'a0', 'b0','N1', 'a1', 'b1']

        #--widths
        if self.spl in ['trans','sivers']:
            self._widths1_uv  = 0.3
            self._widths1_dv  = 0.3
            self._widths1_sea = 0.3

        # internal
        self.widths1 = np.ones(11)

    def setup(self):

        moms = {}

        if self.choice=='valence': 
            u = self.get_moments('uv') + self.get_moments('ub')
            d = self.get_moments('dv') + self.get_moments('db')
        elif self.choice in ['basic','JAM3D']:
            u = self.get_moments('u')
            d = self.get_moments('d')

        #--have ub and db independent
        if self.dbfactor==None:
            moms['g']  = self.get_moments('g')
            moms['up'] = u                     + self.get_moments('ub')
            moms['dp'] = d                     + self.get_moments('db')
            moms['sp'] = self.get_moments('s') + self.get_moments('sb')

            moms['um'] = u                     - self.get_moments('ub')
            moms['dm'] = d                     - self.get_moments('db')
            moms['sm'] = self.get_moments('s') - self.get_moments('sb')
        #--have db = f*ub
        else:
            f = self.dbfactor
            moms['g']  = self.get_moments('g')
            moms['up'] = u                     +   self.get_moments('ub')
            moms['dp'] = d                     + f*self.get_moments('ub')
            moms['sp'] = self.get_moments('s') +   self.get_moments('sb')

            moms['um'] = u                     -   self.get_moments('ub')
            moms['dm'] = d                     - f*self.get_moments('ub')
            moms['sm'] = self.get_moments('s') -   self.get_moments('sb')

        self.moms0 = moms
        self.get_BC(moms)

        #--we will store all Q2 values that has been precalc
        self.storage={}

        #--setup widths
        if self.spl in ['trans','sivers']:
            for i in range(11):
                if   i == 1: 
                    self.widths1[i] = self._widths1_uv
                elif i == 3: 
                    self.widths1[i] = self._widths1_dv
                else:        
                    self.widths1[i] = self._widths1_sea

    def beta(self,a,b):
        return gamma(a)*gamma(b)/gamma(a+b)

    def get_moments(self,flav,N=None):
        """
        if N==None: then parametrization is to be use to compute moments along mellin contour
        else the Nth moment is returned
        """
        if N==None: N=self.mellin.N
        #--note that these are normalized to the FIRST moment
        if  self.choice in ['valence','basic']:
            M,a,b,c,d,M1,a1,b1,c1,d1,M2,a2,b2,c2,d2=self.params[flav]
            norm  = self.beta(1+a  ,b+1)  +c *self.beta(1+a+0.5  ,b+1)  +d *self.beta(1+a+1.0  ,b+1)
            mom   = self.beta(N+a  ,b+1)  +c *self.beta(N+a+0.5  ,b+1)  +d *self.beta(N+a+1.0  ,b+1)
            norm1 = self.beta(1+a1 ,b1+1) +c1*self.beta(1+a1+0.5 ,b1+1) +d1*self.beta(1+a1+1.0 ,b1+1)
            mom1  = self.beta(N+a1 ,b1+1) +c1*self.beta(N+a1+0.5 ,b1+1) +d1*self.beta(N+a1+1.0 ,b1+1)
            norm2 = self.beta(1+a2 ,b2+1) +c2*self.beta(1+a2+0.5 ,b2+1) +d2*self.beta(1+a2+1.0 ,b2+1)
            mom2  = self.beta(N+a2 ,b2+1) +c2*self.beta(N+a2+0.5 ,b2+1) +d2*self.beta(N+a2+1.0 ,b2+1)
            return M*mom/norm + M1*mom1/norm1 + M2*mom2/norm2

        #--note that this is normalized to the SECOND moment
        if self.choice in ['JAM3D']:
            M0,a0,b0,M1,a1,b1=self.params[flav]
            n1 = self.beta(a0+2,b0+1)
            n2 = M1 * self.beta(a0+a1+2,b0+b1+1)
            norm = n1 + n2
            m1 = self.beta(N+a0,b0+1)
            m2 = M1 * self.beta(N+a0+a1,b0+b1+1)
            mom = m1 + m2
            return M0*mom/norm

    def _get_BC(self,g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_):
        N=self.mellin.N

        # flav composition
        vm,vp={},{}
        vm[35]= bm + cm + dm + sm - 5*tm + um
        vm[24]= -4*bm + cm + dm + sm + um
        vm[15]= -3*cm + dm + sm + um
        vm[8] = dm - 2*sp + 2*(-sm + sp) + um
        vm[3] = -dm + um
        vm[0] = np.zeros(N.size,dtype=complex)
        vp[0] = np.zeros(N.size,dtype=complex)
        vp[3] = -dp + up
        vp[8] = dp - 2*sp + up
        vp[15]= -3*cp + dp + sp + up
        vp[24]= -4*bp + cp + dp + sp + up
        vp[35]= bp + cp + dp + sp - 5*tp + up
        qs    = bp + cp + dp + sp + tp + up
        qv    = bm + cm + dm + sm + tm + um
        q     = np.zeros((2,N.size),dtype=complex)
        q[0]=np.copy(qs)
        q[1]=np.copy(g)

        BC={}
        BC['vm']=vm
        BC['vp']=vp
        BC['qv']=qv
        BC['q'] =q
        BC['um_'] = um_
        BC['dm_'] = dm_
        return BC

    def get_state(self):
        return (self.BC3,self.BC4,self.BC5)

    def set_state(self,state):
        self.BC3, self.BC4, self.BC5 = state[:]
        self.storage = {}

    def get_BC(self,moms):

        N=self.mellin.N
        zero=np.zeros(N.size,dtype=complex)

        ###############################################
        # BC for Nf=3
        g   = moms['g']
        up  = moms['up']
        um  = moms['um']
        dp  = moms['dp']
        dm  = moms['dm']
        sp  = moms['sp']
        sm  = moms['sm']
        cp  = zero
        cm  = zero
        bp  = zero
        bm  = zero
        self.BC3=self._get_BC(g,up,um,dp,dm,sp,sm,zero,zero,zero,zero,zero,zero,um,dm)

        ###############################################
        # BC for Nf=4
        BC4=self.dglap.evolve(self.BC3,self.Q20,self.mc2,3)
        g =BC4['g']
        up=BC4['up']
        dp=BC4['dp']
        sp=BC4['sp']
        cp=BC4['cp']
        bp=BC4['bp']
        tp=BC4['tp']
        um=BC4['um']
        dm=BC4['dm']
        sm=BC4['sm']
        cm=BC4['cm']
        bm=BC4['bm']
        tm=BC4['tm']
        um_=BC4['um_']
        dm_=BC4['dm_']
        self.BC4=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

        ###############################################
        # BC for Nf=5
        BC5=self.dglap.evolve(self.BC4,self.mc2,self.mb2,4)
        g =BC5['g']
        up=BC5['up']
        dp=BC5['dp']
        sp=BC5['sp']
        cp=BC5['cp']
        bp=BC5['bp']
        tp=BC5['tp']
        um=BC5['um']
        dm=BC5['dm']
        sm=BC5['sm']
        cm=BC5['cm']
        bm=BC5['bm']
        tm=BC5['tm']
        um_=BC5['um_']
        dm_=BC5['dm_']
        self.BC5=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

    def evolve(self,Q2array):

      for Q2 in Q2array:
          if Q2 not in self.storage:
              if self.mb2<Q2:
                  self.storage[Q2]=self.dglap.evolve(self.BC5,self.mb2,Q2,5)
              elif self.mc2<=Q2 and Q2<=self.mb2:
                  self.storage[Q2]=self.dglap.evolve(self.BC4,self.mc2,Q2,4)
              elif Q2<self.mc2:
                  self.storage[Q2] =self.dglap.evolve(self.BC3,self.Q20,Q2,3)

    def get_xF(self,x,Q2,flav,evolve=True):
        if self.no_evolve: Q2 = self.Q20*np.ones(len(Q2))
        if evolve: self.evolve(Q2)
        #--skip distributions that are set to zero to save time
        if flav in self.params and self.params[flav][0] == 0:
            if self.spl=='trans' and flav=='db': pass
            else:          return np.zeros(len(x))

        return np.array([x[i]*self.mellin.invert(x[i],self.storage[Q2[i]][flav]) for i in range(len(x))])

    #--get higher twist function through WW relation (multiplied by x)
    def get_hL_WW(self,x,Q2,flav,evolve=True):
        if evolve: self.evolve(Q2)
        #--skip distributions that are set to zero to save time
        if flav in self.params and self.params[flav][0] == 0:
            if self.spl=='trans' and flav=='db': pass
            else:          return np.zeros(len(x))

        L = len(Q2)

        N = 100
        #--setup interpolation
        X = np.geomspace(1e-6,0.1,N,endpoint=False)
        X = np.append(X,np.linspace(0.1,1,N))

        X, _Q2 = np.meshgrid(X,Q2)
        X = X.flatten()
        _Q2 = _Q2.flatten()
        func = self.get_xF(X,_Q2,flav)/X/X**2
        func = func.reshape(L,2*N)
        X = X.reshape(L,2*N)
        moment_temp = np.array(cumtrapz(func,X,initial=0.0,axis=1))
        for i in range(len(moment_temp)):
            moment_max  = moment_temp[i][-1]
            moment_temp[i] =moment_max - moment_temp[i]
        xhL = 2*X**2*moment_temp

        result = np.zeros(len(Q2))
        #--loop over Q2 values
        for i in range(len(Q2)):
            result[i] = interp1d(X[i],xhL[i],kind='cubic')(x[i])

        return result
    

    #--Soffer Bound
    def check_SB(self):

        X  = self.X
        Q2 = conf['Q20']

        SB = self.SB

        violations_p = np.zeros(len(X))
        violations_m = np.zeros(len(X))
        #--if SB applied to q and qb, it automatically applies to qv
        flavs = ['u','d','ub','db']
        for flav in flavs:
            h1 = self.get_xF(X,Q2*np.ones(len(X)),flav)/X
            plus  = SB['mean'][flav] + SB['std'][flav] + h1 
            minus = SB['mean'][flav] + SB['std'][flav] - h1 

            for i in range(len(X)):
                if plus[i]  < 0.0: violations_p[i] += plus[i]
                if minus[i] < 0.0: violations_m[i] += minus[i]

        violations = np.append(violations_p, violations_m)

        #--a factor of 10 seems to work
        return violations*10

    #--generate report for Soffer Bound violations   
    def gen_report(self):
          
        L=[]

        if 'SofferBound' in conf and conf['SofferBound']:
            res2 = conf['SB chi2']
            L.append('chi2 from Soffer Bound: %3.5f'%res2)

        return L

    #--for JAM3D
    def get_C(self,x, Q2):
        if self.no_evolve: Q2 = self.Q20 
        self.evolve([Q2])
        flavs = ['g','u','ub','d','db','s','sb','c','cb','b','bb']
        if self.shape=='nderiv': return np.array([self.mellin.invert(x,self.storage[Q2][flav]) for flav in flavs])
        if self.shape=='deriv':  return np.array([self.mellin.invert_deriv(x,self.storage[Q2][flav]) for flav in flavs])
       
    def get_widths(self,Q2):
        return np.abs(self.widths1)

 


