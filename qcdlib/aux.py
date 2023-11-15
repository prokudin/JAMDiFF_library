from mpmath import fp
from scipy.special import gamma
import numpy as np
from tools.config import conf

class AUX:

  def __init__(self):

    self.set_constants()
    self.set_masses()
    self.set_couplings()
    self.set_ckm()

  def set_constants(self):
    self.CA=3.0
    self.CF=4.0/3.0
    self.TR=0.5
    self.TF=0.5
    self.euler=fp.euler 

  def set_masses(self):

    self.me   = 0.000511
    self.mmu  = 0.105658
    self.mtau = 1.77684
    self.mu   = 0.055
    self.md   = 0.055
    self.ms   = 0.2

    # PDG
    if 'mc' in conf: self.mc = conf['mc']
    else:            self.mc = 1.28
    if 'mb' in conf: self.mb = conf['mb']
    else:            self.mb = 4.18
    
    if 'cj masses' in conf:
      if conf['cj masses']:
        self.mc=1.3
        self.mb=4.5

    self.mZ   = 91.1876
    self.mW   = 80.398
    self.M    = 0.93891897
    self.Mpi  = 0.13803
    self.Mk   = 0.493677
    self.Mdelta = 1.232

    self.me2   = self.me**2 
    self.mmu2  = self.mmu**2 
    self.mtau2 = self.mtau**2
    self.mu2   = self.mu**2  
    self.md2   = self.md**2  
    self.ms2   = self.ms**2  
    self.mc2   = self.mc**2  
    self.mb2   = self.mb**2  
    self.mZ2   = self.mZ**2  
    self.mW2   = self.mW**2  
    self.M2    = self.M**2  
    self.Mpi2  = self.Mpi**2  
    self.Mdelta2=self.Mdelta**2

  def set_couplings(self):

    self.c2w = self.mW2/self.mZ2
    self.s2w = 1.0-self.c2w
    self.s2wMZ = 0.23116
    self.c2wMZ = 1.0 - self.s2wMZ
    self.alfa  = 1/137.036
    if 'alphaS(MZ)' in conf:
      self.alphaSMZ=conf['alphaS(MZ)']
    else:
      self.alphaSMZ = 0.118
    self.GF = 1.1663787e-5   # 1/GeV^2

    # axial couplings of quarks to Z boson (JAM3D)
    self.cauz = 0.5 
    self.cacz = 0.5
    self.cadz = -0.5
    self.casz = -0.5
    # vector couplings of quarks to Z boson (JAM3D)
    self.cvuz = self.cauz - 4./3.*self.s2w 
    self.cvcz = self.cacz - 4./3.*self.s2w
    self.cvdz = self.cadz + 2./3.*self.s2w
    self.cvsz = self.casz + 2./3.*self.s2w

    self.alphaS0=0.37259
  
  def _get_psi(self,i,N):
    return fp.psi(i,complex(N.real,N.imag))

  def get_psi(self,i,N):
    return np.array([self._get_psi(i,n) for n in N],dtype=complex)

  def beta(self,a,b):
    return gamma(a)*gamma(b)/gamma(a+b)

  def _get_ckm2fromckm(self, ckm, m_ckmcharge):
      if m_ckmcharge == 0:
          return None

      ckm2 = np.zeros((14, 14))
      for iU in range(3):
          for iD in range(3):
              utype = 2*iU+2
              dtype = 2*iD+1

              #/ W-
              if m_ckmcharge < 0:
                  utype *= -1

              #/ W+
              if m_ckmcharge > 0:
                  dtype *= -1

              #/ translate to array indices
              utype += 6
              dtype += 6

              V2 = ckm[iU, iD]*ckm[iU, iD]

              ckm2[utype, dtype] = V2
              ckm2[dtype, utype] = V2

      return ckm2

  def set_ckm(self):
    self.ckm=np.zeros((3,3))
    self.ckm[0,0]=0.97419 
    self.ckm[0,1]=0.2257 
    self.ckm[0,2]=0.00359
    self.ckm[1,0]=0.2256
    self.ckm[1,1]=0.97334
    self.ckm[1,2]=0.0415

    self.ckm2Wp = self._get_ckm2fromckm(self.ckm, +1)
    self.ckm2Wm = self._get_ckm2fromckm(self.ckm, -1)

    #--for JAM3D
    self.Vud = 0.97420 # CKM matrix element +- 0.00021
    self.Vus = 0.2243  # CKM matrix element +- 0.0005
    self.Vcd = 0.218   # CKM matrix element +- 0.004

  def charge_conj(self,p):
      return np.copy(p[[0,2,1,4,3,6,5,8,7,10,9]])

  def p2n(self, p):
      return np.copy(p[[0,3,4,1,2,5,6,7,8,9,10]])

  def q2qbar(self, p):
      """
      distributions p
      p[0]    # g
      p[1]    # u
      p[2]    # ub
      p[3]    # d
      p[4]    # db
      p[5]    # s
      p[6]    # sb
      p[7]    # c
      p[8]    # cb
      p[9]    # b
      p[10]   # bb
      q <-> qbar used in Drell-Yan 
      """
      return np.copy(p[[0,2,1,4,3,6,5,8,7,10,9]])













