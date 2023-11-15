import sys, os
import numpy as np

#--from qcdlib
from qcdlib import diff, tdiff, tpdf
from qcdlib import aux, eweak, alphaS, mellin

#--from fitlib
from fitlib.parman import PARMAN

#--from tools
from tools.tools    import checkdir
from tools.config   import conf,load_config,options

import lhapdf




class RESMAN:

    def __init__(self,nworkers=1,parallel=False,datasets=True,load_lhapdf=True):

        self.load_lhapdf=load_lhapdf
        self.setup_core()
        self.parman=PARMAN()

    def setup_core(self):

        conf['aux']     = aux.AUX()
        conf['mellin']  = mellin.MELLIN(npts=4)
        conf['mellin-pion']=mellin.MELLIN(npts=8,extended=True)
        conf['dmellin'] = mellin.DMELLIN(nptsN=4,nptsM=4)
        conf['alphaS']  = alphaS.ALPHAS()
        conf['eweak']   = eweak.EWEAK()

        #--setup LHAPDF
        if 'lhapdf_pdf'   in conf: self.lhapdf_pdf   = conf['lhapdf_pdf']
        else:                      self.lhapdf_pdf   = 'JAM22-PDF_proton_nlo_pos'

        if 'lhapdf_ppdf'  in conf: self.lhapdf_ppdf  = conf['lhapdf_ppdf']
        else:                      self.lhapdf_ppdf  = 'JAM22-PPDF_proton_nlo_pos'

        if self.load_lhapdf:
            #--load all replicas
            #lhapdf.setVerbosity(0)
            os.environ['LHAPDF_DATA_PATH'] = 'lhapdf'
            conf['LHAPDF:PDF']   = lhapdf.mkPDFs(self.lhapdf_pdf )
            conf['LHAPDF:PPDF']  = lhapdf.mkPDFs(self.lhapdf_ppdf)
            #lhapdf.setVerbosity(1)

        for func in conf['params']:
            if func=='tpdf':
                conf['tpdf']     = tpdf.TPDF(func=func)
                conf['tpdf-mom'] = tpdf.TPDF(func=func,mellin=mellin.IMELLIN())

            if func=='diffpippim':
                conf[func] = diff.DIFF(func=func)

            if func=='tdiffpippim':
                conf[func] = tdiff .TDIFF(func=func)









