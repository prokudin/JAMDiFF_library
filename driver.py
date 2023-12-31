#!/usr/bin/env python
import os,sys

#--from corelib
from analysis.corelib import core, jar

#--from qpdlib
from analysis.qpdlib import diff, tpdf, tensorcharge

#--primary working directory
try: wdir=sys.argv[1]
except: wdir = None

###################################################
##--Plot QCFs
###################################################

tpdf.gen_xf(wdir,Q2=4)
tpdf.plot_xf(wdir,Q2=4,mode=0)
tpdf.plot_xf(wdir,Q2=4,mode=1)

tpdf.gen_moments(wdir,Q2=4)
tensorcharge.plot_tensorcharge(wdir,mode=1)
tensorcharge.plot_tensorcharge(wdir,mode=0)

diff.gen_D (wdir,Q2=100)
diff.plot_D(wdir,Q2=100,mode=0)
diff.plot_D(wdir,Q2=100,mode=1)

diff.gen_H (wdir,Q2=100)
diff.plot_H(wdir,Q2=100,mode=0)
diff.plot_H(wdir,Q2=100,mode=1)




