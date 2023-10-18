# This script produces chi2 vs. b plot while marginalizing over mass and mixing
import sys
import os

homedir = os.path.realpath(__file__)[:-len('Misc/FitPlotsChi2b.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+'NEOS/')
sys.path.append(homedir+'DayaBay/')
sys.path.append(homedir+'PROSPECT/')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D


path_to_style= homedir+common_dir
datadir_pros = homedir+'PROSPECT/PlotData/'
datadir_neos = homedir+'NEOS/PlotData/'
datadir_daya = homedir+'DayaBay/PlotData/'
datadir_dane = homedir+'GlobalFit/PlotData/'
datadir_best = homedir+'BEST/PlotData/'
plotdir = homedir+'Misc/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#00BFFF'
color4 = '#089099'
color5 = '#800080'

# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

def marg(m_n,a_n,b_n,datab,chi2):
    """
    This function performs marginalization over mass and mixing on a 3D parameter scan.

    Input:
    m_n: steps of Delta m^2_{41}
    a_n: steps of sin^2(2theta_{41})
    b_n: steps of \tilde{b}
    datab: array of \tilde{b}
    chi2: calculated chi2

    Output:
    b_ax: b axis
    bestfit: best-fit chi2 value
    bestb: best-fit b value
    min_chi2_BS: chi2 as a function of b
    """
    # Reshape fourth array to match shape of grids
    chi2_BS_reshaped = np.reshape(chi2, (m_n, a_n, b_n))
    # Compute minimum value of fourth array along first and second axis (Marginalize over mass and mixing)
    min_chi2_BS = np.ravel(np.min(np.min(chi2_BS_reshaped, axis=1), axis=0))
    # Eliminate repeated values in b_BS
    b_ax = np.unique(datab)
    # We find which is the point with minimum chi2, i.e. our best fit.
    min_index = np.where(min_chi2_BS[:] == np.min(min_chi2_BS[:]))[0]
    bestfit = min_chi2_BS[min_index]
    bestb = b_ax[min_index]
    return b_ax, bestfit, bestb, min_chi2_BS

# We load the data
m_n1 = 60
a_n1 = 60
b_n1 = 60
datmass1 = np.logspace(np.log10(0.07256),1,m_n1) # range of mass: 0.07256-10
datangl1 = np.logspace(np.log10(4e-3),0,a_n1) # range of mixing: 0.004-1
datab1 = np.logspace(np.log10(1e-4),np.log10(0.99),b_n1) # range of \tilde{b}: 1e-4-0.99
chi2_BS_1 = np.add(np.load(datadir_pros+'BroadSterileChi2_precise.npy'),np.load(datadir_dane+'BroadSterileChi2_precise.npy'))
chi2_BS_2 = np.load(datadir_pros+'BroadSterileChi2_precise.npy')
chi2_BS_3 = np.load(datadir_dane+'BroadSterileChi2_precise.npy')
chi2_BS_4 = np.load(datadir_best+'BroadSterileChi2_frac.npy')

# Perform marginalization over mass and mixing
b_ax_1, bestfit_1, bestb_1, min_chi2_BS_1=marg(m_n1,a_n1,b_n1,datab1,chi2_BS_1)
b_ax_2, bestfit_2, bestb_2, min_chi2_BS_2=marg(m_n1,a_n1,b_n1,datab1,chi2_BS_2)
b_ax_3, bestfit_3, bestb_3, min_chi2_BS_3=marg(m_n1,a_n1,b_n1,datab1,chi2_BS_3)
b_ax_4, bestfit_4, bestb_4, min_chi2_BS_4=marg(m_n1,a_n1,b_n1,datab1,chi2_BS_4)
margins = dict(left=0.14, right=0.96,bottom=0.19, top=0.9)
plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
#axx.set_xlim([axis[0],axis[1]])
#axx.set_ylim([72.5,79.8])
axx.set_xscale('log')
axx.set_xlim([1e-4,0.99])
axx.set_ylabel(r'$\Delta{\chi^2}$', fontsize = 24)
axx.set_xlabel(r'$\tilde{b}$', fontsize = 24)
axx.plot(b_ax_2,min_chi2_BS_2-bestfit_2, label = 'PROSPECT', color = color2)
axx.plot(b_ax_3,min_chi2_BS_3-bestfit_3, label = 'NEOS+DayaBay', color = color3)
axx.plot(b_ax_4,min_chi2_BS_4-bestfit_4, label = 'BEST', color = color4)
axx.plot(b_ax_1,min_chi2_BS_1-bestfit_1, label = 'Reactor Global', color = color1)
axx.legend(bbox_to_anchor=(2e-4, 1), loc="upper left", ncol=1, fontsize=15)
plot.suptitle(r'Reactor Global best-fit: $\tilde{b}=%.3f$, Total $\chi^2=%.2f$'%(bestb_1, bestfit_1), fontsize=20)
plot.savefig(plotdir+'BroadSterileChi2b_precise.png')