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
plotdir = homedir+'Misc/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#00BFFF'
color4 = '#32CD32'
color5 = '#800080'

# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

def marg(m_n,a_n,b_n,datab,chi2):
    """
    This function performs marginalization over b tilde.

    Input:
    m_n: steps of Delta m^2_{41}
    a_n: steps of sin^2(2theta_{41})
    b_n: steps of \tilde{b}
    datmass: array of mass
    datangl: array of angle
    chi2: calculated chi2

    Output:
    bestfit values of mass, angle, and the corresponding chi2
    mass, angle and chi2 arrays reduced to 2D (with b axis removed)
    1D mass and angle axis for plotting the best fit point
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

# load data with mass range 0.08-2
m_n1 = 60
a_n1 = 60
b_n1 = 60
datmass1 = np.logspace(np.log10(0.08),np.log10(2),m_n1) #0.08-2
datangl1 = np.logspace(np.log10(4e-3),0,a_n1) #0.004-1
datab1 = np.logspace(np.log10(1e-4),np.log10(0.99),b_n1) #fractional breadth
chi2_BS_1 = np.add(np.load(datadir_pros+'BroadSterileChi2_0.08-2.npy'),np.load(datadir_neos+'BroadSterileChi2_0.08-2.npy'),np.load(datadir_daya+'BroadSterileChi2_0.08-2.npy'))
# load data with mass range 2-10
m_n2 = 60
a_n2 = 60
b_n2 = 60
datmass2 = np.logspace(np.log10(2),np.log10(10),m_n2) #2-10
datangl2 = np.logspace(np.log10(4e-3),0,a_n2) #0.004-1
datab2 = np.logspace(np.log10(1e-4),np.log10(0.99),b_n2) #fractional breadth
chi2_BS_2 = np.add(np.load(datadir_pros+'BroadSterileChi2_2-10.npy'),np.load(datadir_neos+'BroadSterileChi2_2-10.npy'),np.load(datadir_daya+'BroadSterileChi2_2-10.npy'))

b_ax_1, bestfit_1, bestb_1, min_chi2_BS_1=marg(m_n1,a_n1,b_n1,datab1,chi2_BS_1)
b_ax_2, bestfit_2, bestb_2, min_chi2_BS_2=marg(m_n2,a_n2,b_n2,datab2,chi2_BS_2)

# choose the minimum chi2 out of 2 chi2 arrays
for i in range(len(min_chi2_BS_1)):
    if min_chi2_BS_2[i] == min(min_chi2_BS_1[i],min_chi2_BS_2[i]):
        min_chi2_BS_1[i] = min_chi2_BS_2[i]

# choose the bestfit from 2 data files
if bestfit_1 == min(bestfit_1,bestfit_2):
    bestb, bestfit = bestb_1, bestfit_1
else:
    bestb, bestfit = bestb_2, bestfit_2

margins = dict(left=0.14, right=0.96,bottom=0.19, top=0.9)
plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
#axx.set_xlim([axis[0],axis[1]])
#axx.set_ylim()
axx.set_xscale('log')
axx.set_xlim([1e-4,0.99])
axx.set_ylabel(r'$\Delta{\chi^2}$', fontsize = 24)
axx.set_xlabel(r'$\tilde{b}$', fontsize = 24)
axx.plot(b_ax_1,min_chi2_BS_1-bestfit, label = 'Global', color = color1)
axx.legend(bbox_to_anchor=(2e-4, 1), loc="upper left", ncol=1, fontsize=15)
plot.suptitle(r'Global best fit: $\tilde{b}=%.3f$, Total $\chi^2=%.2f$'%(bestb, bestfit), fontsize=13)
plot.savefig(plotdir+'BroadSterileChi2b.png')