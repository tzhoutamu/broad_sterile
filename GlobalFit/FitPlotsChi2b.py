# This script produces chi2 vs. b plot while marginalizing over mass and mixing
import sys
import os

homedir = os.path.realpath(__file__)[:-len('GlobalFit/FitPlotsBroad_original.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import GlobalFit as GF
from BroadFitTable import m_n, a_n, b_n

path_to_style= homedir+common_dir
datadir = homedir+'GlobalFit/PlotData/'
plotdir = homedir+'GlobalFit/Figures/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

# We load the class
fitter = GF.GlobalFit()

# We define a function to read the data in PlotData/
# These data are produced by BroadFitTable.py

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)
    chi2 = fitter.get_chi2(model)
    return chi2

# We load the data
mass_PW = np.load(datadir+'BroadSterileMass.npy')
angle_PW = np.load(datadir+'BroadSterileAngle.npy')
b_PW = np.load(datadir+'BroadSterileb.npy')
chi2_PW = np.load(datadir+'BroadSterileChi2.npy')

# Reshape fourth array to match shape of grids
chi2_PW_reshaped = np.reshape(chi2_PW, (m_n, a_n, b_n))
# Compute minimum value of fourth array along first and second axis (Marginalize over mass and mixing)
min_chi2_PW = np.ravel(np.min(np.min(chi2_PW_reshaped, axis=1),axis=0))
# Eliminate repeated values in 3 axes
mass_PW = np.ravel(np.unique(mass_PW))
angle_PW = np.ravel(np.unique(angle_PW))
b_PW = np.ravel(np.unique(b_PW))
# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(min_chi2_PW[:] == np.min(min_chi2_PW[:]))[0]
bestfit = min_chi2_PW[min_index]

# We find which is the chi2 of the null hypothesis. Use np.sinc in models for this.
null_hyp_WP = getChi2(0,0,0, broad_sterile = True)
print('Null hyp chi2: ',null_hyp_WP)

xpoints = b_PW[:]
ypoints = min_chi2_PW[:]-bestfit
print('Best fit b value and chi2: ',b_PW[min_index],bestfit)
plt.plot(xpoints, ypoints)
plt.title(r'Best fit: $b=%.2f$, Total $\chi^2=%.2f$'%(b_PW[min_index], bestfit), fontsize=13)
plt.ylabel(r'$\Delta{\chi^2}$')
plt.xlabel(r'$b$($\text{eV}^2$)')
plt.xscale('log')
plt.subplots_adjust(left=0.16, right=0.97,bottom=0.15, top=0.93)
plt.savefig('Figures/BroadSterileChi2b.png')