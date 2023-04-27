import sys
import os

homedir = os.path.realpath(__file__)[:-len('PROSPECT/FitPlots_PW.py')]
common_dir = 'Common_cython/'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+'PROSPECT/')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import PROSPECT as PS

path_to_style = homedir+common_dir
datadir = homedir+'PROSPECT/PlotData/'
plotdir = homedir + 'PROSPECT/Figures/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------
# We load the PROSPECT class
fitter = PS.Prospect()


# We define a function to read the data in PlotData/
# These data are produced by PWSterileFitTableX.py  or WPSterileFitTableX.py

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass, Sigma = sigma)
    # chi2 = fitter.get_poisson_chi2(model)
    chi2 = fitter.get_chi2(model)
    return chi2

# We apply a common style to all plots
def stylize(axxis,contours,t_ax = [1e-3,1], m_ax = [1e-2,10]):
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([0.004,1])
    axxis.set_ylim([0.08,2])
    legend_elements = [Line2D([0], [0], color=color1, ls = '-', lw=2, label=r'$1\sigma$ (68\% C.L.)'),
                       Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'$2\sigma$ (95\% C.L.)'),
                       Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'$3\sigma$ (99\% C.L.)'),
                       Line2D([0], [0], marker='+', color='c', lw = 0, label='Best Fit', markerfacecolor='b', markersize=8)]
    axxis.legend(handles = legend_elements, loc = 'upper left', fontsize = 16)


# Colorblind-sensitive colors
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'

titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

# -------------------------------------------------
# STERILE CONTOUR - STANDARD FORMALISM
# -------------------------------------------------

print('Standard')
# We load the data
mass_PW = np.load(datadir+'PWSterileMass_2.npy')
angle_PW = np.load(datadir+'PWSterileAngle_2.npy')
chi2_PW = np.load(datadir+'PWSterileChi2_2.npy')

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(chi2_PW[:] == np.min(chi2_PW[:]))[0]
bestfit = chi2_PW[min_index]
print('Best fit values and chi2: ', mass_PW[min_index], angle_PW[min_index], bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0., 0.)
print('Null hyp chi2: ', null_hyp_PW)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts = axBF.tricontour(angle_PW,mass_PW,(chi2_PW-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle_PW[min_index],mass_PW[min_index],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(mass_PW[min_index],angle_PW[min_index], bestfit), fontsize = titlesize)
figBF.savefig(plotdir+'PWContour_bestfit_bs.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle_PW[min_index],mass_PW[min_index],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig(plotdir+'PWContour_nullhyp_bs.png')
