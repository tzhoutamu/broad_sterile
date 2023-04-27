import sys
import os

homedir = os.path.realpath(__file__)[:-len('PROSPECT/FitPlotsPROS+NEOS.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+'NEOS/')
sys.path.append(homedir+'DayaBay/')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import PROSPECT as PS
import NEOS
import DayaBay as DB
from BroadFitTable import m_n, a_n, b_n, datmass1, datangl1

path_to_style= homedir+common_dir
datadir_pros = homedir+'PROSPECT/PlotData/'
datadir_neos = homedir+'NEOS/PlotData/'
datadir_daya = homedir+'DayaBay/PlotData/'
plotdir = homedir+'PROSPECT/Figures/PROS+NEOS/2-10/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

# We load the PROSPECT class
fitter_pros = PS.Prospect()
fitter_neos = NEOS.Neos()
fitter_daya = DB.DayaBay()

# We define a function to read the data in PlotData/
# These data are produced by BSSterileFitTableX.py  or BSSterileFitTableX.py

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)

    chi2 = fitter_pros.get_chi2(model)+fitter_neos.get_chi2(model)+fitter_daya.get_poisson_chi2(model)
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
    axxis.set_ylim([2,10])
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
# STANDARD DELTA FUNCTION MASS STATES FORMALISM
# -------------------------------------------------

print('Standard')
# We load the data from PROSPECT, NEOS, and DayaBay
mass_PW = np.load(datadir_pros+'PWSterileMass_noint.npy')
angle_PW = np.load(datadir_pros+'PWSterileAngle_noint.npy')
chi2_PW = np.add(np.load(datadir_pros+'PWSterileChi2_noint.npy'),np.load(datadir_neos+'PWSterileChi2_p2.npy'))

# We find which is the point with minimum chi2, i.e. our best fit.
min_index_PW = np.where(chi2_PW[:] == np.min(chi2_PW[:]))[0]
bestfit = chi2_PW[min_index_PW]
print('Best fit values and chi2: ', mass_PW[min_index_PW], angle_PW[min_index_PW], bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_PW)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts = axBF.tricontour(angle_PW,mass_PW,(chi2_PW-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(mass_PW[min_index_PW],angle_PW[min_index_PW], bestfit), fontsize = titlesize)
figBF.savefig(plotdir+'PWContour_bestfit_bs.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig(plotdir+'PWContour_nullhyp_bs.png')

# -------------------------------------------------
# BROAD STERILE FORMALISM
# -------------------------------------------------

# We load the data
#mass_BS = np.load(datadir+'BroadSterileMass_frac.npy')
#angle_BS = np.load(datadir+'BroadSterileAngle_frac.npy')
b_BS = np.load(datadir_pros+'BroadSterileb_2-10.npy')
chi2_BS = np.add(np.load(datadir_pros+'BroadSterileChi2_2-10.npy'),np.load(datadir_neos+'BroadSterileChi2_2-10_noint.npy'))


# meshgrid returns 2 2-dimensional arrays that represent the x and y coordinates of all points in the grid.
mass_grid, angle_grid= np.meshgrid(datmass1, datangl1)
# ravel returns the same array but with all internal brackets removed.
mass_BS, angle_BS= np.ravel(mass_grid), np.ravel(angle_grid)

# Reshape fourth array to match shape of grids
chi2_BS_reshaped = np.reshape(chi2_BS, (m_n, a_n, b_n))
# Compute minimum value of fourth array along third axis (Marginalizing over b)
min_chi2_BS = np.min(chi2_BS_reshaped, axis=2)
# Eliminate repeated values in 3 axes
mass_ax = np.ravel(np.unique(mass_BS))
angle_ax = np.ravel(np.unique(angle_BS))
b_ax = np.ravel(np.unique(b_BS))
# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.unravel_index(np.argmin(min_chi2_BS), min_chi2_BS.shape)
angle_index, mass_index = min_index
bestfit = min_chi2_BS[min_index]
print('Best fit values and chi2: ', mass_ax[mass_index], angle_ax[angle_index], bestfit)

# We find which is the chi2 of the null hypothesis. Use np.sinc in models for this.
null_hyp_BS = getChi2(0,0,0, broad_sterile = True)
print('Null hyp chi2: ',null_hyp_BS)
# To make statistic the same dimension as angle_BS and mass_BS
min_chi2_BS = np.ravel(min_chi2_BS)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts = axBF.tricontour(angle_BS,mass_BS,(min_chi2_BS-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle_ax[angle_index],mass_ax[mass_index],marker = '+', label = r'Best fit')
# axBF.scatter(data_BS[:,1],data_BS[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

#figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(mass_BS[min_index],angle_BS[min_index], bestfit), fontsize = titlesize)
figBF.savefig(plotdir+'Broad_bestfit_marg.png')

# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(angle_BS, mass_BS,(min_chi2_BS-null_hyp_BS),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle_ax[angle_index],mass_ax[mass_index],marker = '+', label = 'Our best fit')
# axNH.scatter(data_BS[:,1],data_BS[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_BS), fontsize = titlesize)
figNH.savefig(plotdir+'Broad_nullhyp_marg.png')

# ----------------------------------------------
# 2SIGMA PLOT COMPARISON
# ----------------------------------------------
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
cont_PW = ax_comp.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [6.18],  colors = color2, linestyles = ['solid'])
ax_comp.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', color='r', label = r'Original best fit')
cont_BS = ax_comp.tricontour(angle_BS, mass_BS,(min_chi2_BS-null_hyp_BS),levels = [6.18],  colors = color3, linestyles = ['solid'])
ax_comp.scatter(angle_ax[angle_index],mass_ax[mass_index],marker = '+', color='c', label = r'Our best fit')

ax_comp.annotate('PROS+NEOS', xy = (5e-3,1.5), size = 42)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([0.004,1])
ax_comp.set_ylim([2,10])
legend_elements = [Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'$2\sigma$ Delta Function'),
                   Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'$2\sigma$ Broad Sterile'),
                   Line2D([0], [0], marker='+', color='r', lw = 0, label='Original Best Fit', markerfacecolor='b', markersize=8),
                   Line2D([0], [0], marker='+', color='c', lw = 0, label='Broad Best Fit', markerfacecolor='b', markersize=8)]
ax_comp.legend(handles = legend_elements, loc = 'lower left', fontsize = 16)

fig_comp.savefig(plotdir+'ContourComparison_nullhyp.png')

#print(fitter.get_chi2(Models.BroadSterileFrac(Sin22Th14 = 0.27, DM2_41 = 0.57, bfrac = 0.99)))