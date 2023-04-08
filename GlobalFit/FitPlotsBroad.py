#This script produces contour plot of mass vs angle while marginalizing over b
import sys
import os

homedir = os.path.realpath(__file__)[:-len('GlobalFit/FitPlotsBroad.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import GlobalFit as GF

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
# These data are produced by PWSterileFitTableX.py  or WPSterileFitTableX.py

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)
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
    axxis.set_ylim([0.08,10.])
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
# STERILE PLANE WAVE CONTOUR - PLANE WAVE FORMALISM
# -------------------------------------------------

print('Plane wave')
# We load the data
#mass_PW = np.load(datadir+'BroadSterileMass.npy')
#angle_PW = np.load(datadir+'BroadSterileAngle.npy')
b_PW = np.load(datadir+'BroadSterileb.npy')
chi2_PW = np.load(datadir+'BroadSterileChi2.npy')
mass = np.load(datadir+'BroadSterileMass.npy')
angle = np.load(datadir+'BroadSterileAngle.npy')

m_n = 10
a_n = 10

datmass1 = np.logspace(np.log10(0.08),1,m_n) #0.08-10
datangl1 = np.logspace(np.log10(4e-3),0,a_n) #0.004-1


# meshgrid returns 2 2-dimensional arrays that represent the x and y coordinates of all points in the grid.
mass_grid, angle_grid= np.meshgrid(datmass1, datangl1)
# ravel returns the same array but with all internal brackets removed.
mass_PW, angle_PW= np.ravel(mass_grid), np.ravel(angle_grid)

# Reshape fourth array to match shape of grids
chi2_PW_reshaped = np.reshape(chi2_PW, (10, 10, 5))
# Compute minimum value of fourth array along third axis
min_chi2_PW = np.ravel(np.min(chi2_PW_reshaped, axis=2))

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(chi2_PW[:] == np.min(chi2_PW[:]))[0]
bestfit = chi2_PW[min_index]
print('Best fit values and chi2: ', mass[min_index], angle[min_index], bestfit)

# We find which is the chi2 of the null hypothesis. Use np.sinc in models for this.
null_hyp_WP = getChi2(0,0,0, broad_sterile = True)
print('Null hyp chi2: ',null_hyp_WP)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)


conts = axBF.tricontour(angle_PW,mass_PW,(min_chi2_PW-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle[min_index],mass[min_index],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)

#figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(mass_PW[min_index],angle_PW[min_index], bestfit), fontsize = titlesize)
figBF.savefig(plotdir+'Broad_bestfit_marg.png')

# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(angle_PW, mass_PW,(min_chi2_PW-null_hyp_WP),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle[min_index],mass[min_index],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH,conts)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_WP), fontsize = titlesize)
figNH.savefig(plotdir+'Broad_nullhyp_marg.png')
