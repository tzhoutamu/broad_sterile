import sys
import os

homedir = os.path.realpath(__file__)[:-len('Misc/FitPlotsGlobal.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+'NEOS/')
sys.path.append(homedir+'DayaBay/')
sys.path.append(homedir+'PROSPECT/')
sys.path.append(homedir+'GlobalFit/')
sys.path.append(homedir+"BEST/")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import PROSPECT as PS
import NEOS
import DayaBay as DB
import GlobalFit as GF
import BEST
#from BroadFitTable import m_n, a_n, b_n, datmass1, datangl1

path_to_style= homedir+common_dir
datadir_pros = homedir+'PROSPECT/PlotData/'
datadir_neos = homedir+'NEOS/PlotData/'
datadir_daya = homedir+'DayaBay/PlotData/'
datadir_dane = homedir+'GlobalFit/PlotData/'
datadir_best = homedir+'BEST/PlotData/'
plotdir = homedir+'Misc/GlobalFitPlots/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

# We load the PROSPECT class
fitter_pros = PS.Prospect()
fitter_neos = NEOS.Neos()
fitter_daya = DB.DayaBay()
fitter_dane = GF.GlobalFit() # Actually DayaBay + NEOS minimized over nuisance parameter
fitter_best = BEST.Best()
# We define a function to read the data in PlotData/
# These data are produced by BSSterileFitTableX.py  or BSSterileFitTableX.py

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)

    chi2 = fitter_dane.get_chi2(model)+fitter_pros.get_chi2(model)
    return chi2

# We apply a common style to all plots
def stylize(axxis,t_ax = [1e-3,1], m_ax = [1e-2,10]):
    axxis.annotate('Global', xy=(5e-3, 0.1), size=42)
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([0.004,1])
    axxis.set_ylim([0.08,10])
    legend_elements = [Line2D([0], [0], color=color1, ls = '-', lw=2, label=r'$1\sigma$ (68\% C.L.)'),
                       Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'$2\sigma$ (95\% C.L.)'),
                       Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'$3\sigma$ (99\% C.L.)'),
                       Line2D([0], [0], marker='+', color='c', lw = 0, label='Best Fit', markerfacecolor='b', markersize=8)]
    axxis.legend(handles = legend_elements, loc = 'upper left', fontsize = 16)

def marg_over_b(m_n,a_n,b_n,datmass,datangl,chi2):
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
    # meshgrid returns 2 2-dimensional arrays that represent the x and y coordinates of all points in the grid.
    mass_grid, angle_grid= np.meshgrid(datmass, datangl)
    # ravel returns the same array but with all internal brackets removed.
    mass_BS, angle_BS= np.ravel(mass_grid), np.ravel(angle_grid)
    # Reshape fourth array to match shape of grids
    chi2_BS_reshaped = np.reshape(chi2, (m_n, a_n, b_n))
    # Compute minimum value of fourth array along third axis (Marginalizing over b)
    min_chi2_BS = np.min(chi2_BS_reshaped, axis=2)
    # Eliminate repeated values in 3 axes
    mass_ax = np.unique(mass_BS)
    angle_ax = np.unique(angle_BS)
    # b_ax = np.unique(b_BS)
    # We find which is the point with minimum chi2, i.e. our best fit.
    min_index = np.unravel_index(np.argmin(min_chi2_BS), min_chi2_BS.shape)
    angle_index, mass_index = min_index
    best_mass = mass_ax[mass_index]
    best_angle = angle_ax[angle_index]
    bestfit_BS = min_chi2_BS[min_index]
    # To make statistic the same dimension as angle_BS and mass_BS
    min_chi2_BS = np.ravel(min_chi2_BS)
    return best_mass, best_angle, bestfit_BS, mass_BS, angle_BS, min_chi2_BS, mass_ax, angle_ax

# Colorblind-sensitive colors
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'

titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)


"""
----------------------------------------------
STANDARD DELTA FUNCTION MASS STATES FORMALISM
----------------------------------------------
"""


print('Standard')
# We load the data from PROSPECT, NEOS, and DayaBay
mass_PW = np.load(datadir_pros+'PWSterileMass_8e-2_10.npy') # 0.08-10 in 180 steps
angle_PW = np.load(datadir_pros+'PWSterileAngle_4e-3_1.npy') # 4e-3 to 1 in 180 steps
chi2_PW = np.add(np.load(datadir_pros+'PWSterileChi2_180*180.npy'),np.load(datadir_dane+'PWSterileChi2_180*180.npy'))
# We find which is the point with minimum chi2, i.e. our best fit.

print('PROSPECT best-fit chi2: ', np.min(np.load(datadir_pros+'PWSterileChi2_180*180.npy')))
print('NEOS and DayaBay best-fit chi2: ', np.min(np.load(datadir_dane+'PWSterileChi2_180*180.npy')))

min_index_PW = np.where(chi2_PW[:] == np.min(chi2_PW[:]))[0]
bestfit = chi2_PW[min_index_PW]
print('Reactor Global best fit values and chi2: ', mass_PW[min_index_PW], angle_PW[min_index_PW], bestfit)

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_PW)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

axBF.tricontour(angle_PW,mass_PW,(chi2_PW-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(mass_PW[min_index_PW],angle_PW[min_index_PW], bestfit), fontsize = titlesize)
figBF.savefig(plotdir+'PWContour_bestfit_new.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

axNH.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axNH)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig(plotdir+'PWContour_nullhyp_new.png')

"""
-------------------------------------------------
BROAD STERILE FORMALISM
-------------------------------------------------
"""

print('Broad')

m_n1 = 60
a_n1 = 60
b_n1 = 60
datmass1 = np.logspace(np.log10(0.08),1,m_n1) #0.08-10
datangl1 = np.logspace(np.log10(4e-3),0,a_n1) #0.004-1
datab1 = np.logspace(np.log10(1e-4),np.log10(0.99),b_n1) #fractional breadth
chi2_BS_1 = np.add(np.load(datadir_pros+'BroadSterileChi2_test1.npy'),np.load(datadir_dane+'BroadSterileChi2_test1.npy'))
#chi2_BS_1 = np.add(np.load(datadir_neos+'BroadSterileChi2_frac.npy'),np.load(datadir_daya+'BroadSterileChi2_frac.npy'))

print('PROSPECT best-fit chi2: ', np.min(np.load(datadir_pros+'BroadSterileChi2_test1.npy')))
print('NEOS and DayaBay best-fit chi2: ', np.min(np.load(datadir_dane+'BroadSterileChi2_test1.npy')))

best_mass_1, best_angle_1, bestfit_1, mass_BS_1, angle_BS_1, min_chi2_BS_1, mass_ax_1, angle_ax_1 = marg_over_b(m_n1,a_n1,b_n1,datmass1,datangl1,chi2_BS_1)


print('Reactor global best fit values and chi2: ', best_mass_1, best_angle_1, bestfit_1)

# We find which is the chi2 of the null hypothesis. Use np.sinc in models for this.
null_hyp_BS = getChi2(0,0,0, broad_sterile = True)
print('Null hyp chi2: ',null_hyp_BS)


# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts1 = axBF.tricontour(angle_BS_1,mass_BS_1,(min_chi2_BS_1-bestfit_1),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(best_angle_1,best_mass_1,marker = '+', label = r'Best fit')
# axBF.scatter(data_BS[:,1],data_BS[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF)

figBF.suptitle(r'Best fit:  $\Delta m^2_{41} = %.2f \textrm{ eV}^2$, $\sin^2 2\theta_{14} = %.3f$. Total $\chi^2 = %.2f$'%(best_mass_1,best_angle_1, bestfit_1), fontsize = titlesize)
figBF.savefig(plotdir+'Broad_bestfit_marg_test.png')

# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts1_NH = axNH.tricontour(angle_BS_1, mass_BS_1,(min_chi2_BS_1-null_hyp_BS),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(best_angle_1,best_mass_1,marker = '+', label = r'Best fit')
# axNH.scatter(angle_BS_1[:,1],mass_BS_1[:,0],marker = '+', s = 1.) # This tells us the resolution of our table
stylize(axNH)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_BS), fontsize = titlesize)
figNH.savefig(plotdir+'Broad_nullhyp_marg_test.png')

# ----------------------------------------------
# 2SIGMA PLOT COMPARISON (REACTOR WITH RESPECT TO NULL HYP; BEST WITH RESPECT TO BEST FIT)
# ----------------------------------------------

# Load BEST plane wave data
mass_PW_BEST = np.load(datadir_best+'PWSterileMass.npy') # 0.08-10 in 160 steps
angle_PW_BEST = np.load(datadir_best+'PWSterileAngle.npy') # 0.004-1 in 160 steps
chi2_PW_BEST = np.load(datadir_best+'PWSterileChi2.npy')
# We find which is the point with minimum chi2, i.e. our best fit.
min_index_PW_BEST = np.where(chi2_PW_BEST[:] == np.min(chi2_PW_BEST[:]))[0]
bestfit_PW_BEST = chi2_PW_BEST[min_index_PW_BEST]
print('BEST original best fit values and chi2: ', mass_PW_BEST[min_index_PW_BEST], angle_PW_BEST[min_index_PW_BEST], bestfit_PW_BEST)

# Load BEST broad data
m_n_BEST = 60
a_n_BEST = 60
b_n_BEST = 60
datmassBEST = np.linspace(0.08,10,m_n_BEST) #0.08-10
datanglBEST = np.linspace(4e-3,1,a_n_BEST) #0.004-1
databBEST = np.logspace(np.log10(1e-4),np.log10(0.99),b_n_BEST) #fractional breadth
chi2_BS_BEST = np.load(datadir_best+'BroadSterileChi2_frac.npy')
best_mass_BEST, best_angle_BEST, bestfit_BEST, mass_BS_BEST, angle_BS_BEST, min_chi2_BS_BEST, mass_ax_BEST, angle_ax_BEST = marg_over_b(m_n_BEST,a_n_BEST,b_n_BEST,datmassBEST,datanglBEST,chi2_BS_BEST)
print('BEST broad best fit values and chi2: ', best_mass_BEST, best_angle_BEST, bestfit_BEST)


margins = dict(left=0.16, right=0.97,bottom=0.15, top=0.97)
fig_comp,ax_comp = plt.subplots(figsize = size, gridspec_kw = margins)
cont_PW = ax_comp.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [6.18],  colors = color2, linestyles = ['solid'], zorder = 10)
ax_comp.scatter(angle_PW[min_index_PW],mass_PW[min_index_PW],marker = '+', color='r', label = r'Original best fit')
cont_BS1 = ax_comp.tricontour(angle_BS_1, mass_BS_1,(min_chi2_BS_1-null_hyp_BS),levels = [6.18],  colors = color3, linestyles = ['solid'], zorder = 20)
ax_comp.scatter(best_angle_1,best_mass_1,marker = '+', color='c', label = r'Our best fit')
cont_BEST_PW_patch = ax_comp.tricontourf(angle_PW_BEST,mass_PW_BEST,(chi2_PW_BEST-bestfit_PW_BEST), levels = [0.0,6.18], colors = color2, alpha = 0.3, zorder = 6)
cont_BEST_BS_patch = ax_comp.tricontourf(angle_BS_BEST,mass_BS_BEST,(min_chi2_BS_BEST-bestfit_BEST), levels = [0.0,6.18], colors = color3, alpha = 0.3, zorder = 5)
cont_BEST_PW = ax_comp.tricontour(angle_PW_BEST,mass_PW_BEST,(chi2_PW_BEST-bestfit_PW_BEST), levels = [6.18], colors = color2, linewidths = 1,zorder = 6.5)
cont_BEST_BS = ax_comp.tricontour(angle_BS_BEST,mass_BS_BEST,(min_chi2_BS_BEST-bestfit_BEST), levels = [6.18], colors = color3, linewidths = 1,zorder = 5.5)
ax_comp.annotate('Global', xy = (5e-3,0.1), size = 30)
ax_comp.grid(linestyle = '--')
ax_comp.tick_params(axis='x')
ax_comp.tick_params(axis='y')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
ax_comp.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
ax_comp.set_xlim([0.004,1])
ax_comp.set_ylim([0.08,10])
legend_elements = [Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'$2\sigma$ Delta Function'),
                   Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'$2\sigma$ Broad Sterile'),
                   plt.Rectangle((0.1,0.1),0.8,0.8,fc = color2, alpha = 0.3, label=r'BEST $2\sigma$ Delta'),
                   plt.Rectangle((0.1,0.1),0.8,0.8,fc = color3, alpha = 0.3, label=r'BEST $2\sigma$ Broad'),
                   Line2D([0], [0], marker='+', color='r', lw = 0, label='Original Best Fit', markerfacecolor='b', markersize=8),
                   Line2D([0], [0], marker='+', color='c', lw = 0, label='Broad Best Fit', markerfacecolor='b', markersize=8)]
ax_comp.legend(handles = legend_elements, loc = 'upper left', fontsize = 12)

fig_comp.savefig(plotdir+'ContourComparison_nullhyp_new.png')

#print(fitter_dane.get_chi2(Models.BroadSterileFrac(Sin22Th14 = 0.0472, DM2_41 = 1.732, bfrac = 1e-4)))
