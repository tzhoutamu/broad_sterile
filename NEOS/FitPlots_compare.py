import sys
import os
homedir = os.path.realpath(__file__)[:-len('NEOS/FitPlots_compare.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import NEOS

path_to_style= homedir+common_dir
datadir = homedir+'NEOS/PlotData/'
plotdir = homedir+'NEOS/Figures/'
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# -------------------------------------------------------
# PRELIMINAR FUNCTIONS
# -------------------------------------------------------

# We load the NEOS class
fitter = NEOS.Neos()



# We define a function to read the data in PlotData/
# These data are produced by PWSterileFitTableX.py  or WPSterileFitTableX.py
def txt_to_array(filename, sep = ","):
    """
    Input:
    filename (str): the text file containing a matrix
    which we want to read.

    Output:
    A numpy array of the matrix.
    """
    inputfile = open(filename,'r+')
    file_lines = inputfile.readlines()

    mat = []
    for line in file_lines:
        mat.append(line.strip().split(sep))
    mat = np.array(mat).astype(np.float)
    return mat

# Computes the (A12) Chi2 for given parameters.
# This is necessary to compute the chi2 of the null hypothesis
def getChi2(mass,angl,wave_packet = False):
    if wave_packet == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.WavePacketSterile(Sin22Th14 = angl, DM2_41 = mass)

    chi2 = fitter.get_chi2(model, use_HM = False)
    return chi2

# Colorblind-sensitive colors
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'
color4 = '#FFE1B0'  # Even lighter version of #FFB14E
color5 = '#F5C6D5'  # Even lighter version of #EA5F94
color6 = '#BFBFFF'  # Even lighter version of #0000FF


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


titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

# -------------------------------------------------
# STERILE PLANE WAVE CONTOUR - PLANE WAVE FORMALISM
# -------------------------------------------------

print('Plane wave')
# We load the data from our run
mass_PW = np.load(datadir+'PWSterileMass_p2.npy')
angle_PW = np.load(datadir+'PWSterileAngle_p2.npy')
chi2_PW = np.load(datadir+'PWSterileChi2_p2.npy')

# We load the data from theirs
data_PW = txt_to_array(datadir+'PWSterileChi2.dat')

# We find which is the point with minimum chi2, i.e. our best fit.
min_index = np.where(chi2_PW[:] == np.min(chi2_PW[:]))[0]
bestfit = chi2_PW[min_index]
print('Best fit values and chi2: ', mass_PW[min_index], angle_PW[min_index], bestfit)

# their best fit.
min_index2 = np.where(data_PW[:,2] == np.min(data_PW[:,2]))[0][0]
bestfit2 = data_PW[min_index2]

# We find which is the chi2 of the null hypothesis
null_hyp_PW = getChi2(0., 0.)
print('Null hyp chi2: ', null_hyp_PW)

# PLOT WITH RESPECT TO THE BEST FIT
# ----------------------------------
figBF,axBF = plt.subplots(figsize = size,gridspec_kw=margins)

conts = axBF.tricontour(angle_PW,mass_PW,(chi2_PW-bestfit),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axBF.scatter(angle_PW[min_index],mass_PW[min_index],marker = '+', label = r'Best fit')

conts2 = axBF.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-bestfit2[2]),levels = [2.30,6.18,11.83],  colors = [color4,color5,color6])
axBF.scatter(bestfit2[1],bestfit2[0],marker = '+', label = r'Best fit')
# axBF.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table

stylize(axBF,conts)
stylize(axBF,conts2)

figBF.suptitle(r'Best fit: total $\chi^2 = %.2f$ vs. Original $%.2f$'%(bestfit,bestfit2[2]), fontsize = titlesize)
figBF.savefig(plotdir+'PWContour_bestfit_compare.png')


# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# -----------------------------------------

figNH,axNH = plt.subplots(figsize = size, gridspec_kw = margins)

conts = axNH.tricontour(angle_PW, mass_PW,(chi2_PW-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color1,color2,color3])
axNH.scatter(angle_PW[min_index],mass_PW[min_index],marker = '+', label = 'Our best fit')
# axNH.scatter(data_PW[:,1],data_PW[:,0],marker = '+', s = 1.) # This tells us the resolution of our table
conts2 = axNH.tricontour(data_PW[:,1],data_PW[:,0],(data_PW[:,2]-null_hyp_PW),levels = [2.30,6.18,11.83],  colors = [color4,color5,color6])
axNH.scatter(bestfit2[1],bestfit2[0],marker = '+', label = r'Best fit')

stylize(axNH,conts)
stylize(axNH,conts2)

figNH.suptitle('Null hypothesis: total $\chi^2 = %.2f$'%(null_hyp_PW), fontsize = titlesize)
figNH.savefig(plotdir+'PWContour_nullhyp_compare.png')
