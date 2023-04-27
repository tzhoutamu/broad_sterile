import sys
import os

homedir = os.path.realpath(__file__)[:-len('Misc/2sigmaComparison.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+'NEOS/')
sys.path.append(homedir+'DayaBay/')
sys.path.append(homedir+'PROSPECT/')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D

import Models
import PROSPECT as PS
import NEOS
import DayaBay as DB
#from BroadFitTable import m_n, a_n, b_n, datmass1, datangl1

path_to_style= homedir+common_dir
datadir_pros = homedir+'PROSPECT/PlotData/'
datadir_neos = homedir+'NEOS/PlotData/'
datadir_daya = homedir+'DayaBay/PlotData/'
plotdir = homedir+'Misc/2sigmaPlots/'
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

def getChi2_neos(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)

    chi2 = fitter_neos.get_chi2(model)
    return chi2

def getChi2_daya(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)

    chi2 = fitter_daya.get_poisson_chi2(model)
    return chi2

def getChi2_pros(mass,angl,b,broad_sterile = False):
    if broad_sterile == False:
        model = Models.PlaneWaveSterile(Sin22Th14 = angl, DM2_41 = mass)
    else:
        model = Models.BroadSterileNull(Sin22Th14 = angl, DM2_41 = mass, bvalue = b)

    chi2 = fitter_pros.get_chi2(model)
    return chi2

# We apply a common style to all plots
def stylize(axxis):
    axxis.grid(linestyle = '--')
    axxis.tick_params(axis='x')
    axxis.tick_params(axis='y')
    axxis.set_xscale('log')
    axxis.set_yscale('log')
    axxis.set_ylabel(r"$\Delta m^2_{41} (\textrm{eV}^2)$", fontsize = 24)
    axxis.set_xlabel(r"$\sin^2 2 \theta_{14}$", fontsize = 24)
    axxis.set_xlim([0.004,1])
    axxis.set_ylim([0.08,10])
    legend_elements = [Line2D([0], [0], color=color1, ls = '-', lw=2, label=r'\text{DayaBay}'),
                       Line2D([0], [0], color=color2, ls = '-', lw=2, label=r'\text{NEOS}'),
                       Line2D([0], [0], color=color3, ls = '-', lw=2, label=r'\text{PROSPECT}'),
                       Line2D([0], [0], color=color4, ls='-', lw=2, label=r'\text{DayaBay+NEOS}')]
    axxis.legend(handles = legend_elements, loc = 'upper left', fontsize = 16)


# Colorblind-sensitive colors
color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#0000FF'
color4 = '#8C3434'

titlesize = 13.
size = (7,7)
margins = dict(left=0.16, right=0.97,bottom=0.1, top=0.93)

#chi2_PW = np.add(np.load(datadir_pros+'PWSterileChi2_noint.npy'),np.load(datadir_neos+'PWSterileChi2_p2.npy'),np.load(datadir_daya+'PWSterileChi2_noint.npy'))

"""
Data form DayaBay
"""
print('DayaBay')
mass_DB = np.load(datadir_daya+'PWSterileMass_noint.npy')
angle_DB = np.load(datadir_daya+'PWSterileAngle_noint.npy')
chi2_DB = np.load(datadir_daya+'PWSterileChi2_noint.npy')
# We find which is the point with minimum chi2, i.e. our best fit.
min_index_DB = np.where(chi2_DB[:] == np.min(chi2_DB[:]))[0]
print('Best fit values and chi2: ', mass_DB[min_index_DB], angle_DB[min_index_DB], chi2_DB[min_index_DB])
# We find which is the chi2 of the null hypothesis
null_hyp_DB = getChi2_daya(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_DB)

"""
Data from NEOS
"""
print('NEOS')
mass_NE = np.load(datadir_neos+'PWSterileMass_int.npy')
angle_NE = np.load(datadir_neos+'PWSterileAngle_int.npy')
chi2_NE = np.load(datadir_neos+'PWSterileChi2_int.npy')
# We find which is the point with minimum chi2, i.e. our best fit.
min_index_NE = np.where(chi2_NE[:] == np.min(chi2_NE[:]))[0]
print('Best fit values and chi2: ', mass_NE[min_index_NE], angle_NE[min_index_NE], chi2_NE[min_index_NE])
# We find which is the chi2 of the null hypothesis
null_hyp_NE = getChi2_neos(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_NE)

"""
Data from PROSPECT
"""
print('PROSPECT')
mass_PR = np.load(datadir_pros+'PWSterileMass_noint.npy')
angle_PR = np.load(datadir_pros+'PWSterileAngle_noint.npy')
chi2_PR = np.load(datadir_pros+'PWSterileChi2_noint.npy')
# We find which is the point with minimum chi2, i.e. our best fit.
min_index_PR = np.where(chi2_PR[:] == np.min(chi2_PR[:]))[0]
print('Best fit values and chi2: ', mass_PR[min_index_PR], angle_PR[min_index_PR], chi2_PR[min_index_PR])
# We find which is the chi2 of the null hypothesis
null_hyp_PR = getChi2_pros(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_PR)

"""
Data from DayaBay+NEOS
"""
print('DayaBay+NEOS')
mass_DN = np.load(datadir_neos+'PWSterileMass_int.npy')
angle_DN = np.load(datadir_neos+'PWSterileAngle_int.npy')
chi2_DN = np.add(np.load(datadir_neos+'PWSterileChi2_int.npy'),np.load(datadir_daya+'PWSterileChi2_noint.npy'))
# We find which is the point with minimum chi2, i.e. our best fit.
min_index_DN = np.where(chi2_DN[:] == np.min(chi2_DN[:]))[0]
print('Best fit values and chi2: ', mass_DN[min_index_DN], angle_DN[min_index_DN], chi2_DN[min_index_DN])
# We find which is the chi2 of the null hypothesis
null_hyp_DN = getChi2_neos(0,0,0,broad_sterile=False)+getChi2_daya(0,0,0,broad_sterile=False)
print('Null hyp chi2: ', null_hyp_DN)

# PLOT WITH RESPECT TO THE NULL HYPOTHESIS
# ----------------------------------
figPW,axPW= plt.subplots(figsize = size,gridspec_kw=margins)
contsDB = axPW.tricontour(angle_DB,mass_DB,(chi2_DB-null_hyp_DB),levels = [6.18],  colors = color1)
contsNE = axPW.tricontour(angle_NE,mass_NE,(chi2_NE-null_hyp_NE),levels = [6.18],  colors = color2)
contsPR = axPW.tricontour(angle_PR,mass_PR,(chi2_PR-null_hyp_PR),levels = [6.18],  colors = color3)
contsDN = axPW.tricontour(angle_DN,mass_DN,(chi2_DN-null_hyp_DN),levels = [6.18],  colors = color4)
axPW.scatter(angle_DB[min_index_DB],mass_DB[min_index_DB],marker = '+', color = color1, label = r'Best fit')
axPW.scatter(angle_NE[min_index_NE],mass_NE[min_index_NE],marker = '+', color = color2, label = r'Best fit')
axPW.scatter(angle_PR[min_index_PR],mass_PR[min_index_PR],marker = '+', color = color3, label = r'Best fit')
axPW.scatter(angle_DN[min_index_DN],mass_DN[min_index_DN],marker = '+', color = color4, label = r'Best fit')
stylize(axPW)
figPW.savefig(plotdir+'PW_2sigma_nullhyp.png')