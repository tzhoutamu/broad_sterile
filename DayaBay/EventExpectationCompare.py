import sys
import os
homedir = os.path.realpath(__file__)[:-len('DayaBay/EventExpectationCompare.py')]
common_dir = 'Common_cython/'
plotdir = homedir + 'DayaBay/Figures/'
sys.path.append(homedir+common_dir)

import time
import DayaBay as DB
import Models

import numpy as np
import matplotlib.pyplot as plt

# We load the models we want to compute the expectations for
fitter = DB.DayaBay()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_coh = Models.WavePacketSM()

# # Broad data
# # ------------
dm2_bro = 0.08
sin2_bro = 0.06
b = 0.53

# # Sterile data
# # ------------
dm2_ste = 0.069
sin2_ste = 0.009


Model_ste = Models.PlaneWaveSterile(DM2_41 = dm2_ste,Sin22Th14 = sin2_ste)
Model_bro = Models.BroadSterileFrac(DM2_41 = dm2_bro,Sin22Th14 = sin2_bro,bfrac = b)


# -------------------------------------------------------------
# INITIAL COMPUTATIONS
# ------------------------------------------------------------

def what_do_we_do(mass):
    """Mass must be in eV^2.

    This functions tells us whether to integrate or to average
    the oscillations for the given mass squared.
    The limits of the regimes are approximate.
    """
    if mass <= 0.15:
        return {'integrate':False,'average':False}
    elif (mass > 0.15) and (mass <= 2.):
        return {'integrate':True, 'average':False}
    elif (mass > 2.):
        return {'integrate':False,'average':True}

wdwd_ste = what_do_we_do(dm2_ste)
wdwd_bro = what_do_we_do(dm2_bro)
begin_time = time.time()

predDB = fitter.get_expectation(Model_osc)
predST = fitter.get_expectation(Model_ste, integrate = wdwd_ste['integrate'], average = wdwd_ste['average'])
predBS = fitter.get_expectation(Model_bro, integrate = wdwd_bro['integrate'], average = wdwd_bro['average'])

end_time = time.time()
print(end_time-begin_time) # Prints the time the computation has taken


# We compute the chi^2 to each experimental hall, separately
chi2_per_exp_ste = []
for exp in fitter.sets_names:
    evex = predST[exp][:,0]
    data = fitter.ObservedData[exp]
    chi2_per_exp_ste.append(np.sum(-2*(data-evex+data*np.log(evex/data))))

print('Original total chi^2: %.2f'%(np.sum(chi2_per_exp_ste)))

chi2_per_exp_bro = []
for exp in fitter.sets_names:
    evex = predBS[exp][:,0]
    data = fitter.ObservedData[exp]
    chi2_per_exp_bro.append(np.sum(-2*(data-evex+data*np.log(evex/data))))

print('Broad total chi^2: %.2f'%(np.sum(chi2_per_exp_bro)))

# We define the prompt energy bins
x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges-fitter.DataLowerBinEdges)

# -------------------------------------------------------
# Event expectations
# -------------------------------------------------------

figev,axev = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

axis = [[1.3,6.9,0.,3.5],[1.3,6.9,0.,3.],[1.3,6.9,0.,0.9]]
norm = [1e5,1e5,1e5]


for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    axev[i].errorbar(x_ax,predST[set][:,0]/deltaE/norm[i], yerr = predST[set][:,1]/deltaE/norm[i], xerr = 0.1, label = "Original prediction", fmt = "_", elinewidth = 2)
    axev[i].errorbar(x_ax,predBS[set][:,0]/deltaE/norm[i], yerr = predBS[set][:,1]/deltaE/norm[i], xerr = 0.1, label = "Broad prediction", fmt = "_", elinewidth = 2)
    axev[i].scatter(x_ax,fitter.ObservedData[set]/deltaE/norm[i], label = "{} data".format(fitter.sets_names[i]),color = "black")

    # Other things to plot: DB prediction of no oscillations,
    # axev[i].errorbar(x_ax,fitter.AllData[DB_test.sets_names[i]][:,5]/deltaE/norm[i], fmt = '_', elinewidth = 2, color = "red", label = "DB no oscillations")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(MeV$\ \cdot 10^{%i}$)"%(np.log10(norm[i])), fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f $(Orange); Broad with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$, $\tilde{b} = %.2f$(Blue)'%(dm2_ste,sin2_ste,dm2_bro,sin2_bro,b), fontsize = 17)
figev.savefig(plotdir+"EventExpectation/EventExpectation_%.2f_%.2f_ste_%.2f_%.2f_bro.png"%(dm2_ste,sin2_ste,dm2_bro,sin2_bro))


# ----------------------------------------------
# EVENT EXPECTATIONS RATIO, HEAVY STERILE VS SM
# -----------------------------------------------

figev,axev = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    ste_dat = predST[set][:,0]
    bro_dat = predBS[set][:,0]
    SM_dat = predDB[set][:,0]

    axev[i].errorbar(x_ax,ste_dat/SM_dat, xerr = 0.1, label = "Heavy sterile/SM", fmt = "_", elinewidth = 2)
    axev[i].errorbar(x_ax,bro_dat/SM_dat, xerr = 0.1, label = "Broad sterile/SM", fmt = "_", elinewidth = 2)
    axev[i].plot(x_ax,np.ones([fitter.n_bins]),linestyle = 'dashed')
    axev[i].errorbar(x_ax,fitter.ObservedData[set]/SM_dat, yerr = np.sqrt(fitter.ObservedData[set])/SM_dat, label = "{} data".format(fitter.sets_names[i]), fmt = "ok")

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Ratio ste/DB", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev[i].axis(axis[i])
    axev[i].grid(linestyle="--")
    axev[i].title.set_text(fitter.sets_names[i])
    axev[i].legend(loc="upper right",fontsize=16)

# figev.suptitle(r'Our best fit: $\Delta m^2_{13} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.07821$', fontsize = 17)
# figev.suptitle(r'DB best fit: $\Delta m^2_{13} = 2.4·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$; Broad with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$, $\tilde{b} = %.2f$'%(dm2_ste,sin2_ste,dm2_bro,sin2_bro,b), fontsize = 17)
figev.savefig(plotdir+"EventRatio/EventRatio_%.2f_%.2f_ste_%.2f_%.2f_bro.png"%(dm2_ste,sin2_ste,dm2_bro,sin2_bro))



# ----------------------------------------------
# CHI2 per bin per experimental hall
# ----------------------------------------------

axis = [[1.3,6.9,0.,1.5],[1.3,6.9,0.,2.],[1.3,6.9,0.,5.5]]


figchi,axchi = plt.subplots(1,len(fitter.sets_names),figsize = (20,8),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.91))

for i in range(len(fitter.sets_names)):
    set = fitter.sets_names[i]
    data = fitter.ObservedData[set]
    evexST = predST[set][:,0]
    evexBS = predBS[set][:,0]
    axchi[i].bar(x_ax,-2*(data-evexST+data*np.log(evexST/data)),width = 3/4*deltaE)
    axchi[i].bar(x_ax,-2*(data-evexBS+data*np.log(evexBS/data)),width = 3/4*deltaE)
    axchi[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axchi[i].set_ylabel(r"%s $\chi^2$ per bin"%(set), fontsize = 16)
    axchi[i].tick_params(axis='x', labelsize=13)
    axchi[i].tick_params(axis='y', labelsize=13)
    # axchi[i].axis(axis[i])
    axchi[i].grid(linestyle="--")
    axchi[i].title.set_text(set+r' sterile total $\chi^2 = %.2f $ (Blue); broad total $\chi^2 = %.2f $ (Orange)'%(chi2_per_exp_ste[i],chi2_per_exp_bro[i]))

    axchi[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$; Broad with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$, $\tilde{b} = %.2f$'%(dm2_ste,sin2_ste,dm2_bro,sin2_bro,b), fontsize = 17)
figchi.savefig(plotdir+"Chi2/Chi2_%.2f_%.2f_ste_%.2f_%.2f_bro.png"%(dm2_ste,sin2_ste,dm2_bro,sin2_bro))
