import sys
import os
homedir = os.path.realpath(__file__)[:-len('PROSPECT/EventExpectationPlots.py')]
common_dir = 'Common_cython/'
plotdir = homedir + 'PROSPECT/Figures/'
sys.path.append(homedir+common_dir)

import time
import PROSPECT as PS
import Models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


fitter = PS.Prospect()
Model_noosc = Models.NoOscillations()
Model_osc = Models.PlaneWaveSM()
# Model_coh = Models.WavePacketSM()


# # Broad data
# # ------------
dm2_bro = 0.57
sin2_bro = 0.27
b = 0.99

# # Sterile data
# # ------------
dm2_ste = 0.602
sin2_ste = 0.169

Model_ste = Models.PlaneWaveSterile(DM2_41 = dm2_ste,Sin22Th14 = sin2_ste)
Model_bro = Models.BroadSterileFrac(DM2_41 = dm2_bro,Sin22Th14 = sin2_bro,bfrac = b)
#Model_ste = Models.WavePacketSterile(DM2_41 = dm2,Sin22Th14 = sin2)
#Model_ste = Models.BroadSterileFrac(DM2_41 = dm2,Sin22Th14 = sin2,bfrac = b)

# -----------------------------------------------------
# PRELIMINAR COMPUTATIONS
# -----------------------------------------------------

begin_time = time.time()

predSM = fitter.get_expectation(Model_osc, do_we_integrate = False)
predST = fitter.get_expectation(Model_ste, do_we_integrate = True)
predBS = fitter.get_expectation(Model_bro, do_we_integrate = True)

end_time = time.time()
print(end_time-begin_time)


data = fitter.get_data_per_baseline()
Me = 0.0
PeST = 0.0
PeBS = 0.0
PeSM = 0.0
for bl in fitter.Baselines:
    Me += np.sum(data[bl])
    PeST += np.sum(predST[bl])
    PeBS += np.sum(predBS[bl])
    PeSM += np.sum(predSM[bl])

bkg = fitter.get_bkg_per_baseline()


# -------------------------------------------------------
# Event expectations in SM - total
# -------------------------------------------------------

x_ax = (fitter.DataLowerBinEdges+fitter.DataUpperBinEdges)/2
deltaE = (fitter.DataUpperBinEdges -fitter.DataLowerBinEdges)

figSM,axSM = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axSM = axSM.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axSM[i].step(x_ax,predSM[bl],where = 'mid', label = "Our prediction" )
    axSM[i].step(x_ax,fitter.PredictedData[bl], where = 'mid', label = "No oscillation prediction")
    axSM[i].errorbar(x_ax,data[bl], fmt = 'ok', label = "PROSPECT data")
    axSM[i].errorbar(x_ax,bkg[bl], xerr = 0.1, label = "NEOS background", fmt = "_", elinewidth = 2)
    axSM[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axSM[i].set_ylabel("Events/(0.1 MeV)", fontsize = 16)
    axSM[i].tick_params(axis='x', labelsize=13)
    axSM[i].tick_params(axis='y', labelsize=13)
    # axSM.axis([1.,7.,0.,60.])
    axSM[i].grid(linestyle="--")
    # axSM[i].legend(loc="upper right",fontsize=16)

figSM.suptitle(r'SM fit: $\Delta m^2_{31} = 2.5·10^{-3} eV^2$, $\sin^2 2\theta_{13} = 0.0841$', fontsize = 17)
figSM.savefig(plotdir+"EventExpectation/EventExpectation_SM.png")


# -------------------------------------------------------
# Event expectations - total
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axev[i].step(x_ax,predBS[bl], where='mid', label="Broad prediction", linewidth=2)
    axev[i].step(x_ax,predST[bl], where = 'mid', label = "Original prediction", linewidth = 2 )
    axev[i].step(x_ax,predSM[bl], where = 'mid', label = "No oscillation prediction", linewidth = 2)
    axev[i].errorbar(x_ax,data[bl], fmt = 'ok', label = "PROSPECT data")
    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/(0.1 MeV)", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    # axev.axis([1.,7.,0.,60.])
    axev[i].grid(linestyle="--")
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f $(Orange); Broad with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$, $\tilde{b} = %.2f$(Blue)'%(dm2_ste,sin2_ste,dm2_bro,sin2_bro,b), fontsize = 17)
figev.savefig(plotdir+"EventExpectation/EventExpectation_%.2f_%.2f_ste_%.2f_%.2f_bro.png"%(dm2_ste,sin2_ste,dm2_bro,sin2_bro))


# -------------------------------------------------------
# Event expectations - ratio to SM
# -------------------------------------------------------

figev,axev = plt.subplots(2,5,figsize = (21,10),gridspec_kw=dict(left=0.05, right=0.98,bottom=0.1, top=0.93))
axev = axev.flatten()

for bl in fitter.Baselines:
    i = bl-1
    axev[i].step(x_ax,predBS[bl]*PeSM/PeBS/predSM[bl], where = 'mid', label = "Broad prediction", linewidth = 2 )
    axev[i].step(x_ax,predST[bl]*PeSM/PeST/predSM[bl], where = 'mid', label = "Original prediction", linewidth = 2 )
    axev[i].errorbar(x_ax,data[bl]/Me*PeBS/predBS[bl], fmt = 'ok', label = "PROSPECT data based on broad prediction")
    axev[i].errorbar(x_ax, data[bl] / Me * PeST / predST[bl], fmt='ok', label="PROSPECT data based on original prediction")

    axev[i].plot(x_ax,[1 for x in x_ax], linestyle = 'dashed', color = 'k', zorder = 0.01)

    axev[i].set_xlabel("Energy (MeV)", fontsize = 16)
    axev[i].set_ylabel("Events/EventsSM", fontsize = 16)
    axev[i].tick_params(axis='x', labelsize=13)
    axev[i].tick_params(axis='y', labelsize=13)
    axev[i].axis([0.8,7.2,0.0,2.0])
    axev[i].grid(linestyle="--")
    axev[i].legend(loc="upper right",fontsize=16)

figev.suptitle(r'Sterile with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$; Broad with $\Delta m^2_{41} = %.2f eV^2$, $\sin^2 2\theta_{14} = %.2f$, $\tilde{b} = %.2f$'%(dm2_ste,sin2_ste,dm2_bro,sin2_bro,b), fontsize = 17)
figev.savefig(plotdir+"EventRatio/EventRatio_%.2f_%.2f_ste_%.2f_%.2f_bro.png"%(dm2_ste,sin2_ste,dm2_bro,sin2_bro))
