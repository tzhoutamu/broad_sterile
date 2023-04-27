import sys
import os
homedir = os.path.realpath(__file__)[:-len('Misc/ProbabilityPlots.py')]
common_dir = 'Common_cython'
sys.path.append(homedir+common_dir)
sys.path.append(homedir+"/NEOS")
sys.path.append(homedir+"/DayaBay")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import Models

path_to_style = homedir + common_dir
plt.style.use(path_to_style+r"/paper.mplstyle")
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

color1 = '#FFB14E'
color2 = '#EA5F94'
color3 = '#00BFFF'
color4 = '#32CD32'
color5 = '#800080'


C = 80
dm_DB = 0.25
dm_NEOS = C/24
# print(dm_NEOS)
angl = 0.4
SM_PW = Models.PlaneWaveSM()
SM_WP = Models.WavePacketSM()
DB_PW = Models.PlaneWaveSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
DB_WP = Models.WavePacketSterile(DM2_41 = dm_DB, Sin22Th14 = angl)
NEOS_PW = Models.PlaneWaveSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)
NEOS_WP = Models.WavePacketSterile(DM2_41 = dm_NEOS, Sin22Th14 = angl)

# Oscillation parameters for PROSPECT
dm_PROS = 10 #0.57
angl_PROS = 0.27
b1 = 0.53
b2 = 0.69
b3 = 0.89
b4 = 0.99
L_PROS = 7 # Baseline of PROSPECT (m)
PROS_PW = Models.PlaneWaveSterile(DM2_41 = dm_PROS, Sin22Th14 = angl_PROS)
PROS_BS_b1 = Models.BroadSterileFrac(DM2_41 = dm_PROS, Sin22Th14 = angl_PROS, bfrac = b1)
PROS_BS_b2 = Models.BroadSterileFrac(DM2_41 = dm_PROS, Sin22Th14 = angl_PROS, bfrac = b2)
PROS_BS_b3 = Models.BroadSterileFrac(DM2_41 = dm_PROS, Sin22Th14 = angl_PROS, bfrac = b3)
PROS_BS_b4 = Models.BroadSterileFrac(DM2_41 = dm_PROS, Sin22Th14 = angl_PROS, bfrac = b4)

L_DB = C/0.25
# print(L_DB)
L_NEOS = 24
# print(L_NEOS*dm_NEOS)

axis = [1.3,16.,0.85,1]
datx = np.arange(axis[0],axis[1],0.001)

prob_DB_PW =  [DB_PW.oscProbability(x,L_DB)/SM_PW.oscProbability(x,L_DB) for x in datx]
prob_DB_WP =  [DB_WP.oscProbability(x,L_DB)/SM_WP.oscProbability(x,L_DB) for x in datx]
prob_NEOS_PW =  [NEOS_PW.oscProbability(x,L_NEOS)/SM_PW.oscProbability(x,L_NEOS) for x in datx]
prob_NEOS_WP =  [NEOS_WP.oscProbability(x,L_NEOS)/SM_WP.oscProbability(x,L_NEOS) for x in datx]

prob_PROS_PW = [PROS_PW.oscProbability(x,L_PROS) for x in datx]
prob_PROS_BS_b1 = [PROS_BS_b1.oscProbability(x,L_PROS) for x in datx]
prob_PROS_BS_b2 = [PROS_BS_b2.oscProbability(x,L_PROS) for x in datx]
prob_PROS_BS_b3 = [PROS_BS_b3.oscProbability(x,L_PROS) for x in datx]
prob_PROS_BS_b4 = [PROS_BS_b4.oscProbability(x,L_PROS) for x in datx]

margins = dict(left=0.14, right=0.96,bottom=0.19, top=0.9)
plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)

#print(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6)

# axx.plot(datx,prob_DB_PW)
# axx.plot(datx,prob_DB_WP)
axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_xlim([1.3,8])
#axx.set_xlim([axis[0],axis[1]])
axx.set_ylim([0.7,1.])
axx.set_ylabel(r"$P_{3+1}$", fontsize = 24)
axx.set_xlabel(r"$E (\textrm{MeV})$", fontsize = 24)
axx.plot(datx,prob_PROS_PW, label = r'$\tilde{b} = 0$', color = color1)
axx.plot(datx,prob_PROS_BS_b1, label = r'$\tilde{b} = %.2f$'%(b1), color = color2)
axx.plot(datx,prob_PROS_BS_b2, label = r'$\tilde{b} = %.2f$'%(b2), color = color3)
axx.plot(datx,prob_PROS_BS_b3, label = r'$\tilde{b} = %.2f$'%(b3), color = color4)
axx.plot(datx,prob_PROS_BS_b4, label = r'$\tilde{b} = %.2f$'%(b4), color = color5)
#axx.annotate(r'$E = E^{\textrm{coh}}$',(8.3,0.97), size = 20, color = 'k')
#axx.vlines(np.sqrt(L_NEOS*dm_NEOS/(4*np.sqrt(2)*2.1e-13))*1e-6, ymin = 0.6, ymax = 1., linestyle = '--', color = 'gray')
#axx.legend(bbox_to_anchor=(0,-0.31, 1, 0), loc="lower center", ncol = 2, fontsize = 20)
axx.legend(bbox_to_anchor=(0.66, 1), loc="upper left", ncol=1, fontsize=15)
plot.suptitle(r'$m^2_{41} = %.2f\ \textrm{eV}^2, \ \sin^2 2\theta_{14} = %.2f$'%(dm_PROS, angl_PROS))
plot.savefig(homedir+'Misc/Probability_b_10.png')
'''
# -----------------------------------------------------------------

dm = 0.2
sin2 = 0.1
SM_PW = Models.PlaneWaveSM()
PW = Models.PlaneWaveSterile(DM2_41 = dm, Sin22Th14 = sin2)
WP = Models.WavePacketSterile(DM2_41 = dm, Sin22Th14 = sin2)

E = 20
axis = [1.3,16.,0.85,1]
datx = np.arange(400,2500,1)
# L = ratio*E
sigmax = 2.1e-13
#Lcoh = 4*np.sqrt(2)*sigmax/(dm*1e-12)*E
#print(Lcoh)

prob_SM =  [SM_PW.oscProbability(E,x) for x in datx]
prob_PW =  [PW.oscProbability(E,x) for x in datx]
prob_WP =  [WP.oscProbability(E,x) for x in datx]

plot,axx =plt.subplots(figsize = (7,6),gridspec_kw=margins)
axx.plot(datx,prob_PW, label = 'Plane Wave', color = color2)
axx.plot(datx,prob_WP, label = 'Wave packet', color = color3)

#axx.grid(linestyle = '--')
axx.tick_params(axis='x')
axx.tick_params(axis='y')
axx.set_ylabel(r"$\textrm{Probability}$", fontsize = 24)
axx.set_xlabel(r"$L (\textrm{km})$", fontsize = 24)
#axx.set_xlim([0,1500])
#axx.set_ylim([0.0,1.0])

#axx.legend(bbox_to_anchor=(0,-0.31, 1, 0), loc="lower center", ncol = 2, fontsize = 20)
#axx.vlines(Lcoh, ymin = 0.9, ymax = 1., linestyle = '--', color = 'gray')
#axx.annotate(r'$L = L^{\textrm{coh}}$',(Lcoh*1.05,0.99), size = 20, color = 'k')

#plot.suptitle(r'$E_\nu = %.1f\textrm{ MeV},\ \Delta m^2_{41} = 0.2\textrm{eV}^2,\ \sin^2 2\theta_{14} = 0.1$'%(E))
plot.savefig(homedir+'Misc/Probability_L.png')
'''