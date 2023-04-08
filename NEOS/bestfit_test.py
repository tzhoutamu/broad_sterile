import FitClass as FC
import numpy as np
import time


bestmass = 1.71877
bestangle = 0.07131

fit = FC.SterileFit(wave_packet = False, use_HM = False)
chi2 = fit.getChi2(bestmass,bestangle)
print(chi2)

