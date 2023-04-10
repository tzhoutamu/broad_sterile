# This is an adaptation of the original code optimized for parallel computation using 'multiprocessing'

import multiprocessing as mp
import FitClass as FC
import numpy as np
import time
import os

homedir = os.path.realpath(__file__)[:-len('NEOS/BroadFitTable.py')]
datadir = homedir + 'NEOS/PlotData/'


# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study

# create arrays of mass and angle evenly spaced on a logarithmic scale.
m_n = 60
a_n = 60
b_n = 60
datmass1 = np.logspace(np.log10(0.08),np.log10(2),m_n) #0.08-2
datangl1 = np.logspace(np.log10(4e-3),0,a_n) #0.004-1
datab1 = np.logspace(np.log10(1e-4),np.log10(0.99),b_n) #fractional breadth

# meshgrid returns 2 2-dimensional arrays that represent the x and y coordinates of all points in the grid.
mass_grid, angle_grid, b_grid = np.meshgrid(datmass1, datangl1, datab1)
# ravel returns the same array but with all internal brackets removed.
mass_data, angle_data, b_data = np.ravel(mass_grid), np.ravel(angle_grid), np.ravel(b_grid)
# empty_like returns a new array with the same shape and data type as the input array, but with undefined or uninitialized values.
chi2_data = np.empty_like(mass_data)

fit = FC.BroadFit(broad_sterile = True, use_HM = False)
np.save(datadir+'BroadSterileMass_frac.npy', mass_data)
np.save(datadir+'BroadSterileAngle_frac.npy', angle_data)
np.save(datadir+'BroadSterileb_frac.npy', b_data)

def job(i):
    m = mass_data[i]
    a = angle_data[i]
    b = b_data[i]
    chi2_data[i] = fit.getChi2(m,a,b)

if __name__ == '__main__':
    begin = time.time()
    # creating a shared memory array that can be accessed by multiple processes.
    chi2_data = mp.Array('f', range(mass_data.shape[0]))
    # use the line below for personal machine. It will use the maximum amount of cores available.
    pool = mp.Pool()
    # use lines below for HPC runs
    #num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
    #pool = mp.Pool(num_cores)
    pool.map(job, range(mass_data.shape[0]))
    np.save(datadir+'BroadSterileChi2_frac.npy', chi2_data[:])
    end = time.time()
    pool.close()
    # print out total run time
    print('Time = '+str(end-begin)[:6]+' s.')
    # total run time in hours
    print('Time = '+ str((end-begin)/3600)[:6]+' h')

