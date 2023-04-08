import multiprocessing as mp
import FitClass as FC
import numpy as np
import time
import os

homedir = os.path.realpath(__file__)[:-len('PROSPECT/PWSterileFitTable2.py')]
# The data is saved inside BEST/PlotData
datadir = homedir + 'PROSPECT/PlotData/'

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# create arrays of mass and angle evenly spaced on a logarithmic scale.
datangl1 = np.logspace(np.log10(4e-3),0,150)
datmass1 = np.logspace(np.log10(0.08),0,160)

# meshgrid returns 2 2-dimensional arrays that represent the x and y coordinates of all points in the grid.
mass_grid, angle_grid = np.meshgrid(datmass1, datangl1)
# ravel returns the same array but with all internal brackets removed.
mass_data, angle_data = np.ravel(mass_grid), np.ravel(angle_grid)
# empty_like returns a new array with the same shape and data type as the input array, but with undefined or uninitialized values.
chi2_data = np.empty_like(mass_data)

fit = FC.SterileFit(wave_packet = False) # Here we choose PW formalism
np.save(datadir + 'PWSterileMass.npy', mass_data)
np.save(datadir + 'PWSterileAngle.npy', angle_data)

def job(i):
    m = mass_data[i]
    a = angle_data[i]
    chi2_data[i] = fit.getChi2(m,a)

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
    np.save(datadir + 'PWSterileChi2.npy', chi2_data[:])
    end = time.time()
    pool.close()
    # print out total run time
    print('Time = '+str(end-begin)[:6]+' s.')
    # total run time in hours
    print('Time = '+ str((end-begin)/3600)[:6]+' h')

