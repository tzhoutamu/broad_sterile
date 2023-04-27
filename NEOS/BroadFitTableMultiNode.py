import multiprocessing as mp
import numpy as np
import time
import os
from mpi4py import MPI
import FitClass as FC

homedir = os.path.realpath(__file__)[:-len('NEOS/BroadFitTableMultiNode.py')]
datadir = homedir + 'NEOS/PlotData/'

# This is an example program of how to obtain points of the chi2
# Here we use the variables and functions from FitClass.py

# Tune these arrays to the interval of parameters you wish to study

# create arrays of mass and angle evenly spaced on a logarithmic scale.
m_n = 10
a_n = 10
b_n = 10
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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Split mass_data into chunks for each MPI process
angle_data_split = np.array_split(angle_data, size)
angle_data_local = angle_data_split[rank]
mass_data_local = mass_data
b_data_local = b_data
chi2_data_local = np.empty_like(angle_data_local)

# Initialize multiprocessing pool
pool = mp.Pool(48)

def job(i):
    m = mass_data_local[i]
    a = angle_data_local[i]
    b = b_data_local[i]
    chi2_data_local[i] = fit.getChi2(m,a,b)


if __name__ == '__main__':
    begin = time.time()

    # Use multiprocessing pool to compute chi2 values in parallel
    pool.map(job, range(len(angle_data_local)))

    # Gather chi2 data from all MPI processes
    comm.Allgather(chi2_data_local, chi2_data)

    end = time.time()

    if rank == 0:
        print("Total time taken: {:.2f} seconds".format(end - begin))
        print("Saving data...")
        # Save the computed chi2 values to file
        np.save(datadir + 'BroadSterileChi2_frac.npy', chi2_data)
        print("Data saved successfully!")
