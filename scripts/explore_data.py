# Making exploratory plots of the data

# %%
# imports -------------------------------------------------------------------------------------------------------------

# base python -------------------------------------------------------------

import sys
import os
import multiprocessing
import pickle
import time
from time import strftime, localtime
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# packages ----------------------------------------------------------------

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import scipy
import gstools           as gs
import geopandas         as gpd
import rpy2.robjects     as robjects
from   rpy2.robjects import r
from   rpy2.robjects.numpy2ri import numpy2rpy
from   rpy2.robjects.packages import importr

# %%
# Load data

datafolder = '../data/realdata/'
datafile   = 'JJA_precip_nonimputed.RData'

r(f'''
    load("{datafolder}/{datafile}")
''')

Y = np.array(r('Y'))
GP_estimates = np.array(r('GP_estimates')).T
logsigma_estimates = GP_estimates[:,1]
ksi_estimates      = GP_estimates[:,2]
elevations         = np.array(r('elev'))
stations           = np.array(r('stations')).T

# this `u_vec` is the threshold,
# spatially varying but temporally constant
# ie, each site has its own threshold
u_vec              = GP_estimates[:,0]


# %%
# Exploratory analysis - sigma(s) surface

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Generate the grid for interpolation
grid_x, grid_y = np.mgrid[
    np.min(stations[:, 0]):np.max(stations[:, 0]):100j,  # Define grid range and resolution
    np.min(stations[:, 1]):np.max(stations[:, 1]):100j
]

# Interpolate the data
grid_z = griddata(stations, logsigma_estimates, (grid_x, grid_y), method='cubic')

# Create the heatplot
plt.figure(figsize=(8, 6))
plt.imshow(grid_z.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                             np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='OrRd', aspect='auto')
plt.colorbar(label='Logsigma Estimates')
plt.title('Smoothed Heatmap of Logsigma Estimates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Show the plot
plt.show()

# A different smooth

from scipy.ndimage import gaussian_filter

# Interpolate the data
grid_z = griddata(stations, logsigma_estimates, (grid_x, grid_y), method='linear')

# Apply Gaussian filter for smoothing
grid_z_smooth = gaussian_filter(grid_z, sigma=3.0)  # Adjust sigma for more or less smoothing

# Create the heatmap with smoothed data
plt.imshow(grid_z_smooth.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                                    np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Logsigma Estimates (Smoothed)')
plt.title('Smoothed Heatmap with Gaussian Filter')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()


# %%
# Exploratory analysis - xi(s) surface

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Generate the grid for interpolation
grid_x, grid_y = np.mgrid[
    np.min(stations[:, 0]):np.max(stations[:, 0]):100j,  # Define grid range and resolution
    np.min(stations[:, 1]):np.max(stations[:, 1]):100j
]

# Interpolate the data
grid_z = griddata(stations, ksi_estimates, (grid_x, grid_y), method='cubic')

# Create the heatplot
plt.figure(figsize=(8, 6))
plt.imshow(grid_z.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                             np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='OrRd', aspect='auto')
plt.colorbar(label='Logsigma Estimates')
plt.title('Smoothed Heatmap of Logsigma Estimates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Show the plot
plt.show()

# A different smooth

from scipy.ndimage import gaussian_filter

# Interpolate the data
grid_z = griddata(stations, ksi_estimates, (grid_x, grid_y), method='linear')

# Apply Gaussian filter for smoothing
grid_z_smooth = gaussian_filter(grid_z, sigma=3.0)  # Adjust sigma for more or less smoothing

# Create the heatmap with smoothed data
plt.imshow(grid_z_smooth.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                                    np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Logsigma Estimates (Smoothed)')
plt.title('Smoothed Heatmap with Gaussian Filter')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()



# %%
# Exploratory analysis - elev(s) surface

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Generate the grid for interpolation
grid_x, grid_y = np.mgrid[
    np.min(stations[:, 0]):np.max(stations[:, 0]):100j,  # Define grid range and resolution
    np.min(stations[:, 1]):np.max(stations[:, 1]):100j
]

# Interpolate the data
grid_z = griddata(stations, elevations, (grid_x, grid_y), method='cubic')

# Create the heatplot
plt.figure(figsize=(8, 6))
plt.imshow(grid_z.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                             np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='OrRd', aspect='auto')
plt.colorbar(label='Logsigma Estimates')
plt.title('Smoothed Heatmap of Logsigma Estimates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Show the plot
plt.show()

# A different smooth

from scipy.ndimage import gaussian_filter

# Interpolate the data
grid_z = griddata(stations, elevations, (grid_x, grid_y), method='linear')

# Apply Gaussian filter for smoothing
grid_z_smooth = gaussian_filter(grid_z, sigma=3.0)  # Adjust sigma for more or less smoothing

# Create the heatmap with smoothed data
plt.imshow(grid_z_smooth.T, extent=(np.min(stations[:, 0]), np.max(stations[:, 0]),
                                    np.min(stations[:, 1]), np.max(stations[:, 1])),
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Logsigma Estimates (Smoothed)')
plt.title('Smoothed Heatmap with Gaussian Filter')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
# %%
