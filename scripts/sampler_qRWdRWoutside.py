"""
MCMC Sampler for Spatial Extreme Model

Date: March 05, 2025
Author: Muyang Shi

Description:
  The qRW and dRW calls are done *outside* the likelihood function to
  let a neural net handle the qRW and reduce the number of dRW calls.
"""

# %% IMPORTS ----------------------------------------------------------------------------------------------------------

# base python -----------------------------------------------------------------

import sys
import os
import multiprocessing
import pickle
import time
from time import strftime, localtime
from pathlib import Path
os.environ["OMP_NUM_THREADS"]        = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"]   = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"]        = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"]    = "1" # export NUMEXPR_NUM_THREADS=1
os.environ["KMP_AFFINITY"]           = "disabled" # export KMP_AFFINITY=disabled

# packages --------------------------------------------------------------------

from mpi4py import MPI
comm             = MPI.COMM_WORLD
rank             = comm.Get_rank()
size             = comm.Get_size()

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import scipy
import gstools           as gs
import geopandas         as gpd
import rpy2.robjects     as robjects
from   rpy2.robjects          import r
from   rpy2.robjects.numpy2ri import numpy2rpy
from   rpy2.robjects.packages import importr

# custom module ---------------------------------------------------------------

from utilities import *

# load settings from environment ----------------------------------------------

if rank == 0:
    print('link function:', norm_pareto, 'Pareto')
    state_map = gpd.read_file('./cb_2018_us_state_20m/cb_2018_us_state_20m.shp')

# Random number generator
random_generator = np.random.RandomState((rank+1)*7)
try:
    data_seed = int(sys.argv[1])
except:
    data_seed = 2345
finally:
    if rank == 0: print('data_seed:', data_seed)
    np.random.seed(data_seed)

# Load iter from pickle
try:
    with open('iter.pkl','rb') as file:
        start_iter = pickle.load(file) + 1
        if rank == 0: print('start_iter loaded from pickle, set to be:', start_iter)
except Exception as e:
    if rank == 0:
        print('Exception loading iter.pkl:', e)
        print('Setting start_iter to 1')
    start_iter = 1

# Manual MCMC chain settings --------------------------------------------------

from_simulation = True
n_iters         = 20000
SAVE_SIZE       = 50   # we are not saving the counters, so this must be a multiple of ADAPT_SIZE!
THIN            = 10   # print to console every `THIN` iterations

# Adaptive Update: tuning constants

C0             = 1
C1             = 0.8
OFFSET         = 3
R_OPT          = .35
ADAPT_SIZE     = 25

assert SAVE_SIZE % ADAPT_SIZE == 0 # SAVE_SIZE must be a multiple of ADAPT_SIZE

# Block Update Specification

phi_block_idx_size = 1
rho_block_idx_size = 1
Z_block_idx_size   = 1

# Debug settings

UPDATE_S              = True
UPDATE_Z              = True
UPDATE_phi            = True
UPDATE_rho            = True
UPDATE_gamma_k        = False
UPDATE_tau            = True
UPDATE_GPD_sigma      = False
UPDATE_GPD_xi         = False
UPDATE_Regularization = False
UPDATE_Imputation     = False

# helper functions ------------------------------------------------------------
def extend_trace(trace, new_n_iters):
    old_n_iters = trace.shape[0] - 1  # since shape[0] = old_n_iters + 1
    assert not np.any(np.isnan(trace)), "Trace array contains NaNs; double-check last save state."
    
    new_shape = (new_n_iters + 1,) + trace.shape[1:]
    extended_trace = np.full(new_shape, np.nan)
    extended_trace[:old_n_iters + 1] = trace

    return extended_trace


# %% LOAD DATASET -----------------------------------------------------------------------------------------------------

# data

if from_simulation == True:
    datafolder = './simulated_seed-2345_t-60_s-625_phi-nonstatsc2_rho-nonstat_tau-10.0/'
    datafile   = 'simulated_data.RData'
if from_simulation == False:
    datafolder = './realdata/'
    datafile   = 'JJA_precip_nonimputed.RData'

# Load from .RData file the following
#   Y,
#   GP_estimates (u, logsigma, xi),
#   elev,
#   stations

r(f'''
    load("{datafolder}/{datafile}")
''')

Y = np.array(r('Y'))
GP_estimates = np.array(r('GP_estimates')).T
logsigma_estimates = GP_estimates[:,1]
xi_estimates       = GP_estimates[:,2]
elevations         = np.array(r('elev'))
stations           = np.array(r('stations')).T

# this `u_vec` is the threshold,
# spatially varying but temporally constant, i.e., each site has its own threshold
# estimated emprically using p = 0.95
u_vec              = GP_estimates[:,0]


# truncate if only running a random subset


# missing indicator matrix

miss_matrix = np.isnan(Y)


# %% SETUP (COVARIATES and CONSTANTS) ---------------------------------------------------------------------------------

# Ns, Nt

Ns = Y.shape[0] # number of sites/stations
Nt = Y.shape[1] # number of time replicates
start_year = -Nt/2
end_year   = Nt/2-1
all_years  = np.linspace(start_year, end_year, Nt)
Time       = (all_years - np.mean(all_years))/np.std(all_years, ddof=1) # delta degress of freedom, to match the n-1 in R
Time       = Time[0:Nt] # if there is any truncation specified above

# Knots number and radius

N_outer_grid_S   = 9
N_outer_grid_phi = 9
N_outer_grid_rho = 9
radius_S         = 3 # radius of Wendland Basis for S
eff_range_phi    = 3 # effective range for phi
eff_range_rho    = 3 # effective range for rho

# threshold probability and quantile

p        = 0.95
u_matrix = np.full(shape = (Ns, Nt), fill_value = np.nanquantile(Y, p)) # threshold u on Y, i.e. p = Pr(Y <= u)
u_vec    = u_matrix[:,rank]

# Sites

sites_xy = stations
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# define the lower and upper limits for x and y
if from_simulation:
    minX, maxX = 0.0, 10.0
    minY, maxY = 0.0, 10.0
else:
    minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
    minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

# Knots - isometric grid of 9 + 4 = 13 knots ----------------------------------

# # isometric knot grid
# N_outer_grid = 16
# h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
# v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
# x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2,
#                                        num = int(2*np.sqrt(N_outer_grid)))
# y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2,
#                                        num = int(2*np.sqrt(N_outer_grid)))
# x_outer_pos              = x_pos[0::2]
# x_inner_pos              = x_pos[1::2]
# y_outer_pos              = y_pos[0::2]
# y_inner_pos              = y_pos[1::2]
# X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
# X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
# knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
# knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
# knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
# knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
# knots_xy                 = knots_xy[knots_id_in_domain]
# knots_x                  = knots_xy[:,0]
# knots_y                  = knots_xy[:,1]
# k                        = len(knots_id_in_domain)

# isometric knot grid - for R (de-coupled from phi and rho)

h_dist_between_knots_S     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_S))-1)
v_dist_between_knots_S     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_S))-1)
x_pos_S                    = np.linspace(minX + h_dist_between_knots_S/2, maxX + h_dist_between_knots_S/2,
                                        num = int(2*np.sqrt(N_outer_grid_S)))
y_pos_S                    = np.linspace(minY + v_dist_between_knots_S/2, maxY + v_dist_between_knots_S/2,
                                        num = int(2*np.sqrt(N_outer_grid_S)))
x_outer_pos_S              = x_pos_S[0::2]
x_inner_pos_S              = x_pos_S[1::2]
y_outer_pos_S              = y_pos_S[0::2]
y_inner_pos_S              = y_pos_S[1::2]
X_outer_pos_S, Y_outer_pos_S = np.meshgrid(x_outer_pos_S, y_outer_pos_S)
X_inner_pos_S, Y_inner_pos_S = np.meshgrid(x_inner_pos_S, y_inner_pos_S)
knots_outer_xy_S           = np.vstack([X_outer_pos_S.ravel(), Y_outer_pos_S.ravel()]).T
knots_inner_xy_S           = np.vstack([X_inner_pos_S.ravel(), Y_inner_pos_S.ravel()]).T
knots_xy_S                 = np.vstack((knots_outer_xy_S, knots_inner_xy_S))
knots_id_in_domain_S       = [row for row in range(len(knots_xy_S)) if (minX < knots_xy_S[row,0] < maxX and minY < knots_xy_S[row,1] < maxY)]
knots_xy_S                 = knots_xy_S[knots_id_in_domain_S]
knots_x_S                  = knots_xy_S[:,0]
knots_y_S                  = knots_xy_S[:,1]
k_S                        = len(knots_id_in_domain_S)

# isometric knot grid - for phi (de-coupled from R and rho)

h_dist_between_knots_phi     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_phi))-1)
v_dist_between_knots_phi     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_phi))-1)
x_pos_phi                    = np.linspace(minX + h_dist_between_knots_phi/2, maxX + h_dist_between_knots_phi/2,
                                        num = int(2*np.sqrt(N_outer_grid_phi)))
y_pos_phi                    = np.linspace(minY + v_dist_between_knots_phi/2, maxY + v_dist_between_knots_phi/2,
                                        num = int(2*np.sqrt(N_outer_grid_phi)))
x_outer_pos_phi              = x_pos_phi[0::2]
x_inner_pos_phi              = x_pos_phi[1::2]
y_outer_pos_phi              = y_pos_phi[0::2]
y_inner_pos_phi              = y_pos_phi[1::2]
X_outer_pos_phi, Y_outer_pos_phi = np.meshgrid(x_outer_pos_phi, y_outer_pos_phi)
X_inner_pos_phi, Y_inner_pos_phi = np.meshgrid(x_inner_pos_phi, y_inner_pos_phi)
knots_outer_xy_phi           = np.vstack([X_outer_pos_phi.ravel(), Y_outer_pos_phi.ravel()]).T
knots_inner_xy_phi           = np.vstack([X_inner_pos_phi.ravel(), Y_inner_pos_phi.ravel()]).T
knots_xy_phi                 = np.vstack((knots_outer_xy_phi, knots_inner_xy_phi))
knots_id_in_domain_phi       = [row for row in range(len(knots_xy_phi)) if (minX < knots_xy_phi[row,0] < maxX and minY < knots_xy_phi[row,1] < maxY)]
knots_xy_phi                 = knots_xy_phi[knots_id_in_domain_phi]
knots_x_phi                  = knots_xy_phi[:,0]
knots_y_phi                  = knots_xy_phi[:,1]
k_phi                        = len(knots_id_in_domain_phi)

# isometric knot grid - for rho (de-coupled from R and phi)

h_dist_between_knots_rho     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid_rho))-1)
v_dist_between_knots_rho     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid_rho))-1)
x_pos_rho                    = np.linspace(minX + h_dist_between_knots_rho/2, maxX + h_dist_between_knots_rho/2,
                                        num = int(2*np.sqrt(N_outer_grid_rho)))
y_pos_rho                    = np.linspace(minY + v_dist_between_knots_rho/2, maxY + v_dist_between_knots_rho/2,
                                        num = int(2*np.sqrt(N_outer_grid_rho)))
x_outer_pos_rho              = x_pos_rho[0::2]
x_inner_pos_rho              = x_pos_rho[1::2]
y_outer_pos_rho              = y_pos_rho[0::2]
y_inner_pos_rho              = y_pos_rho[1::2]
X_outer_pos_rho, Y_outer_pos_rho = np.meshgrid(x_outer_pos_rho, y_outer_pos_rho)
X_inner_pos_rho, Y_inner_pos_rho = np.meshgrid(x_inner_pos_rho, y_inner_pos_rho)
knots_outer_xy_rho           = np.vstack([X_outer_pos_rho.ravel(), Y_outer_pos_rho.ravel()]).T
knots_inner_xy_rho           = np.vstack([X_inner_pos_rho.ravel(), Y_inner_pos_rho.ravel()]).T
knots_xy_rho                 = np.vstack((knots_outer_xy_rho, knots_inner_xy_rho))
knots_id_in_domain_rho       = [row for row in range(len(knots_xy_rho)) if (minX < knots_xy_rho[row,0] < maxX and minY < knots_xy_rho[row,1] < maxY)]
knots_xy_rho                 = knots_xy_rho[knots_id_in_domain_rho]
knots_x_rho                  = knots_xy_rho[:,0]
knots_y_rho                  = knots_xy_rho[:,1]
k_rho                        = len(knots_id_in_domain_rho)

# Copula Splines --------------------------------------------------------------

# Basis Parameters - for the Gaussian and Wendland Basis

radius_S_from_knots = np.repeat(radius_S, k_S) # influence radius from a knot
bandwidth_phi       = eff_range_phi**2/6
bandwidth_rho       = eff_range_rho**2/6

# Generate the weight matrices

# Weight matrix generated using wendland basis for S
wendland_weight_matrix_S = np.full(shape = (Ns,k_S), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                                XB = knots_xy_S)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_S_from_knots)
    wendland_weight_matrix_S[site_id, :] = weight_from_knots

# Weight matrix generated using Gaussian Smoothing Kernel for phi
gaussian_weight_matrix_phi = np.full(shape = (Ns, k_phi), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                                XB = knots_xy_phi)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius_S, bandwidth_phi, cutoff = False) # radius not used when cutoff = False
    gaussian_weight_matrix_phi[site_id, :] = weight_from_knots

# Weight matrix generated using Gaussian Smoothing Kernel for rho
gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)),
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius_S, bandwidth_rho, cutoff = False) # radius not used when cutoff = False
    gaussian_weight_matrix_rho[site_id, :] = weight_from_knots


# Marginal Model - GP(sigma, xi) threshold u ---------------------------------

# Scale logsigma(s)
Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0
C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Shape xi(s)
Beta_xi_m   = 2 # just intercept and elevation
C_xi        = np.full(shape = (Beta_xi_m, Ns, Nt), fill_value = np.nan) # xi design matrix
C_xi[0,:,:] = 1.0
C_xi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Setup For the Copula/Data Model - X = e + X_star = e + R^phi * g(Z) ---------

# Covariance K for Gaussian Field g(Z)
nu        = 0.5                # exponential kernel for matern with nu = 1/2
sigsq_vec = np.repeat(1.0, Ns) # sill for Z, hold at 1

# Scale Mixture R^phi
delta = 0.0 # this is the delta in levy, stays 0
alpha = 0.5 # alpha in the Stable, stays 0.5


# %% ESTIMATE PARAMETERS ----------------------------------------------------------------------------------------------

if from_simulation == False:
    # We estimate parameter's initial values to start the chains

    # Marginal Parameters - GP(sigma, xi) ------------------------------------

    # scale
    Beta_logsigma = np.linalg.lstsq(a=C_logsigma[:,:,0].T, b=logsigma_estimates,rcond=None)[0]
    sigma_vec     = np.exp((C_logsigma.T @ Beta_logsigma).T)[:,rank]

    # shape
    Beta_xi = np.linalg.lstsq(a=C_xi[:,:,0].T, b=xi_estimates,rcond=None)[0]
    xi_vec  = ((C_xi.T @ Beta_xi).T)[:,rank]

    # regularization
    sigma_Beta_logsigma = 1
    sigma_Beta_xi      = 1

    # Dependence Model Parameters - X = e + R^phi * g(Z) ----------------------

    # Nugget Variance (var = tau^2, stdev = tau)

    tau = 10.0 # potentially, use empirical semivariogram estimates

    # rho - covariance K

    # select sites that are "local" to each knot
    rho_at_knots = np.array([])
    distance_matrix = np.full(shape=(Ns, k_rho), fill_value=np.nan)
    # distance from knots
    for site_id in np.arange(Ns):
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), XB = knots_xy_rho)
        distance_matrix[site_id,:] = d_from_knots
    # each knot's "own" sites
    sites_within_knots = {}
    for knot_id in np.arange(k_rho):
        knot_name = 'knot_' + str(knot_id)
        sites_within_knots[knot_name] = np.where(distance_matrix[:,knot_id] <= bandwidth_rho)[0]

    # empirical variogram estimates
    for key in sites_within_knots.keys():
        selected_sites           = sites_within_knots[key]
        demeaned_Y               = (Y.T - np.nanmean(Y, axis = 1)).T
        bin_center, gamma_variog = gs.vario_estimate((sites_x[selected_sites], sites_y[selected_sites]),
                                                    np.nanmean(demeaned_Y[selected_sites], axis=1))
        fit_model = gs.Exponential(dim=2)
        fit_model.fit_variogram(bin_center, gamma_variog, nugget=False)
        # ax = fit_model.plot(x_max = 4)
        # ax.scatter(bin_center, gamma_variog)
        rho_at_knots = np.append(rho_at_knots, fit_model.len_scale)
    if rank == 0:
        print('estimated range:',rho_at_knots)

    # check for unreasonably large values, intialize at some smaller ones
    rho_upper_bound = 4
    if len(np.where(rho_at_knots > rho_upper_bound)[0]) > 0:
        if rank == 0: print('estimated rho >', rho_upper_bound, ' at:', np.where(rho_at_knots > rho_upper_bound)[0])
        if rank == 0: print('rho at those knots set to be at', rho_upper_bound)
        rho_at_knots[np.where(rho_at_knots > rho_upper_bound)[0]] = rho_upper_bound
    # check for unreasonably small values, initialize at some larger ones
    rho_lower_bound = 0.01
    if len(np.where(rho_at_knots < rho_lower_bound)[0]) > 0:
        if rank == 0: print('estimated rho <', rho_lower_bound, ' at:', np.where(rho_at_knots < rho_lower_bound)[0])
        if rank == 0: print('rho at those knots set to be at', rho_lower_bound)
        rho_at_knots[np.where(rho_at_knots < rho_lower_bound)[0]] = rho_lower_bound

    # g(Z)

    rho_vec = gaussian_weight_matrix_rho @ rho_at_knots
    K       = ns_cov(range_vec = rho_vec, sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')
    Z       = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)),
                                                  cov  = K,
                                                  size = Nt).T
    W       = g(Z)

    # phi

    phi_at_knots = np.array([0.5] * k_phi)
    phi_vec      = gaussian_weight_matrix_phi @ phi_at_knots

    # S ~ Stable

    gamma_k_vec   = np.repeat(1.0, k_S) # initialize at 1
    gamma_bar_vec = np.sum(np.multiply(wendland_weight_matrix_S, gamma_k_vec)**(alpha),
                        axis = 1)**(1/alpha) # gamma_bar, axis = 1 to sum over K knots

    if size == 1:
        S_at_knots = np.full(shape = (k_S, Nt), fill_value = np.nan)
        for t in np.arange(Nt):
            obs_idx_1t  = np.where(miss_matrix[:,t] == False)[0]

            pY_1t = pCGP(Y[obs_idx_1t, t], p,
                            u_vec[obs_idx_1t], sigma_vec[obs_idx_1t], xi_vec[obs_idx_1t])

            # S_at_knots[:,t] = np.median(qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_bar_vec[obs_idx_1t], tau
            #                                 ) / W[obs_idx_1t, t])**(1/phi_at_knots)
            S_at_knots[:,t] = np.min(qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_bar_vec[obs_idx_1t], tau
                                            ) / W[obs_idx_1t, t])**(1/phi_at_knots)
    if size > 1: # use MPI to parallelize computation
        comm.Barrier()
        obs_idx_1t  = np.where(miss_matrix[:,rank] == False)[0]
        pY_1t = pCGP(Y[obs_idx_1t, rank], p,
                        u_vec[obs_idx_1t], sigma_vec[obs_idx_1t], xi_vec[obs_idx_1t])
        X_1t  = qRW(pY_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_bar_vec[obs_idx_1t], tau)
        # S_1t  = np.median(X_1t / W[obs_idx_1t, rank]) ** (1/phi_at_knots)
        S_1t  = np.min(X_1t / W[obs_idx_1t, rank]) ** (1/phi_at_knots)

        S_gathered = comm.gather(S_1t, root = 0)
        S_at_knots = np.array(S_gathered).T if rank == 0 else None
        S_at_knots = comm.bcast(S_at_knots, root = 0)

    # X_star = R^phi * g(Z)
    X_star = ((wendland_weight_matrix_S @ S_at_knots).T ** phi_vec).T * W

    # X and dX calculated outside

    if size == 1:
        # X
        obs_idx              = np.where(~miss_matrix.reshape(-1, order = 'F'))[0]

        Y_stacked            = Y.reshape(-1, order = 'F')
        u_matrix_stacked     = u_matrix.reshape(-1, order='F')
        sigma_matrix_stacked = np.exp((C_logsigma.T @ Beta_logsigma)).T.reshape(-1, order = 'F')
        xi_matrix_stacked    = (C_xi.T @ Beta_xi).T.reshape(-1, order = 'F')

        pY_stacked_obs       = pCGP(Y_stacked[obs_idx], p, u_matrix_stacked[obs_idx], sigma_matrix_stacked[obs_idx], xi_matrix_stacked[obs_idx])
        X_stacked            = np.full((Ns*Nt,), fill_value = np.nan)
        X_stacked[obs_idx]   = qRW_NN_2p(pY_stacked_obs,
                                         np.tile(phi_vec, Nt)[obs_idx],
                                         np.tile(gamma_bar_vec, Nt)[obs_idx],
                                         np.full((Ns*Nt,), tau)[obs_idx])
        X_matrix             = X_stacked.reshape(Ns, Nt, order = 'F')

        # dX
        dX_stacked          = np.full((Ns*Nt,), fill_value = np.nan)
        dX_stacked[obs_idx] = dRW(X_stacked[obs_idx], np.tile(phi_vec, Nt)[obs_idx], np.tile(gamma_bar_vec, Nt)[obs_idx], np.full((Ns*Nt,), tau)[obs_idx])
        dX_matrix           = dX_stacked.reshape(Ns, Nt, order = 'F')

    if size > 1:
        # X
        obs_idx_1t       = np.where(~miss_matrix[:,rank])[0]
        pY_1t_obs        = pCGP(Y[obs_idx_1t, rank], p, u_vec[obs_idx_1t], sigma_vec[obs_idx_1t], xi_vec[obs_idx_1t])
        X_1t             = np.full((Ns,), fill_value = np.nan)
        X_1t[obs_idx_1t] = qRW_NN_2p(pY_1t_obs, phi_vec[obs_idx_1t], gamma_bar_vec[obs_idx_1t], np.full((Ns,), tau)[obs_idx_1t])
        X_matrix         = comm.gather(X_1t, root = 0)
        X_matrix         = np.array(X_matrix).T if rank == 0 else None
        # X_matrix = comm.bcast(X_matrix, root = 0)

        # dX
        dX_1t             = np.full((Ns,), fill_value = np.nan)
        dX_1t[obs_idx_1t] = dRW(X_1t[obs_idx_1t], phi_vec[obs_idx_1t], gamma_bar_vec[obs_idx_1t], np.full((Ns,), tau)[obs_idx_1t])
        dX_matrix         = comm.gather(dX_1t, root = 0)
        dX_matrix         = np.array(dX_matrix).T if rank == 0 else None
        # dX_matrix = comm.bcast(dX_matrix, root = 0)

# %% LOAD/HARDCODE PARAMETERS -----------------------------------------------------------------------------------------

# True values as intials with the simulation
if from_simulation == True:

    simulation_threshold = 60.0
    Beta_logsigma        = np.array([3.0, 0.1])
    Beta_xi              = np.array([0.1, 0.05])
    rho_at_knots         = np.sqrt(0.3*knots_x_rho + 0.4*knots_y_rho)/2
    phi_at_knots         = 0.65 - np.sqrt((knots_x_phi-5.1)**2/5 + (knots_y_phi-5.3)**2/4)/11.6
    gamma_k_vec          = np.repeat(0.5, k_S)
    tau                  = 10

    np.random.seed(data_seed)

    # Marginal Model

    u_matrix = np.full(shape = (Ns, Nt), fill_value = simulation_threshold)
    u_vec    = u_matrix[:,rank]

    sigma_Beta_logsigma = 1
    sigma_Beta_xi       = 1

    # g(Z) Transformed Gaussian Process

    rho_vec = gaussian_weight_matrix_rho @ rho_at_knots
    K       = ns_cov(range_vec = rho_vec, sigsq_vec = sigsq_vec,
                     coords = sites_xy, kappa = nu, cov_model = "matern")
    Z       = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
    W       = g(Z)

    # phi Dependence parameter

    phi_vec = gaussian_weight_matrix_phi @ phi_at_knots

    # R^phi Random Scaling

    gamma_bar_vec = np.sum(np.multiply(wendland_weight_matrix_S, gamma_k_vec)**(alpha),
                        axis = 1)**(1/alpha) # gamma_bar, axis = 1 to sum over K knots

    S_at_knots     = np.full(shape = (k_S, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        S_at_knots[:,t] = rlevy(n = k_S, m = delta, s = gamma_k_vec) # generate R at time t, spatially varying k knots
    R_at_sites = wendland_weight_matrix_S @ S_at_knots
    R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
    for t in np.arange(Nt):
        R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

    # Nuggets

    nuggets = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)),
                                                cov  = tau**2,
                                                size = Nt).T
    X_star       = R_phi * W
    X_truth      = X_star + nuggets

    # X and dX calculated outside

    # X
    X_matrix = X_truth

    # dX
    if size == 1:
        obs_idx             = np.where(~miss_matrix.reshape(-1, order='F'))[0]
        dX_stacked          = np.full((Ns*Nt,), fill_value = np.nan)
        dX_stacked[obs_idx] = dRW(X_matrix.reshape(-1, order='F')[obs_idx],
                                  np.tile(phi_vec, Nt)[obs_idx],
                                  np.tile(gamma_bar_vec, Nt)[obs_idx],
                                  np.full((len(obs_idx),), tau))
        dX_matrix           = dX_stacked.reshape((Ns,Nt), order='F')

    if size > 1:
        obs_idx_1t        = np.where(~miss_matrix[:,rank])[0]
        dX_1t             = np.full((Ns,), fill_value = np.nan)
        dX_1t[obs_idx_1t] = dRW(X_matrix[:,rank][obs_idx_1t],
                                phi_vec[obs_idx_1t],
                                gamma_bar_vec[obs_idx_1t],
                                tau)
        dX_matrix         = comm.gather(dX_1t, root = 0)
        dX_matrix         = np.array(dX_matrix).T if rank == 0 else None

    phi_at_knots = np.array([0.5] * k_phi)

# %% PLOT PARAMETER SURFACES --------------------------------------------------------------------------------------
# Plot Parameter Surface
if rank == 0 and start_iter == 1:

    # Grids for plots

    plotgrid_res_x = 75
    plotgrid_res_y = 100
    plotgrid_res_xy = plotgrid_res_x * plotgrid_res_y
    plotgrid_x = np.linspace(minX,maxX,plotgrid_res_x)
    plotgrid_y = np.linspace(minY,maxY,plotgrid_res_y)
    plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
    plotgrid_xy = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

    wendland_weight_matrix_S_for_plot = np.full(shape = (plotgrid_res_xy,k_S), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                                    XB = knots_xy_S)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_S_from_knots)
        wendland_weight_matrix_S_for_plot[site_id, :] = weight_from_knots

    gaussian_weight_matrix_phi_for_plot = np.full(shape = (plotgrid_res_xy, k_phi), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                                    XB = knots_xy_phi)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius_S, bandwidth_phi, cutoff = False)
        gaussian_weight_matrix_phi_for_plot[site_id, :] = weight_from_knots

    gaussian_weight_matrix_rho_for_plot = np.full(shape = (plotgrid_res_xy, k_rho), fill_value = np.nan)
    for site_id in np.arange(plotgrid_res_xy):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)),
                                                    XB = knots_xy_rho)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius_S, bandwidth_rho, cutoff = False)
        gaussian_weight_matrix_rho_for_plot[site_id, :] = weight_from_knots


    # 0. weight from knot plots --------------------------------------------------------------------------------------

    # Define the colors for the colormap (white to red)
    # Create a LinearSegmentedColormap
    colors = ["#ffffff", "#ff0000"]
    n_bins = 50  # Number of discrete color bins
    cmap_name = "white_to_red"
    colormap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    min_w = 0
    max_w = 1
    n_ticks = 11  # (total) number of ticks
    ticks = np.linspace(min_w, max_w, n_ticks).round(3)

    idx = 5
    wendland_weights_for_plot = wendland_weight_matrix_S_for_plot[:,idx]
    gaussian_weights_for_plot = gaussian_weight_matrix_phi_for_plot[:,idx]

    # Plotting Wendland weight for S
    fig, axes = plt.subplots(1,2)
    state_map.boundary.plot(ax=axes[0], color = 'black', linewidth = 0.5)
    heatmap = axes[0].imshow(wendland_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                        cmap = colormap, vmin = min_w, vmax = max_w,
                        interpolation='nearest',
                        extent = [minX, maxX, maxY, minY])
    axes[0].invert_yaxis()
    # axes[0].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
    axes[0].scatter(knots_x_S, knots_y_S, s = 30, color = 'white', marker = '+')
    axes[0].set_xlim(minX, maxX)
    axes[0].set_ylim(minY, maxY)
    axes[0].set_aspect('equal', 'box')
    axes[0].title.set_text('wendland weights radius ' + str(radius_S))

    # Plotting Gaussian weight for phi
    state_map.boundary.plot(ax=axes[1], color = 'black', linewidth = 0.5)
    heatmap = axes[1].imshow(gaussian_weights_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                        cmap = colormap, vmin = min_w, vmax = max_w,
                        interpolation='nearest',
                        extent = [minX, maxX, maxY, minY])
    axes[1].invert_yaxis()
    # axes[1].scatter(sites_x, sites_y, s = 5, color = 'grey', marker = 'o', alpha = 0.8)
    axes[1].scatter(knots_x_phi, knots_y_phi, s = 30, color = 'white', marker = '+')
    axes[1].set_xlim(minX, maxX)
    axes[1].set_ylim(minY, maxY)
    axes[1].set_aspect('equal', 'box')
    axes[1].title.set_text('gaussian weights eff_range ' + str(eff_range_phi))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
    fig.colorbar(heatmap, cax = cbar_ax, ticks = ticks)
    plt.savefig('Plot:knot_weights.pdf')
    # plt.show()
    plt.close()

    # 1. Station and Knot-Radius Setup

    if from_simulation:
        fig, ax = plt.subplots()
        fig.set_size_inches(10,8)
        ax.set_aspect('equal', 'box')
        for i in range(k_S):
            circle_i = plt.Circle((knots_xy_S[i,0], knots_xy_S[i,1]), radius_S_from_knots[i],
                                color='r', fill=True, fc='lightgrey', ec='grey', alpha = 0.2)
            ax.add_patch(circle_i)
        ax.scatter(sites_x, sites_y, marker = '.', c = 'blue', label='sites')
        ax.scatter(knots_x_S, knots_y_S, marker = '+', c = 'red', label = 'knot', s = 300)
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill = False, color = 'black')
        ax.add_patch(space_rectangle)
        ax.set_xticks(np.linspace(minX, maxX,num=3))
        ax.set_yticks(np.linspace(minY, maxY,num=5))
        box = ax.get_position()
        legend_elements = [mpl.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                        mpl.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=20),
                        mpl.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'lightgrey', markeredgecolor='grey', markersize = 20, alpha = 0.2),
                        mpl.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=20, label='Spatial Domain')]
        plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel('longitude', fontsize = 20)
        plt.ylabel('latitude', fontsize = 20)
        plt.subplots_adjust(right=0.6)
        plt.savefig('Plot:Knot_Radius_Setup.pdf',bbox_inches="tight")
        # plt.show()
        plt.close()

    if not from_simulation:

        # Plot of US overview and a Zoomed in Station-Knot setup

        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2, 1]})

        # US Overview
        ax = axes[0]
        state_map.boundary.plot(ax=ax, color='lightgrey', zorder = 1)
        ax.scatter(sites_x, sites_y, marker='.', c='blue',
                edgecolor = 'white', label='training', zorder = 2)
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill=False, color='black')
        ax.add_patch(space_rectangle)
        ax.set_xticks(np.linspace(-130, -70, num=7))
        ax.set_yticks(np.linspace(25, 50, num=6))
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('Longitude', fontsize = 20)
        ax.set_ylabel('Latitude', fontsize = 20)
        ax.set_xlim([-130, -65])
        ax.set_ylim([25,50])
        ax.set_aspect('auto')

        # Knot-Radius Setup
        ax = axes[1]
        # Plot knots and circles
        for i in range(k_S):
            circle_i = plt.Circle((knots_xy_S[i, 0], knots_xy_S[i, 1]), radius_S_from_knots[i],
                                color='r', fill=True, fc='lightgrey', ec='grey', alpha=0.2)
            ax.add_patch(circle_i)
        # Scatter plot for sites and knots
        ax.scatter(sites_x, sites_y, marker='.', c='blue', label='sites')
        ax.scatter(knots_x_S, knots_y_S, marker='+', c='red', label='knot', s=300)
        # Plot spatial domain rectangle
        space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                        fill=False, color='black')
        ax.add_patch(space_rectangle)
        # Set ticks and labels
        ax.set_xticks(np.linspace(minX, maxX, num=3))
        ax.set_yticks(np.linspace(minY, maxY, num=5))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Longitude', fontsize=20)
        plt.ylabel('Latitude', fontsize=20)
        plt.xlim([-106, -88])
        plt.ylim([28,48])
        # Plot boundary
        state_map.boundary.plot(ax=ax, color='black')
        ax.set_aspect('equal', 'box')  # Ensures 1:1 ratio for data units
        # Adjust the position of the legend to avoid overlap with the plot
        box = ax.get_position()
        legend_elements = [mpl.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                        mpl.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=20),
                        mpl.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'lightgrey', markeredgecolor = 'grey', markersize = 20, alpha = 0.2),
                        mpl.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=20, label='Spatial Domain')]
        plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 20)

        plt.savefig('Plot:USMap_Knot_Radius_Setup.pdf',bbox_inches='tight')
        # plt.show()
        plt.close()

    # 2. Elevation

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                                cmap = 'OrRd')
    ax.set_aspect('equal', 'box')
    plt.colorbar(elev_scatter)
    plt.savefig('Plot:station_elevation.pdf')
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('y', fontsize = 20)
    plt.title('Station elevations', fontsize = 20)
    plt.savefig('Plot:station_elevation.pdf', bbox_inches="tight")
    # plt.show()
    plt.close()

    # 3. phi - initial surface
    # plt set figure size
    plt.figure(figsize=(5,4))
    phi_vec_for_plot = (gaussian_weight_matrix_phi_for_plot @ phi_at_knots).round(3)
    graph, ax = plt.subplots()
    heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                        cmap ='RdBu_r',
                        interpolation='nearest',
                        extent = [minX, maxX, maxY, minY],
                        vmin = 0.35, vmax = 0.65)
    ax.invert_yaxis()
    graph.colorbar(heatmap)
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('y', fontsize = 20)
    plt.title(r'initial $\phi(s)$ surface', fontsize = 20)
    plt.savefig('Plot:initial_phi_surface.pdf')
    # plt.show()
    plt.close()


    # 4. Plot rho surface

    plt.figure(figsize=(5,4))
    rho_vec_for_plot = gaussian_weight_matrix_rho_for_plot @ rho_at_knots
    graph, ax = plt.subplots()
    heatmap = ax.imshow(rho_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                        cmap ='OrRd', interpolation='nearest',
                        extent = [minX, maxX, minY, maxY], origin='lower')
    graph.colorbar(heatmap)
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('y', fontsize = 20)
    plt.title(r'initial $\rho(s)$ surface', fontsize = 20)
    plt.savefig('Plot:initial_rho_surface.pdf')
    # plt.show()
    plt.close()


    # 5. GP Surfaces (initial values)

    def my_ceil(a, precision=0):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    # Scale #

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(10, 4.5)

    # initial spline smoothed values for logsigma
    logsigma_init_spline = ((C_logsigma.T @ Beta_logsigma).T)[:,rank]
    vmin                 = min(my_floor(min(logsigma_estimates), 1), my_floor(min(logsigma_init_spline), 1))
    vmax                 = max(my_ceil(max(logsigma_estimates), 1), my_ceil(max(logsigma_init_spline), 1))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    divnorm              = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)

    # initial site values

    logsigma_MLE_scatter = ax[0].scatter(sites_x, sites_y, s = 10,
                                         cmap = 'OrRd', c = logsigma_estimates, norm = divnorm)
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel('x', fontsize = 20)
    ax[0].set_ylabel('y', fontsize = 20)
    ax[0].set_title(r'MLE $\log(\sigma)$', fontsize = 20)

    # initial spline fitted values

    logsigma_spline_scatter = ax[1].scatter(sites_x, sites_y, s = 10,
                                            cmap = 'OrRd', c = logsigma_init_spline, norm = divnorm)
    ax[1].set_aspect('equal','box')
    ax[1].set_xlabel('x', fontsize = 20)
    ax[1].set_ylabel('y', fontsize = 20)
    ax[1].set_title(r'spline smoothed $\log(\sigma)$', fontsize = 20)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(logsigma_spline_scatter, cax = cbar_ax)
    plt.savefig('Plot:initial_logsigma_estimates.pdf')
    # plt.show()
    plt.close()

    # Shape #

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(10, 4.5)

    # initial spline smoothed values for xi
    xi_init_spline = ((C_xi.T @ Beta_xi).T)[:,rank]
    vmin            = min(my_floor(min(xi_estimates), 1), my_floor(min(xi_init_spline), 1))
    vmax            = max(my_ceil(max(xi_estimates), 1), my_ceil(max(xi_init_spline), 1))
    if vmin == vmax:
        vmin -= vmin*0.1
        vmax += vmax*0.1
    divnorm         = mpl.colors.TwoSlopeNorm(vcenter = (vmin + vmax)/2, vmin = vmin, vmax = vmax)

    # initial site values

    xi_MLE_scatter = ax[0].scatter(sites_x, sites_y, s = 10,
                                    cmap = 'OrRd', c = xi_estimates,
                                    norm = divnorm)
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel('x', fontsize = 20)
    ax[0].set_ylabel('y', fontsize = 20)
    ax[0].set_title(r'MLE $\xi$ estimates', fontsize = 20)

    # initial spline fitted values

    xi_spline_scatter = ax[1].scatter(sites_x, sites_y, s = 10,
                                       cmap = 'OrRd', c = xi_init_spline,
                                       norm = divnorm)
    ax[1].set_aspect('equal','box')
    ax[1].set_xlabel('x', fontsize = 20)
    ax[1].set_ylabel('y', fontsize = 20)
    ax[1].set_title(r'spline smoothed $\xi$ estimates', fontsize = 20)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(xi_spline_scatter, cax = cbar_ax)

    plt.savefig('Plot:initial_xi_estimates.pdf')
    # plt.show()
    plt.close()

# %% ADAPTIVE UPDATE & BLOCK UPDATE SETUP -----------------------------------------------------------------------------

# Create Coefficient Index Blocks - each block size does not exceed size specified above ------------------------------

## phi
phi_block_idx_dict = {}
lst = list(range(k_phi))
for i in range(0, k_phi, phi_block_idx_size):
    start_index = i
    end_index   = i + phi_block_idx_size
    key         = 'phi_block_idx_'+str(i//phi_block_idx_size+1)
    phi_block_idx_dict[key] = lst[start_index:end_index]

## rho
rho_block_idx_dict = {}
lst = list(range(k_rho))
for i in range(0, k_rho, rho_block_idx_size):
    start_index = i
    end_index   = i + rho_block_idx_size
    key         = 'rho_block_idx_'+str(i//rho_block_idx_size+1)
    rho_block_idx_dict[key] = lst[start_index:end_index]

## Z_t
Z_block_idx_dict = {}
lst = list(range(Ns))
for i in range(0, Ns, Z_block_idx_size):
    start_index = i
    end_index   = min(i + Z_block_idx_size, Ns)
    key         = f"Z_block_idx_{i//Z_block_idx_size+1}"
    Z_block_idx_dict[key] = lst[start_index:end_index]

# Z_block_idx_dict = {}
# lst = list(range(Ns))

# # Assuming you have these two sets/lists:
# exceed_indices = set([...])   # your exceed index list
# censored_indices = set([...]) # your censored index list

# # Split into two separate lists to keep them apart
# exceed_list = [i for i in lst if i in exceed_indices]
# censored_list = [i for i in lst if i in censored_indices]

# # Helper to chunk a list into blocks
# def create_blocks(input_list, block_size, prefix, start_key=1):
#     blocks = {}
#     block_num = start_key
#     for i in range(0, len(input_list), block_size):
#         blocks[f"{prefix}_{block_num}"] = input_list[i:i + block_size]
#         block_num += 1
#     return blocks, block_num

# # Block the exceed and censored lists separately
# Z_block_idx_dict_exceed, next_key = create_blocks(exceed_list, Z_block_idx_size, "Z_block_idx")
# Z_block_idx_dict_censored, _ = create_blocks(censored_list, Z_block_idx_size, "Z_block_idx", start_key=next_key)

# # Merge the dictionaries
# Z_block_idx_dict = {**Z_block_idx_dict_exceed, **Z_block_idx_dict_censored}


# Adaptive Update: Counter --------------------------------------------------------------------------------------------

## St
if norm_pareto == 'standard':
    num_accepted_St_list = [[0] * k_S] * size if rank == 0 else None
    num_accepted_St      = comm.scatter(num_accepted_St_list, root= 0) if size>1 else num_accepted_St_list[0]

## Zt
# num_accepted_Zt_list = [[0] * Ns] * size if rank == 0 else None
# num_accepted_Zt      = comm.scatter(num_accepted_Zt_list, root = 0) if size>1 else num_accepted_Zt_list[0]
num_accepted_Zt = {}
for block_key in Z_block_idx_dict.keys():
    num_accepted_Zt[block_key] = 0

## Other variables: phi, rho, gamma, tau, marginal Y, regularizaiton

if rank == 0:
    num_accepted = {}
    # phi
    for key in phi_block_idx_dict.keys(): num_accepted[key] = 0

    # rho
    for key in rho_block_idx_dict.keys(): num_accepted[key] = 0

    # gamma
    num_accepted['gamma_k_vec'] = [0] * k_S

    # tau
    num_accepted['tau'] = 0

    # marginal Y
    num_accepted['Beta_logsigma'] = 0
    num_accepted['Beta_xi']       = 0

    # regularization
    num_accepted['sigma_Beta_logsigma'] = 0
    num_accepted['sigma_Beta_xi']       = 0

# Adaptive Update: Proposal Variance Scalar and Covariance Matrix -------------------------------------------------

if start_iter == 1: # initialize the proposal scalar variance and covariance

    import proposal_cov as pc

    # sigma_m: proposal scalar variance for St, Zt, phi, rho, tau, marginal Y, and regularization terms ---------

    S_log_cov               = pc.S_log_cov               if pc.S_log_cov               is not None else np.tile(0.05*np.eye(k_S)[:,:,None], reps = (1,1,Nt))
    # Z_cov                   = pc.Z_cov                   if pc.Z_cov                   is not None else np.tile(0.01*np.eye(Ns)[:,:,None],reps = (1,1,Nt))
    gamma_k_cov             = pc.gamma_k_cov             if pc.gamma_k_cov             is not None else 0.1*np.eye(k_S)
    tau_var                 = pc.tau_var                 if pc.tau_var                 is not None else 1
    sigma_Beta_logsigma_var = pc.sigma_Beta_logsigma_var if pc.sigma_Beta_logsigma_var is not None else 1
    sigma_Beta_xi_var       = pc.sigma_Beta_xi_var       if pc.sigma_Beta_xi_var       is not None else 1

    # St
    sigma_m_sq_St_list = [(np.diag(S_log_cov[:,:,t])) for t in range(Nt)] if rank == 0 else None
    sigma_m_sq_St      = comm.scatter(sigma_m_sq_St_list, root = 0)       if size>1 else sigma_m_sq_St_list[0]

    # Zt
    # sigma_m_sq_Zt_list = [(np.diag(Z_cov[:,:,t])) for t in range(Nt)] if rank == 0 else None
    # sigma_m_sq_Zt      = comm.scatter(sigma_m_sq_Zt_list, root = 0)   if size>1 else sigma_m_sq_Zt_list[0]
    sigma_m_sq_Zt = {}
    for block_key in Z_block_idx_dict.keys():
        block_size = len(Z_block_idx_dict[block_key])
        sigma_m_sq_Zt[block_key] = (2.4**2)/block_size

    if rank == 0:
        sigma_m_sq = {}

        # phi
        for key in phi_block_idx_dict.keys(): sigma_m_sq[key] = (2.4**2)/len(phi_block_idx_dict[key])

        # rho
        for key in rho_block_idx_dict.keys(): sigma_m_sq[key] = (2.4**2)/len(rho_block_idx_dict[key])

        # gamma
        sigma_m_sq['gamma_k_vec'] = list(np.diag(gamma_k_cov))

        # tau
        sigma_m_sq['tau'] = tau_var

        # marginal Y
        sigma_m_sq['Beta_logsigma'] = (2.4**2)/Beta_logsigma_m
        sigma_m_sq['Beta_xi']       = (2.4**2)/Beta_xi_m

        # regularization
        sigma_m_sq['sigma_Beta_logsigma'] = sigma_Beta_logsigma_var
        sigma_m_sq['sigma_Beta_xi']       = sigma_Beta_xi_var

    # Sigma0: proposal covariance matrix for phi, rho, and marginal Y -------------------------------------------

    Z_cov                   = pc.Z_cov             if pc.Z_cov             is not None else np.tile(1e-2 * np.eye(Ns)[:,:,None],reps = (1,1,Nt))
    Sigma_0_Zt = {}
    for block_key in Z_block_idx_dict.keys():
        Sigma_0_Zt[block_key] = Z_cov[:,:,rank][Z_block_idx_dict[block_key],:][:,Z_block_idx_dict[block_key]]

    phi_cov                 = pc.phi_cov           if pc.phi_cov           is not None else 1e-5 * np.identity(k_phi)
    rho_cov                 = pc.rho_cov           if pc.rho_cov           is not None else 1e-1 * np.identity(k_rho)
    Beta_logsigma_cov       = pc.Beta_logsigma_cov if pc.Beta_logsigma_cov is not None else 1e-8 * np.identity(Beta_logsigma_m)
    Beta_xi_cov             = pc.Beta_xi_cov       if pc.Beta_xi_cov       is not None else 1e-8 * np.identity(Beta_xi_m)

    if rank == 0:
        Sigma_0 = {}

        # phi
        phi_block_cov_dict = {}
        for key in phi_block_idx_dict.keys():
            start_idx               = phi_block_idx_dict[key][0]
            end_idx                 = phi_block_idx_dict[key][-1]+1
            phi_block_cov_dict[key] = phi_cov[start_idx:end_idx, start_idx:end_idx]
        Sigma_0.update(phi_block_cov_dict)

        # rho
        rho_block_cov_dict = {}
        for key in rho_block_idx_dict.keys():
            start_idx               = rho_block_idx_dict[key][0]
            end_idx                 = rho_block_idx_dict[key][-1]+1
            rho_block_cov_dict[key] = rho_cov[start_idx:end_idx, start_idx:end_idx]
        Sigma_0.update(rho_block_cov_dict)

        # marginal Y
        Sigma_0['Beta_logsigma'] = Beta_logsigma_cov
        Sigma_0['Beta_xi']       = Beta_xi_cov

    # Checking dimensions -----------------------------------------------------------------------------------------

    assert k_phi           == phi_cov.shape[0]
    assert k_rho           == rho_cov.shape[0]
    assert k_S             == S_log_cov.shape[0]
    assert Nt              == S_log_cov.shape[2]
    assert Ns              == Z_cov.shape[0]
    assert Beta_logsigma_m == Beta_logsigma_cov.shape[0]
    assert Beta_xi_m       == Beta_xi_cov.shape[0]

else: # start_iter != 1
    # pickle load the Proposal Variance Scalar, Covariance Matrix

    # sigma_m: proposal scalar variance for St, Zt, phi, rho, tau, marginal Y, and regularization terms ---------

    ## St
    if UPDATE_S:
        if rank == 0:
            with open('sigma_m_sq_St_list.pkl', 'rb') as file: sigma_m_sq_St_list = pickle.load(file)
        else:
            sigma_m_sq_St_list = None
        if size != 1: sigma_m_sq_St = comm.scatter(sigma_m_sq_St_list, root = 0)

    ## Zt
    if UPDATE_Z:
        if rank == 0:
            with open('sigma_m_sq_Zt_list.pkl', 'rb') as file: sigma_m_sq_Zt_list = pickle.load(file)
            with open('Sigma_0_Zt_list.pkl', 'rb')    as file: Sigma_0_Zt_list    = pickle.load(file)
        else:
            sigma_m_sq_Zt_list = None
            Sigma_0_Zt_list    = None
        sigma_m_sq_Zt = comm.scatter(sigma_m_sq_Zt_list, root = 0) if size>1 else sigma_m_sq_Zt_list[0]
        Sigma_0_Zt    = comm.scatter(Sigma_0_Zt_list, root = 0)    if size>1 else Sigma_0_Zt_list[0]

    ## phi, rho, gamma, tau, marginal Y, regularizations
    if UPDATE_phi or UPDATE_rho or UPDATE_gamma_k or UPDATE_tau or UPDATE_GPD_sigma or UPDATE_GPD_xi or UPDATE_Regularization:
        if rank == 0:
            with open('sigma_m_sq.pkl','rb') as file: sigma_m_sq = pickle.load(file)

    # Sigma0: proposal covariance matrix for Z, phi, rho, and marginal Y
    if UPDATE_Z or UPDATE_phi or UPDATE_rho or UPDATE_GPD_sigma or UPDATE_GPD_xi:
        if rank == 0:
            with open('Sigma_0.pkl', 'rb') as file: Sigma_0 = pickle.load(file)

# Adaptive Update track history ---------------------------------------------------------------------------------------

if start_iter == 1:
    N_ADAPTS = n_iters // ADAPT_SIZE if rank == 0 else None
    if UPDATE_S:              r_hat_S_history             = np.full((N_ADAPTS, Nt, k_S),                   fill_value = np.nan) if rank == 0 else None
    if UPDATE_Z:              r_hat_Z_history             = np.full((N_ADAPTS, Nt, len(Z_block_idx_dict)), fill_value = np.nan) if rank == 0 else None
    if UPDATE_phi:            r_hat_phi_history           = np.full((N_ADAPTS, len(phi_block_idx_dict)),   fill_value = np.nan) if rank == 0 else None
    if UPDATE_rho:            r_hat_rho_history           = np.full((N_ADAPTS, len(rho_block_idx_dict)),   fill_value = np.nan) if rank == 0 else None
    if UPDATE_gamma_k:        r_hat_gamma_k_history       = np.full((N_ADAPTS, k_S),                       fill_value = np.nan) if rank == 0 else None
    if UPDATE_tau:            r_hat_tau_history           = np.full((N_ADAPTS, 1),                         fill_value = np.nan) if rank == 0 else None
    if UPDATE_GPD_sigma:      r_hat_Beta_logsigma_history = np.full((N_ADAPTS, 1),                         fill_value = np.nan) if rank == 0 else None
    if UPDATE_GPD_xi:         r_hat_Beta_xi_history       = np.full((N_ADAPTS, 1),                         fill_value = np.nan) if rank == 0 else None
    if UPDATE_Regularization: r_hat_sigma_Beta_history    = np.full((N_ADAPTS, 2),                         fill_value = np.nan) if rank == 0 else None
else: # start_iter != 1
    if UPDATE_S:              r_hat_S_history             = np.load('r_hat_S_history.npy')             if rank == 0 else None
    if UPDATE_Z:              r_hat_Z_history             = np.load('r_hat_Z_history.npy')             if rank == 0 else None
    if UPDATE_phi:            r_hat_phi_history           = np.load('r_hat_phi_history.npy')           if rank == 0 else None
    if UPDATE_rho:            r_hat_rho_history           = np.load('r_hat_rho_history.npy')           if rank == 0 else None
    if UPDATE_gamma_k:        r_hat_gamma_k_history       = np.load('r_hat_gamma_k_history.npy')       if rank == 0 else None
    if UPDATE_tau:            r_hat_tau_history           = np.load('r_hat_tau_history.npy')           if rank == 0 else None
    if UPDATE_GPD_sigma:      r_hat_Beta_logsigma_history = np.load('r_hat_Beta_logsigma_history.npy') if rank == 0 else None
    if UPDATE_GPD_xi:         r_hat_Beta_xi_history       = np.load('r_hat_Beta_xi_history.npy')       if rank == 0 else None
    if UPDATE_Regularization: r_hat_sigma_Beta_history    = np.load('r_hat_sigma_Beta_history.npy')    if rank == 0 else None
    
    if rank == 0: # check if we need to extend the trace of r_hat
        N_ADAPTS = n_iters // ADAPT_SIZE
        if N_ADAPTS > r_hat_S_history.shape[0]:
            if UPDATE_S:              r_hat_S_history             = np.append(r_hat_S_history,             np.full((N_ADAPTS - r_hat_S_history.shape[0], Nt, k_S),                     fill_value = np.nan), axis = 0)
            if UPDATE_Z:              r_hat_Z_history             = np.append(r_hat_Z_history,             np.full((N_ADAPTS - r_hat_Z_history.shape[0], Nt, len(Z_block_idx_dict)),   fill_value = np.nan), axis = 0)
            if UPDATE_phi:            r_hat_phi_history           = np.append(r_hat_phi_history,           np.full((N_ADAPTS - r_hat_phi_history.shape[0], len(phi_block_idx_dict)),   fill_value = np.nan), axis = 0)
            if UPDATE_rho:            r_hat_rho_history           = np.append(r_hat_rho_history,           np.full((N_ADAPTS - r_hat_rho_history.shape[0], len(rho_block_idx_dict)),   fill_value = np.nan), axis = 0)
            if UPDATE_gamma_k:        r_hat_gamma_k_history       = np.append(r_hat_gamma_k_history,       np.full((N_ADAPTS - r_hat_gamma_k_history.shape[0], k_S),                   fill_value = np.nan), axis = 0)
            if UPDATE_tau:            r_hat_tau_history           = np.append(r_hat_tau_history,           np.full((N_ADAPTS - r_hat_tau_history.shape[0], 1),                         fill_value = np.nan), axis = 0)
            if UPDATE_GPD_sigma:      r_hat_Beta_logsigma_history = np.append(r_hat_Beta_logsigma_history, np.full((N_ADAPTS - r_hat_Beta_logsigma_history.shape[0], 1),               fill_value = np.nan), axis = 0)
            if UPDATE_GPD_xi:         r_hat_Beta_xi_history       = np.append(r_hat_Beta_xi_history,       np.full((N_ADAPTS - r_hat_Beta_xi_history.shape[0], 1),                     fill_value = np.nan), axis = 0)
            if UPDATE_Regularization: r_hat_sigma_Beta_history    = np.append(r_hat_sigma_Beta_history,    np.full((N_ADAPTS - r_hat_sigma_Beta_history.shape[0], 2),                  fill_value = np.nan), axis = 0)


# %% STORAGE ----------------------------------------------------------------------------------------------------------

if start_iter == 1:
    loglik_trace              = np.full(shape = (n_iters+1, 1), fill_value = np.nan)                                         if rank == 0 else None # overall likelihood
    loglik_detail_trace       = np.full(shape = (n_iters+1, 4), fill_value = np.nan)                                         if rank == 0 else None # detail likelihood
    X_trace                   = np.full(shape = (n_iters+1, Ns, Nt), fill_value = np.nan)                                    if rank == 0 else None
    dX_trace                  = np.full(shape = (n_iters+1, Ns, Nt), fill_value = np.nan)                                    if rank == 0 else None
    if UPDATE_Imputation:     Y_trace                   = np.full(shape = (n_iters+1, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
    if UPDATE_S:              S_trace_log               = np.full(shape = (n_iters+1, k_S, Nt), fill_value = np.nan)         if rank == 0 else None # log(S)
    if UPDATE_phi:            phi_knots_trace           = np.full(shape = (n_iters+1, k_phi), fill_value = np.nan)           if rank == 0 else None # phi_at_knots
    if UPDATE_rho:            rho_knots_trace           = np.full(shape = (n_iters+1, k_rho), fill_value = np.nan)           if rank == 0 else None # rho_at_knots
    if UPDATE_GPD_sigma:      Beta_logsigma_trace       = np.full(shape = (n_iters+1, Beta_logsigma_m), fill_value = np.nan) if rank == 0 else None # logsigma Covariate Coefficients
    if UPDATE_GPD_xi:         Beta_xi_trace             = np.full(shape = (n_iters+1, Beta_xi_m), fill_value = np.nan)       if rank == 0 else None # xi Covariate Coefficients
    if UPDATE_Regularization: sigma_Beta_logsigma_trace = np.full(shape = (n_iters+1, 1), fill_value = np.nan)               if rank == 0 else None # prior sd for beta_logsigma's
    if UPDATE_Regularization: sigma_Beta_xi_trace       = np.full(shape = (n_iters+1, 1), fill_value = np.nan)               if rank == 0 else None # prior sd for beta_xi's
    if UPDATE_tau:            tau_trace                 = np.full(shape = (n_iters+1, 1), fill_value = np.nan)               if rank == 0 else None
    if UPDATE_Z:              Z_trace                   = np.full(shape = (n_iters+1, Ns, Nt), fill_value = np.nan)          if rank == 0 else None
    if UPDATE_gamma_k:        gamma_k_vec_trace         = np.full(shape = (n_iters+1, k_S), fill_value = np.nan)             if rank == 0 else None
else: # start_iter != 1, load from environment
    # load the trace from previous run
    loglik_trace              = np.load('loglik_trace.npy')              if rank == 0 else None
    loglik_detail_trace       = np.load('loglik_detail_trace.npy')       if rank == 0 else None
    X_trace                   = np.load('X_trace.npy')                   if rank == 0 else None
    dX_trace                  = np.load('dX_trace.npy')                  if rank == 0 else None
    if UPDATE_Imputation:     Y_trace                   = np.load('Y_trace.npy')                   if rank == 0 else None    
    if UPDATE_S:              S_trace_log               = np.load('S_trace_log.npy')               if rank == 0 else None
    if UPDATE_phi:            phi_knots_trace           = np.load('phi_knots_trace.npy')           if rank == 0 else None
    if UPDATE_rho:            rho_knots_trace           = np.load('rho_knots_trace.npy')           if rank == 0 else None
    if UPDATE_GPD_sigma:      Beta_logsigma_trace       = np.load('Beta_logsigma_trace.npy')       if rank == 0 else None
    if UPDATE_GPD_xi:         Beta_xi_trace             = np.load('Beta_xi_trace.npy')             if rank == 0 else None
    if UPDATE_Regularization: sigma_Beta_logsigma_trace = np.load('sigma_Beta_logsigma_trace.npy') if rank == 0 else None
    if UPDATE_Regularization: sigma_Beta_xi_trace       = np.load('sigma_Beta_xi_trace.npy')       if rank == 0 else None
    if UPDATE_tau:            tau_trace                 = np.load('tau_trace.npy')                 if rank == 0 else None
    if UPDATE_Z:              Z_trace                   = np.load('Z_trace.npy')                   if rank == 0 else None
    if UPDATE_gamma_k:        gamma_k_vec_trace         = np.load('gamma_k_vec_trace.npy')         if rank == 0 else None
    # X_star_trace              = np.load('X_star_trace.npy')              if rank == 0 else None
    
    # check if we need to extend the trace
    if rank == 0:
        if n_iters > loglik_trace.shape[0] - 1:
            loglik_trace              = extend_trace(loglik_trace, n_iters)              
            loglik_detail_trace       = extend_trace(loglik_detail_trace, n_iters)       
            X_trace                   = extend_trace(X_trace, n_iters)                   
            dX_trace                  = extend_trace(dX_trace, n_iters)                  
            if UPDATE_Imputation:     Y_trace                   = extend_trace(Y_trace, n_iters)                   
            if UPDATE_S:              S_trace_log               = extend_trace(S_trace_log, n_iters)               
            if UPDATE_phi:            phi_knots_trace           = extend_trace(phi_knots_trace, n_iters)           
            if UPDATE_rho:            rho_knots_trace           = extend_trace(rho_knots_trace, n_iters)           
            if UPDATE_GPD_sigma:      Beta_logsigma_trace       = extend_trace(Beta_logsigma_trace, n_iters)       
            if UPDATE_GPD_xi:         Beta_xi_trace             = extend_trace(Beta_xi_trace, n_iters)             
            if UPDATE_Regularization: sigma_Beta_logsigma_trace = extend_trace(sigma_Beta_logsigma_trace, n_iters) 
            if UPDATE_Regularization: sigma_Beta_xi_trace       = extend_trace(sigma_Beta_xi_trace, n_iters)       
            if UPDATE_tau:            tau_trace                 = extend_trace(tau_trace, n_iters)                 
            if UPDATE_Z:              Z_trace                   = extend_trace(Z_trace, n_iters)                   
            if UPDATE_gamma_k:        gamma_k_vec_trace         = extend_trace(gamma_k_vec_trace, n_iters)         
            # X_star_trace              = extend_trace(X_star_trace, n_iters)              

# Initialize ----------------------------------------------------------------------------------------------------------

# Initialize Parameters for rank 0 worker only, other workers bcast "current value" later -----------------------------

# Initialize at the truth/estimated value from above code
Y_matrix_init            = Y                   if rank == 0 else None
X_init                   = X_matrix            if rank == 0 else None
dX_init                  = dX_matrix           if rank == 0 else None
S_matrix_init_log        = np.log(S_at_knots)  if rank == 0 else None
phi_knots_init           = phi_at_knots        if rank == 0 else None
rho_knots_init           = rho_at_knots        if rank == 0 else None
Beta_logsigma_init       = Beta_logsigma       if rank == 0 else None
Beta_xi_init             = Beta_xi             if rank == 0 else None
sigma_Beta_logsigma_init = sigma_Beta_logsigma if rank == 0 else None
sigma_Beta_xi_init       = sigma_Beta_xi       if rank == 0 else None
tau_init                 = tau                 if rank == 0 else None
Z_init                   = Z                   if rank == 0 else None
gamma_k_vec_init         = gamma_k_vec         if rank == 0 else None
# X_star_init              = X_star              if rank == 0 else None

# Initialize from the last iteration of the saved traceplot
if start_iter != 1:
    last_iter                = start_iter - 1
    X_init                   = X_trace[last_iter,:,:]                 if rank == 0 else None
    dX_init                  = dX_trace[last_iter,:,:]                if rank == 0 else None
    if UPDATE_Imputation:     Y_matrix_init            = Y_trace[last_iter,:,:]                 if rank == 0 else None
    if UPDATE_S:              S_matrix_init_log        = S_trace_log[last_iter,:,:]             if rank == 0 else None
    if UPDATE_phi:            phi_knots_init           = phi_knots_trace[last_iter,:]           if rank == 0 else None
    if UPDATE_rho:            rho_knots_init           = rho_knots_trace[last_iter,:]           if rank == 0 else None
    if UPDATE_GPD_sigma:      Beta_logsigma_init       = Beta_logsigma_trace[last_iter,:]       if rank == 0 else None
    if UPDATE_GPD_xi:         Beta_xi_init             = Beta_xi_trace[last_iter,:]             if rank == 0 else None
    if UPDATE_Regularization: sigma_Beta_logsigma_init = sigma_Beta_logsigma_trace[last_iter,0] if rank == 0 else None # must be value, can't be array([value])
    if UPDATE_Regularization: sigma_Beta_xi_init       = sigma_Beta_xi_trace[last_iter,0]       if rank == 0 else None # must be value, can't be array([value])
    if UPDATE_tau:            tau_init                 = tau_trace[last_iter,:]                 if rank == 0 else None
    if UPDATE_Z:              Z_init                   = Z_trace[last_iter,:,:]                 if rank == 0 else None
    if UPDATE_gamma_k:        gamma_k_vec_init         = gamma_k_vec_trace[last_iter,:]         if rank == 0 else None
    # X_star_init              = X_star_trace[last_iter,:,:]            if rank == 0 else None

# Store initial value into first row of traceplot
if start_iter == 1: 
    if rank == 0: 
        X_trace[0,:,:]                 = X_init
        dX_trace[0,:,:]                = dX_init
        if UPDATE_Imputation:     Y_trace[0,:,:]                 = Y_matrix_init
        if UPDATE_S:              S_trace_log[0,:,:]             = S_matrix_init_log # matrix (k, Nt)
        if UPDATE_phi:            phi_knots_trace[0,:]           = phi_knots_init
        if UPDATE_rho:            rho_knots_trace[0,:]           = rho_knots_init
        if UPDATE_GPD_sigma:      Beta_logsigma_trace[0,:]       = Beta_logsigma_init
        if UPDATE_GPD_xi:         Beta_xi_trace[0,:]             = Beta_xi_init
        if UPDATE_Regularization: sigma_Beta_logsigma_trace[0,:] = sigma_Beta_logsigma_init
        if UPDATE_Regularization: sigma_Beta_xi_trace[0,:]       = sigma_Beta_xi_init
        if UPDATE_tau:            tau_trace[0,:]                 = tau_init
        if UPDATE_Z:              Z_trace[0,:,:]                 = Z_init
        if UPDATE_gamma_k:        gamma_k_vec_trace[0,:]         = gamma_k_vec_init
        # X_star_trace[0,:,:]            = X_star_init

# Set Current Values using broadcast from worker 0 --------------------------------------------------------------------

## Marginal Model -----------------------------------------------------------------------------------------------------

## GPD covariate coefficients --> GPD surface ---------------------------------
Beta_logsigma_current = comm.bcast(Beta_logsigma_init, root = 0)
Beta_xi_current       = comm.bcast(Beta_xi_init, root = 0)
Scale_vec_current     = np.exp((C_logsigma.T @ Beta_logsigma_current).T)[:,rank]
Shape_vec_current     = ((C_xi.T @ Beta_xi_current).T)[:,rank]

## GPD covariate coefficients prior variance ----------------------------------
sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_init, root = 0)
sigma_Beta_xi_current       = comm.bcast(sigma_Beta_xi_init, root = 0)

## Dependence Model ---------------------------------------------------------------------------------------------------

## S Stable -------------------------------------------------------------------
# note: directly comm.scatter an numpy nd array along an axis is tricky,
#       hence we first "redundantly" broadcast an entire S_matrix then split
S_matrix_init_log = comm.bcast(S_matrix_init_log, root = 0) # matrix (k, Nt)
S_current_log     = np.array(S_matrix_init_log[:,rank]) # vector (k,)
R_vec_current     = wendland_weight_matrix_S @ np.exp(S_current_log)

## gamma ----------------------------------------------------------------------
gamma_k_vec_current   = comm.bcast(gamma_k_vec_init, root = 0)
gamma_bar_vec_current = np.sum(np.multiply(wendland_weight_matrix_S, gamma_k_vec_current)**(alpha),
                                axis = 1)**(1/alpha)

## Z --------------------------------------------------------------------------

Z_matrix_init = comm.bcast(Z_init, root = 0)    # matrix (Ns, Nt)
Z_1t_current  = np.array(Z_matrix_init[:,rank]) # vector (Ns,)

## phi ------------------------------------------------------------------------

phi_knots_current = comm.bcast(phi_knots_init, root = 0)
phi_vec_current   = gaussian_weight_matrix_phi @ phi_knots_current

## rho_vec (length_scale) ---------------------------------------------------

rho_knots_current  = comm.bcast(rho_knots_init, root = 0)
rho_vec_current    = gaussian_weight_matrix_rho @ rho_knots_current
K_current          = ns_cov(range_vec = rho_vec_current,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
# cholesky_matrix_current = scipy.linalg.cholesky(K_current, lower = False)
MVN_frozen_current = scipy.stats.multivariate_normal(mean = None, cov = K_current)

## Nugget standard deviation: tau ---------------------------------------------

tau_current = comm.bcast(tau_init, root = 0)

## X_star ---------------------------------------------------------------------

X_star_1t_current = (R_vec_current ** phi_vec_current) * g(Z_1t_current)

## X (Ns, Nt) -----------------------------------------------------------------

X_matrix_init = comm.bcast(X_init, root = 0) # (Ns, Nt)
X_1t_current  = X_matrix_init[:,rank]        # (Ns,)

## dX (Ns, Nt) ----------------------------------------------------------------

dX_matrix_init = comm.bcast(dX_init, root = 0) # (Ns, Nt)
dX_1t_current  = dX_matrix_init[:,rank]        # (Ns,)

## Y (Ns, Nt) -----------------------------------------------------------------

Y_matrix_init = comm.bcast(Y_matrix_init, root = 0) # (Ns, Nt)
Y_1t_current  = Y_matrix_init[:,rank]               # (Ns,)

# initial imputation ----------------------------------------------------------
# Note:
#   miss_idx_1t and obs_idx_1t STAYS THE SAME throughout the entire MCMC
#   they are part of the "dataset's attribute"
miss_idx_1t = np.where(miss_matrix[:,rank])[0]
obs_idx_1t  = np.where(~miss_matrix[:,rank])[0]

if start_iter == 1:
    if len(miss_idx_1t) > 0:
        X_1t_imputed  = X_star_1t_current[miss_idx_1t] + \
                        scipy.stats.norm.rvs(loc = 0, scale = tau_current, size = len(miss_idx_1t), random_state = random_generator)
        dX_1t_imputed = dRW(X_1t_imputed, phi_vec_current[miss_idx_1t], gamma_bar_vec_current[miss_idx_1t], tau_current)
        Y_1t_imputed  = qCGP(pRW(X_1t_imputed, phi_vec_current[miss_idx_1t], gamma_bar_vec_current[miss_idx_1t], tau_current),
                             p, u_vec[miss_idx_1t], Scale_vec_current[miss_idx_1t], Shape_vec_current[miss_idx_1t])

        X_1t_current[miss_idx_1t]  = X_1t_imputed
        dX_1t_current[miss_idx_1t] = dX_1t_imputed
        Y_1t_current[miss_idx_1t]  = Y_1t_imputed

    assert 0 == len(np.where(np.isnan(X_1t_current))[0])
    assert 0 == len(np.where(np.isnan(dX_1t_current))[0])
    assert 0 == len(np.where(np.isnan(Y_1t_current))[0])

    X_1t_gathered  = comm.gather(X_1t_current,  root = 0)
    dX_1t_gathered = comm.gather(dX_1t_current, root = 0)
    Y_1t_gathered  = comm.gather(Y_1t_current,  root = 0)
    if rank == 0:
        X_trace[0, :, :]  = np.array(X_1t_gathered).T
        dX_trace[0, :, :] = np.array(dX_1t_gathered).T
        Y_trace[0, :, :]  = np.array(Y_1t_gathered).T

# censor/exceedance index -----------------------------------------------------

# Note:
#   The censor/exceedance index NEED TO CHANGE whenever we do imputation
censored_idx_1t_current = np.where(Y_1t_current <= u_vec)[0]
exceed_idx_1t_current   = np.where(Y_1t_current  > u_vec)[0]

# the union of exceed_idx_1t and miss_idx_1t, used for dRW
miss_union_exceed_idx_1t_current = np.union1d(exceed_idx_1t_current, miss_idx_1t)













# %% Metropolis-Hasting Updates -----------------------------------------------------------------------------------

comm.Barrier() # Blocking before the update starts

if rank == 0:
    start_time = time.time()
    print('started on:', strftime('%Y-%m-%d %H:%M:%S', localtime(time.time())))

######################################################################
#### ----- Checking log-likelihood before loops begin -----       ####
######################################################################

# Marginal ----------------------------------------------------------------

if np.any(dCGP(Y_1t_current[exceed_idx_1t_current], p, u_vec[exceed_idx_1t_current], Scale_vec_current[exceed_idx_1t_current], Shape_vec_current[exceed_idx_1t_current]) == 0):
    print('initial Y not possible', 'rank:', rank)
    sys.exit()

# Joint Likelihood --------------------------------------------------------

# llik_1t_current = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
#                         R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
#                         S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

llik_1t_current = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                X_1t_current, dX_1t_current, MVN_frozen_current)

censored_ll_1t, exceed_ll_1t, S_ll_1t, D_gauss_ll_1t = ll_1t_qRWdRWout_detail(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                                                R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                                                                S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                                                                X_1t_current, dX_1t_current, MVN_frozen_current)

if not (np.isfinite(llik_1t_current) and np.isfinite(censored_ll_1t) and np.isfinite(exceed_ll_1t) and np.isfinite(S_ll_1t) and np.isfinite(D_gauss_ll_1t)):
    print('initial likelihood not finite', 'rank:', rank)
    sys.exit()
else:
    censored_ll_gathered     = comm.gather(censored_ll_1t, root = 0)
    exceed_ll_gathered       = comm.gather(exceed_ll_1t,   root = 0)
    S_ll_gathered            = comm.gather(S_ll_1t,        root = 0)
    D_gauss_ll_gathered      = comm.gather(D_gauss_ll_1t,  root = 0)
    llik_1t_current_gathered = comm.gather(llik_1t_current, root = 0)

if start_iter == 1 and rank == 0:
    loglik_trace[0, 0]                = np.sum(llik_1t_current_gathered)
    loglik_detail_trace[0, [0,1,2,3]] = np.sum(np.array([censored_ll_gathered, 
                                                            exceed_ll_gathered, 
                                                            S_ll_gathered,
                                                            D_gauss_ll_gathered]),
                                                axis = 1)

##################################
# Metropolis-Hasting Update Loop #
##################################

for iter in range(start_iter, n_iters+1):
    # %% Update St ------------------------------------------------------------------------------------------------
    ###########################################################
    #### ----- Update St ----- Parallelized Across Nt time ####
    ###########################################################

    if UPDATE_S:

        if rank == 0: print('iter:', iter, 'Update St')

        for i in range(k_S):
            # propose new Stable St at knot i (No need truncation now?) -----------------------------------------------
            change_idx = np.array([i])

            S_proposal_log             = S_current_log.copy()
            S_proposal_log[change_idx] = S_proposal_log[change_idx] + random_generator.normal(0.0, sigma_m_sq_St[i], size = 1)

            R_vec_proposal             = wendland_weight_matrix_S @ np.exp(S_proposal_log)

            # Data Likelihood -----------------------------------------------------------------------------------------

            # # "Full" version, X and dX are calculated within the likelihood function
            # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
            #                          R_vec_proposal, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
            #                          S_proposal_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

            # optimized version, X and dX are not calculated within the likelihood function
            llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                            R_vec_proposal, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                            S_proposal_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_current, dX_1t_current, MVN_frozen_current)

            # Update --------------------------------------------------------------------------------------------------
            r = np.exp(llik_1t_proposal - llik_1t_current)
            u = random_generator.uniform()
            if np.isfinite(r) and r >= u:
                num_accepted_St[i] += 1
                S_current_log       = S_proposal_log.copy()
                R_vec_current       = wendland_weight_matrix_S @ np.exp(S_current_log)
                llik_1t_current     = llik_1t_proposal
            if not np.isfinite(r) and llik_1t_proposal > llik_1t_current and np.isfinite(llik_1t_proposal):
                num_accepted_St[i] += 1
                S_current_log       = S_proposal_log.copy()
                R_vec_current       = wendland_weight_matrix_S @ np.exp(S_current_log)
                llik_1t_current     = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        S_current_log_gathered = comm.gather(S_current_log, root = 0)
        if rank == 0: S_trace_log[iter,:,:]  = np.vstack(S_current_log_gathered).T

        comm.Barrier()

    # %% Update Zt ------------------------------------------------------------------------------------------------
    ###########################################################
    ####                 Update Zt                         ####
    ###########################################################

    if UPDATE_Z:

        if rank == 0: print('iter:', iter, 'Update Zt')

        for block_key in Z_block_idx_dict.keys():
            block_idx     = np.array(Z_block_idx_dict[block_key])
            block_idx_obs = np.intersect1d(block_idx, obs_idx_1t)
            # local_block_idx_obs = block_idx_obs - block_idx[0]
            local_block_idx_obs = np.searchsorted(block_idx, block_idx_obs)
            block_size    = len(block_idx_obs)
            if block_size == 0: continue

            # propose new Zt at sites in block_key  -------------------------------

            Z_1t_proposal                 = Z_1t_current.copy()
            Z_1t_proposal[block_idx_obs] += random_generator.multivariate_normal(np.zeros(block_size),
                                                                                 sigma_m_sq_Zt[block_key] * \
                                                                                 Sigma_0_Zt[block_key][np.ix_(local_block_idx_obs, local_block_idx_obs)])

            # Data Likelihood -----------------------------------------------------

            # optimized version, X and dX are not calculated within the likelihood function
            llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                            R_vec_current, Z_1t_proposal, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_current, dX_1t_current, MVN_frozen_current)

            # MH acceptance -------------------------------------------------------

            r = np.exp(llik_1t_proposal - llik_1t_current)
            Zt_accepted = False
            if np.isfinite(r) and r >= random_generator.uniform():
                Zt_accepted = True
            if (not np.isfinite(r)) and (llik_1t_proposal > llik_1t_current) and np.isfinite(llik_1t_proposal):
                Zt_accepted = True

            if Zt_accepted:
                Z_1t_current                = Z_1t_proposal.copy()
                llik_1t_current             = llik_1t_proposal
                num_accepted_Zt[block_key] += 1

        # Save --------------------------------------------------------------------
        Z_1t_current_gathered = comm.gather(Z_1t_current, root = 0)
        if rank == 0:
            Z_trace[iter,:,:] = np.vstack(Z_1t_current_gathered).T

        # for i in obs_idx_1t: # only update the observed sites
        #     # propose new Zt at site i  -------------------------------------------------------------------------------
        #     idx                = np.array([i])
        #     Z_1t_proposal      = Z_1t_current.copy()
        #     Z_1t_proposal[idx] = Z_1t_current[idx] + np.sqrt(sigma_m_sq_Zt[i]) * random_generator.normal(0.0, 1.0, size = 1)

        #     # Data Likelihood -----------------------------------------------------------------------------------------

        #     # # "Full" version, X and dX are calculated within the likelihood function
        #     # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
        #     #                          R_vec_current, Z_1t_proposal, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
        #     #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

        #     # optimized version, X and dX are not calculated within the likelihood function
        #     llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
        #                                        R_vec_current, Z_1t_proposal, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
        #                                        S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
        #                                        X_1t_current, dX_1t_current, MVN_frozen_current)

        #     # Update --------------------------------------------------------------------------------------------------
        #     r = np.exp(llik_1t_proposal - llik_1t_current)
        #     if np.isfinite(r) and r >= random_generator.uniform():
        #         num_accepted_Zt[i] += 1
        #         Z_1t_current      = Z_1t_proposal.copy()
        #         llik_1t_current   = llik_1t_proposal
        #     if not np.isfinite(r) and llik_1t_proposal > llik_1t_current and np.isfinite(llik_1t_proposal):
        #         num_accepted_Zt[i] += 1
        #         Z_1t_current      = Z_1t_proposal.copy()
        #         llik_1t_current   = llik_1t_proposal

        # # Save --------------------------------------------------------------------------------------------------------
        # Z_1t_current_gathered = comm.gather(Z_1t_current, root = 0)
        # if rank == 0: Z_trace[iter,:,:]  = np.vstack(Z_1t_current_gathered).T

        comm.Barrier()

    # %% Update phi ------------------------------------------------------------------------------------------------
    ############################################################
    ####                 Update phi                         ####
    ############################################################

    if UPDATE_phi:

        if rank == 0: print('iter:', iter, 'Update phi')

        for key in phi_block_idx_dict.keys():
            # Propose new phi_block at the change_indices -------------------------------------------------------------
            idx = np.array(phi_block_idx_dict[key])
            if rank == 0:
                phi_knots_proposal       = phi_knots_current.copy()
                phi_knots_proposal[idx] += np.sqrt(sigma_m_sq[key]) * random_generator.multivariate_normal(np.zeros(len(idx)), Sigma_0[key])
            else:
                phi_knots_proposal = None
            phi_knots_proposal     = comm.bcast(phi_knots_proposal, root = 0)

            # Data Likelihood -----------------------------------------------------------------------------------------
            if not all(0 < phi < 1 for phi in phi_knots_proposal):
                llik_1t_proposal = -np.inf
            else:
                phi_vec_proposal = gaussian_weight_matrix_phi @ phi_knots_proposal

                # # "full" version as X and dX are calculated within the likelihood function
                # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                #                          R_vec_current, Z_1t_current, K_current, phi_vec_proposal, gamma_bar_vec_current, tau_current,
                #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

                # "optimized" version as X and dX are calculated outside
                X_1t_proposal    = qRW_NN_2p(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                                             phi_vec_proposal, gamma_bar_vec_current, tau_current)
                # dX_1t_proposal   = dRW(X_1t_proposal, phi_vec_proposal, gamma_bar_vec_current, tau_current)
                dX_1t_proposal   = dX_1t_current.copy()
                dX_1t_proposal[miss_union_exceed_idx_1t_current] = dRW(X_1t_proposal[miss_union_exceed_idx_1t_current],
                                                                       phi_vec_proposal[miss_union_exceed_idx_1t_current],
                                                                       gamma_bar_vec_current[miss_union_exceed_idx_1t_current],
                                                                       tau_current)
                llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                   R_vec_current, Z_1t_current, K_current, phi_vec_proposal, gamma_bar_vec_current, tau_current,
                                                   S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                                   X_1t_proposal, dX_1t_proposal, MVN_frozen_current)

            # Update --------------------------------------------------------------------------------------------------
            phi_accepted = False
            llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
            llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
            if rank == 0:
                llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(scipy.stats.beta.logpdf(phi_knots_current, a = 5, b = 5))
                llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(scipy.stats.beta.logpdf(phi_knots_proposal, a = 5, b = 5))
                # print('phi_update', 'llik_proposal:', llik_proposal, 'llik_current:', llik_current)
                r = np.exp(llik_proposal - llik_current)
                if np.isfinite(r) and r >= random_generator.uniform():
                    num_accepted[key] += 1
                    phi_accepted       = True
                if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                    num_accepted[key] += 1
                    phi_accepted       = True
            phi_accepted = comm.bcast(phi_accepted, root = 0)

            if phi_accepted:
                phi_knots_current = phi_knots_proposal.copy()
                phi_vec_current   = phi_vec_proposal.copy()
                X_1t_current      = X_1t_proposal.copy()
                dX_1t_current     = dX_1t_proposal.copy()
                llik_1t_current   = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
        dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
        if rank == 0:
            phi_knots_trace[iter,:] = phi_knots_current.copy()
            X_trace[iter,:,:]       = np.vstack(X_1t_current_gathered).T
            dX_trace[iter,:,:]      = np.vstack(dX_1t_current_gathered).T

        comm.Barrier()

    # %% Update rho ------------------------------------------------------------------------------------------------
    ############################################################
    ####                 Update rho                         ####
    ############################################################

    if UPDATE_rho:

        if rank == 0: print('iter:', iter, 'Update rho')

        for key in rho_block_idx_dict.keys():
            # Propose new rho_block at the change_indices -----------------------------------------------------------
            idx = np.array(rho_block_idx_dict[key])
            if rank == 0:
                rho_knots_proposal = rho_knots_current.copy()
                rho_knots_proposal[idx] += np.sqrt(sigma_m_sq[key]) * random_generator.multivariate_normal(np.zeros(len(idx)), Sigma_0[key])
            else:
                rho_knots_proposal = None
            rho_knots_proposal     = comm.bcast(rho_knots_proposal, root = 0)

            # Data Likelihood -----------------------------------------------------------------------------------------
            if not all(rho > 0 for rho in rho_knots_proposal):
                llik_1t_proposal = -np.inf
            else:
                rho_vec_proposal = gaussian_weight_matrix_rho @ rho_knots_proposal
                K_proposal = ns_cov(range_vec = rho_vec_proposal,
                                    sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = "matern")
                MVN_frozen_proposal = scipy.stats.multivariate_normal(mean = None, cov = K_proposal)

                # # "full" version as X and dX are calculated within the likelihood function
                # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                #                          R_vec_current, Z_1t_current, K_proposal, phi_vec_current, gamma_bar_vec_current, tau_current,
                #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

                # "optimized" version as X and dX are calculated outside the likelihood function
                llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                R_vec_current, Z_1t_current, K_proposal, phi_vec_current, gamma_bar_vec_current, tau_current,
                                                S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                                X_1t_current, dX_1t_current, MVN_frozen_proposal)

            # Update --------------------------------------------------------------------------------------------------
            rho_accepted = False
            llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
            llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
            if rank == 0:
                llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(rho_knots_current, loc = 0, scale = 2))
                llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(rho_knots_proposal, loc = 0, scale = 2))
                r = np.exp(llik_proposal - llik_current)
                if np.isfinite(r) and r >= random_generator.uniform():
                    num_accepted[key] += 1
                    rho_accepted     = True
                if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                    num_accepted[key] += 1
                    rho_accepted     = True
            rho_accepted = comm.bcast(rho_accepted, root = 0)

            if rho_accepted:
                rho_knots_current = rho_knots_proposal.copy()
                K_current           = K_proposal.copy()
                llik_1t_current     = llik_1t_proposal
                MVN_frozen_current  = scipy.stats.multivariate_normal(mean = None, cov = K_current)

        # Save --------------------------------------------------------------------------------------------------------
        if rank == 0: rho_knots_trace[iter,:] = rho_knots_current.copy()
        comm.Barrier()

    # %% Update gamma_k_vec ------------------------------------------------------------------------------------------------
    ###########################################################
    ####                 Update gamma_k                    ####
    ###########################################################

    if UPDATE_gamma_k:

        if rank == 0: print('iter:', iter, 'Update gamma_k_vec')

        for i in range(k_S):
            # propose new gamma at knot i ---------------------------------------------------------------------------------

            if rank == 0:
                gamma_k_vec_proposal    = gamma_k_vec_current.copy()
                gamma_k_vec_proposal[i] = gamma_k_vec_current[i] + \
                                            np.sqrt(sigma_m_sq['gamma_k_vec'][i]) * random_generator.normal(0.0, 1.0, size = None)
                # Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
            else:
                gamma_k_vec_proposal = None
            gamma_k_vec_proposal = comm.bcast(gamma_k_vec_proposal, root = 0)

            # Data Likelihood -----------------------------------------------------------------------------------------
            if not np.all(0 < gamma_k_vec_proposal):
                llik_1t_proposal = -np.inf
            else:
                gamma_bar_vec_proposal = np.sum(np.multiply(wendland_weight_matrix_S, gamma_k_vec_proposal)**(alpha),
                                            axis = 1)**(1/alpha)

                # # "Full" version, X and dX are calculated within the likelihood function
                # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                #                             R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_proposal, tau_current,
                #                             S_current_log, gamma_k_vec_proposal, censored_idx_1t_current, exceed_idx_1t_current)

                # optimized version, X and dX are not calculated within the likelihood function
                X_1t_proposal    = qRW_NN_2p(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                                            phi_vec_current, gamma_bar_vec_proposal, tau_current)
                # dX_1t_proposal   = dRW(X_1t_proposal, phi_vec_current, gamma_bar_vec_proposal, tau_current)
                dX_1t_proposal   = dX_1t_current.copy()
                dX_1t_proposal[miss_union_exceed_idx_1t_current] = dRW(X_1t_proposal[miss_union_exceed_idx_1t_current],
                                                                    phi_vec_current[miss_union_exceed_idx_1t_current],
                                                                    gamma_bar_vec_proposal[miss_union_exceed_idx_1t_current],
                                                                    tau_current)
                llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_proposal, tau_current,
                                                S_current_log, gamma_k_vec_proposal, censored_idx_1t_current, exceed_idx_1t_current,
                                                X_1t_proposal, dX_1t_proposal, MVN_frozen_current)

            # Update --------------------------------------------------------------------------------------------------
            gamma_accepted = False
            llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
            llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
            if rank == 0:
                llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(scipy.stats.halfnorm.logpdf(gamma_k_vec_current, loc = 0, scale = 2))
                llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(scipy.stats.halfnorm.logpdf(gamma_k_vec_proposal, loc = 0, scale = 2))
                r = np.exp(llik_proposal - llik_current)
                if np.isfinite(r) and r >= random_generator.uniform():
                    num_accepted['gamma_k_vec'][i] += 1
                    gamma_accepted                  = True
                if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                    num_accepted['gamma_k_vec'][i] += 1
                    gamma_accepted                  = True
            gamma_accepted = comm.bcast(gamma_accepted, root = 0)

            if gamma_accepted:
                gamma_k_vec_current   = gamma_k_vec_proposal.copy()
                gamma_bar_vec_current = gamma_bar_vec_proposal.copy()
                X_1t_current          = X_1t_proposal.copy()
                dX_1t_current         = dX_1t_proposal.copy()
                llik_1t_current       = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
        dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
        if rank == 0:
            gamma_k_vec_trace[iter,:] = gamma_k_vec_current
            X_trace[iter,:,:]         = np.vstack(X_1t_current_gathered).T
            dX_trace[iter,:,:]        = np.vstack(dX_1t_current_gathered).T
        comm.Barrier()

    # %% Update tau ------------------------------------------------------------------------------------------------
    ############################################################
    ####                 Update tau                         ####
    ############################################################

    if UPDATE_tau:

        if rank == 0: print('iter:', iter, 'Update tau')

        # Propose new tau ---------------------------------------------------------------------------------------------
        if rank == 0:
            tau_proposal = tau_current + np.sqrt(sigma_m_sq['tau']) * random_generator.normal(0.0, 1.0)
        else:
            tau_proposal = None
        tau_proposal = comm.bcast(tau_proposal, root = 0)

        # Data Likelihood ---------------------------------------------------------------------------------------------
        if not tau_proposal > 0:
            llik_1t_proposal = -np.inf
        else:
            # # "full" version as X and dX are calculated within the likelihood function
            # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
            #                          R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_proposal,
            #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

            # "optimized" version as X and dX are calculated outside the likelihood function
            X_1t_proposal   = qRW_NN_2p(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current),
                                        phi_vec_current, gamma_bar_vec_current, tau_proposal)
            # dX_1t_proposal   = dRW(X_1t_proposal, phi_vec_current, gamma_bar_vec_current, tau_proposal)
            dX_1t_proposal   = dX_1t_current.copy()
            dX_1t_proposal[miss_union_exceed_idx_1t_current] = dRW(X_1t_proposal[miss_union_exceed_idx_1t_current],
                                                                phi_vec_current[miss_union_exceed_idx_1t_current],
                                                                gamma_bar_vec_current[miss_union_exceed_idx_1t_current],
                                                                tau_proposal)
            llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                            R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_proposal,
                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_proposal, dX_1t_proposal, MVN_frozen_current)

        # Update ------------------------------------------------------------------------------------------------------
        tau_accepted = False
        llik_1t_current_gathered  = comm.gather(llik_1t_current, root = 0)
        llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)
        if rank == 0:
            lprior_tau_current  = np.log(dhalft(tau_current, nu = 1, mu = 0, sigma = 5))
            lprior_tau_proposal = np.log(dhalft(tau_proposal, nu = 1, mu = 0, sigma = 5)) if tau_proposal > 0 else -np.inf
            llik_current  = np.sum(llik_1t_current_gathered) + lprior_tau_current
            llik_proposal = np.sum(llik_1t_proposal_gathered) + lprior_tau_proposal
            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['tau'] += 1
                tau_accepted         = True
            if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                num_accepted['tau'] += 1
                tau_accepted         = True
        tau_accepted = comm.bcast(tau_accepted, root = 0)

        if tau_accepted:
            tau_current     = tau_proposal
            X_1t_current    = X_1t_proposal.copy()
            dX_1t_current   = dX_1t_proposal.copy()
            llik_1t_current = llik_1t_proposal

        # Save --------------------------------------------------------------------------------------------------------
        X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
        dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
        if rank == 0:
            tau_trace[iter,:]  = tau_current
            X_trace[iter,:,:]  = np.vstack(X_1t_current_gathered).T
            dX_trace[iter,:,:] = np.vstack(dX_1t_current_gathered).T
        comm.Barrier()

    # %% Update GPD sigma ---------------------------------------------------------------------------------------------
    ############################################################
    ####                 Update GPD sigma                   ####
    ############################################################

    if UPDATE_GPD_sigma:

        if rank == 0: print('iter:', iter, 'Update GPD sigma')

        # Propose new Beta's for logsigma ---------------------------------------------------------------------------------
        if rank == 0:
            Beta_logsigma_proposal = Beta_logsigma_current + np.sqrt(sigma_m_sq['Beta_logsigma']) * \
                                    random_generator.multivariate_normal(np.zeros(Beta_logsigma_m), Sigma_0['Beta_logsigma'])
        else:
            Beta_logsigma_proposal = None
        Beta_logsigma_proposal = comm.bcast(Beta_logsigma_proposal, root = 0)
        Scale_vec_proposal     = np.exp((C_logsigma.T @ Beta_logsigma_proposal).T)[:,rank]

        # Data Likelihood ---------------------------------------------------------------------------------------------
        if np.any(Scale_vec_proposal <= 0):
            llik_1t_proposal = -np.inf
        elif np.any(dCGP(Y_1t_current[exceed_idx_1t_current], p, u_vec[exceed_idx_1t_current], Scale_vec_proposal[exceed_idx_1t_current], Shape_vec_current[exceed_idx_1t_current]) == 0):
            llik_1t_proposal = -np.inf
        else:
            # # "full" version as X and dX are calculated within the likelihood function
            # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_proposal, Shape_vec_current,
            #                          R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
            #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

            # "optimized" version as X and dX are calculated outside the likelihood function
            X_1t_proposal   = qRW_NN_2p(pCGP(Y_1t_current, p, u_vec, Scale_vec_proposal, Shape_vec_current),
                                        phi_vec_current, gamma_bar_vec_current, tau_current)
            # dX_1t_proposal  = dRW(X_1t_proposal, phi_vec_current, gamma_bar_vec_current, tau_current)
            dX_1t_proposal   = dX_1t_current.copy()
            dX_1t_proposal[miss_union_exceed_idx_1t_current] = dRW(X_1t_proposal[miss_union_exceed_idx_1t_current],
                                                                phi_vec_current[miss_union_exceed_idx_1t_current],
                                                                gamma_bar_vec_current[miss_union_exceed_idx_1t_current],
                                                                tau_current)
            llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_proposal, Shape_vec_current,
                                            R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_proposal, dX_1t_proposal, MVN_frozen_current)

        # Update ------------------------------------------------------------------------------------------------------
        Beta_logsigma_accepted = False
        llik_1t_current_gathered  = comm.gather(llik_1t_current,  root = 0)
        llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)

        if rank == 0:
            lprior_Beta_logsigma_current  = scipy.stats.norm.logpdf(Beta_logsigma_current,
                                                                    loc = 0, scale = sigma_Beta_logsigma_current)
            lprior_Beta_logsigma_proposal = scipy.stats.norm.logpdf(Beta_logsigma_proposal,
                                                                    loc = 0, scale = sigma_Beta_logsigma_current)

            llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(lprior_Beta_logsigma_current)
            llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(lprior_Beta_logsigma_proposal)

            # print('GPD sigma update', 'llik_proposal:', llik_proposal, 'llik_current:', llik_current)

            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['Beta_logsigma'] += 1
                Beta_logsigma_accepted = True
            if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                num_accepted['Beta_logsigma'] += 1
                Beta_logsigma_accepted = True
        Beta_logsigma_accepted = comm.bcast(Beta_logsigma_accepted, root = 0)

        if Beta_logsigma_accepted:
            Beta_logsigma_current = Beta_logsigma_proposal.copy()
            llik_1t_current       = llik_1t_proposal
            X_1t_current          = X_1t_proposal.copy()
            dX_1t_current         = dX_1t_proposal.copy()

        # Save --------------------------------------------------------------------------------------------------------
        X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
        dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
        if rank == 0:
            Beta_logsigma_trace[iter,:] = Beta_logsigma_current
            X_trace[iter,:,:]           = np.vstack(X_1t_current_gathered).T
            dX_trace[iter,:,:]          = np.vstack(dX_1t_current_gathered).T
        comm.Barrier()

    # %% Update GPD xi ------------------------------------------------------------------------------------------------
    ############################################################
    ####                 Update GPD xi                      ####
    ############################################################

    if UPDATE_GPD_xi:

        if rank == 0: print('iter:', iter, 'Update GPD xi')

        # propose new Beta's for xi ---------------------------------------------------------------------------------------
        if rank == 0:
            Beta_xi_proposal = Beta_xi_current + np.sqrt(sigma_m_sq['Beta_xi']) * \
                                random_generator.multivariate_normal(np.zeros(Beta_xi_m), Sigma_0['Beta_xi'])
        else:
            Beta_xi_proposal = None
        Beta_xi_proposal   = comm.bcast(Beta_xi_proposal, root = 0)
        Shape_vec_proposal = ((C_xi.T @ Beta_xi_proposal).T)[:,rank]

        # Data Likelihood ---------------------------------------------------------------------------------------------

        if np.any(Shape_vec_proposal <= -1) or np.any(Shape_vec_proposal >= 1):
            llik_1t_proposal = -np.inf
        elif np.any(dCGP(Y_1t_current[exceed_idx_1t_current], p, u_vec[exceed_idx_1t_current], Scale_vec_current[exceed_idx_1t_current], Shape_vec_proposal[exceed_idx_1t_current]) == 0):
            llik_1t_proposal = -np.inf
        else:
            # # "full" version as X and dX are calculated within the likelihood function
            # llik_1t_proposal = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_proposal,
            #                          R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
            #                          S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

            # "optimized" version as X and dX are calculated outside the likelihood function
            X_1t_proposal    = qRW_NN_2p(pCGP(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_proposal),
                                        phi_vec_current, gamma_bar_vec_current, tau_current)
            # dX_1t_proposal   = dRW(X_1t_proposal, phi_vec_current, gamma_bar_vec_current, tau_current)
            dX_1t_proposal   = dX_1t_current.copy()
            dX_1t_proposal[miss_union_exceed_idx_1t_current] = dRW(X_1t_proposal[miss_union_exceed_idx_1t_current],
                                                                   phi_vec_current[miss_union_exceed_idx_1t_current],
                                                                   gamma_bar_vec_current[miss_union_exceed_idx_1t_current],
                                                                   tau_current)
            llik_1t_proposal = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_proposal,
                                            R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_proposal, dX_1t_proposal, MVN_frozen_current)

        # Update ------------------------------------------------------------------------------------------------------
        Beta_xi_accepted = False
        llik_1t_current_gathered  = comm.gather(llik_1t_current,  root = 0)
        llik_1t_proposal_gathered = comm.gather(llik_1t_proposal, root = 0)

        if rank == 0:
            lprior_Beta_xi_current  = scipy.stats.norm.logpdf(Beta_xi_current,
                                                               loc = 0, scale = sigma_Beta_xi_current)
            lprior_Beta_xi_proposal = scipy.stats.norm.logpdf(Beta_xi_proposal,
                                                               loc = 0, scale = sigma_Beta_xi_current)

            llik_current  = np.sum(llik_1t_current_gathered)  + np.sum(lprior_Beta_xi_current)
            llik_proposal = np.sum(llik_1t_proposal_gathered) + np.sum(lprior_Beta_xi_proposal)

            # print('GPD xi update', 'llik_proposal:', llik_proposal, 'llik_current:', llik_current)

            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['Beta_xi'] += 1
                Beta_xi_accepted = True
            if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                num_accepted['Beta_xi'] += 1
                Beta_xi_accepted = True
        Beta_xi_accepted = comm.bcast(Beta_xi_accepted, root = 0)

        if Beta_xi_accepted:
            Beta_xi_current = Beta_xi_proposal.copy()
            llik_1t_current = llik_1t_proposal
            X_1t_current    = X_1t_proposal.copy()
            dX_1t_current   = dX_1t_proposal.copy()

        # Save --------------------------------------------------------------------------------------------------------
        X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
        dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
        if rank == 0:
            Beta_xi_trace[iter,:] = Beta_xi_current
            X_trace[iter,:,:]     = np.vstack(X_1t_current_gathered).T
            dX_trace[iter,:,:]    = np.vstack(dX_1t_current_gathered).T
        comm.Barrier()

    # %% Update Regularization ------------------------------------------------------------------------------------
    ############################################################
    ####       Update Regularization (sigma_Beta_xx)        ####
    ############################################################

    if UPDATE_Regularization:

        if rank == 0: print('iter:', iter, 'Update Regularization (sigma_Beta_xx)')

        # sigma_Beta_logsigma ---------------------------------------------------------------------------------------------
        if rank == 0:
            # propose new sigma_Beta_logsigma -----------------------------------------------------------------------------
            sigma_Beta_logsigma_proposal = sigma_Beta_logsigma_current + np.sqrt(sigma_m_sq['sigma_Beta_logsigma']) * random_generator.standard_normal()

            # likelihood --------------------------------------------------------------------------------------------------
            lprior_sigma_Beta_logsigma_current  = np.log(dhalft(sigma_Beta_logsigma_current, nu = 2))
            lprior_sigma_Beta_logsigma_proposal = np.log(dhalft(sigma_Beta_logsigma_proposal, nu = 2)) if sigma_Beta_logsigma_proposal > 0 else -np.inf

            llik_current  = lprior_sigma_Beta_logsigma_current  + np.sum(scipy.stats.norm.logpdf(Beta_logsigma_current, loc = 0, scale = sigma_Beta_logsigma_current))
            llik_proposal = lprior_sigma_Beta_logsigma_proposal + np.sum(scipy.stats.norm.logpdf(Beta_logsigma_current, loc = 0, scale = sigma_Beta_logsigma_proposal))

            # Update ------------------------------------------------------------------------------------------------------
            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['sigma_Beta_logsigma'] += 1
                sigma_Beta_logsigma_current = sigma_Beta_logsigma_proposal
            if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                num_accepted['sigma_Beta_logsigma'] += 1
                sigma_Beta_logsigma_current = sigma_Beta_logsigma_proposal

            # Save --------------------------------------------------------------------------------------------------------
            sigma_Beta_logsigma_trace[iter,:] = sigma_Beta_logsigma_current

        # (unnecessary) Broadcast
        sigma_Beta_logsigma_current = comm.bcast(sigma_Beta_logsigma_current, root = 0)


        # sigma_Beta_xi ---------------------------------------------------------------------------------------------------
        if rank == 0:
            # propose new sigma_Beta_xi -----------------------------------------------------------------------------------
            sigma_Beta_xi_proposal = sigma_Beta_xi_current + np.sqrt(sigma_m_sq['sigma_Beta_xi']) * random_generator.standard_normal()

            # likelihood --------------------------------------------------------------------------------------------------
            lprior_sigma_Beta_xi_current  = np.log(dhalft(sigma_Beta_xi_current,  nu = 2))
            lprior_sigma_Beta_xi_proposal = np.log(dhalft(sigma_Beta_xi_proposal, nu = 2)) if sigma_Beta_xi_proposal > 0 else -np.inf

            llik_current  = lprior_sigma_Beta_xi_current  + np.sum(scipy.stats.norm.logpdf(Beta_xi_current, loc = 0, scale = sigma_Beta_xi_current))
            llik_proposal = lprior_sigma_Beta_xi_proposal + np.sum(scipy.stats.norm.logpdf(Beta_xi_current, loc = 0, scale = sigma_Beta_xi_proposal))

            # Update ------------------------------------------------------------------------------------------------------
            r = np.exp(llik_proposal - llik_current)
            if np.isfinite(r) and r >= random_generator.uniform():
                num_accepted['sigma_Beta_xi'] += 1
                sigma_Beta_xi_current = sigma_Beta_xi_proposal
            if not np.isfinite(r) and llik_proposal > llik_current and np.isfinite(llik_proposal):
                num_accepted['sigma_Beta_xi'] += 1
                sigma_Beta_xi_current = sigma_Beta_xi_proposal

            # Save --------------------------------------------------------------------------------------------------------
            sigma_Beta_xi_trace[iter,:] = sigma_Beta_xi_current

        # (unnecessary) Broadcast
        sigma_Beta_xi_current = comm.bcast(sigma_Beta_xi_current, root = 0)

    # %% Imputation ---------------------------------------------------------------------------------------------------
    ######################################################################
    #### ----- Imputation of (Z_miss, Y_miss)  -----                  ####
    ######################################################################

    if UPDATE_Imputation:

        if rank == 0: print('iter:', iter, 'Imputation of (Z_miss, Y_miss)')

        if len(miss_idx_1t) > 0:

            # Impute Z and Y --------------------------------------------------------------------------------------------------
            Z_1t_miss, X_1t_miss, Y_1t_miss = impute_ZY_1t(p, u_vec, Scale_vec_current, Shape_vec_current,
                                                        R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                                        obs_idx_1t, miss_idx_1t)

            # Update ----------------------------------------------------------------------------------------------------------
            Z_1t_current[miss_idx_1t]  = Z_1t_miss
            X_1t_current[miss_idx_1t]  = X_1t_miss
            Y_1t_current[miss_idx_1t]  = Y_1t_miss
            dX_1t_current[miss_idx_1t] = dRW(X_1t_miss, phi_vec_current[miss_idx_1t], gamma_bar_vec_current[miss_idx_1t], tau_current)

            # # "full" version as X and dX are calculated within the likelihood function
            # llik_1t_current = ll_1t(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
            #                         R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
            #                         S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

            # "optimized" version as X and dX are calculated outside the likelihood function
            llik_1t_current = ll_1t_qRWdRWout(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                            R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                            X_1t_current, dX_1t_current, MVN_frozen_current)

            # update censoring
            censored_idx_1t_current = np.where(Y_1t_current <= u_vec)[0]
            exceed_idx_1t_current   = np.where(Y_1t_current >  u_vec)[0]
            miss_union_exceed_idx_1t_current = np.union1d(exceed_idx_1t_current, miss_idx_1t)

            # Save ------------------------------------------------------------------------------------------------------------
            Z_1t_current_gathered  = comm.gather(Z_1t_current, root = 0)
            X_1t_current_gathered  = comm.gather(X_1t_current, root = 0)
            Y_1t_current_gathered  = comm.gather(Y_1t_current, root = 0)
            dX_1t_current_gathered = comm.gather(dX_1t_current, root = 0)
            if rank == 0:
                Z_trace[iter,:,:]  = np.vstack(Z_1t_current_gathered).T
                X_trace[iter,:,:]  = np.vstack(X_1t_current_gathered).T
                Y_trace[iter,:,:]  = np.vstack(Y_1t_current_gathered).T
                dX_trace[iter,:,:] = np.vstack(dX_1t_current_gathered).T

        comm.Barrier()

    # %% After iteration likelihood
    ######################################################################
    #### ----- Keeping track of likelihood after this iteration ----- ####
    ######################################################################

    llik_1t_current_gathered = comm.gather(llik_1t_current, root = 0)
    if rank == 0: loglik_trace[iter, 0] = np.sum(llik_1t_current_gathered)

    # # "full" version, as X and dX are calculated within the likelihood function
    # censored_ll_1t, exceed_ll_1t, S_ll_1t, D_gauss_ll_1t = ll_1t_detail(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
    #                                                            R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
    #                                                            S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current)

    # "optimized" version as X and dX are calculated outside the likelihood function
    censored_ll_1t, exceed_ll_1t, S_ll_1t, D_gauss_ll_1t = ll_1t_qRWdRWout_detail(Y_1t_current, p, u_vec, Scale_vec_current, Shape_vec_current,
                                                                                  R_vec_current, Z_1t_current, K_current, phi_vec_current, gamma_bar_vec_current, tau_current,
                                                                                  S_current_log, gamma_k_vec_current, censored_idx_1t_current, exceed_idx_1t_current,
                                                                                  X_1t_current, dX_1t_current, MVN_frozen_current)

    censored_ll_gathered = comm.gather(censored_ll_1t, root = 0)
    exceed_ll_gathered   = comm.gather(exceed_ll_1t,   root = 0)
    S_ll_gathered        = comm.gather(S_ll_1t,        root = 0)
    D_gauss_ll_gathered  = comm.gather(D_gauss_ll_1t,  root = 0)

    if rank == 0: loglik_detail_trace[iter, [0,1,2,3]] = np.sum(np.array([censored_ll_gathered, exceed_ll_gathered, S_ll_gathered,D_gauss_ll_gathered]),
                                                                axis = 1)
    comm.Barrier()

    # %% Adaptive Update tunings --------------------------------------------------------------------------------------
    #####################################################
    ###### ----- Adaptive Update autotunings ----- ######
    #####################################################

    if iter % ADAPT_SIZE == 0:

        comm.Barrier()

        gamma1 = 1 / ((iter/ADAPT_SIZE + OFFSET) ** C1)
        gamma2 = C0 * gamma1

        # St ------------------------------------------------------------------

        if UPDATE_S and norm_pareto == 'standard':

            r_hat_St = np.full(k_S, np.nan)

            for i in range(k_S):
                r_hat              = num_accepted_St[i]/ADAPT_SIZE
                num_accepted_St[i] = 0
                r_hat_St[i]        = r_hat
                log_sigma_m_sq_hat = np.log(sigma_m_sq_St[i]) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq_St[i]   = np.exp(log_sigma_m_sq_hat)
            comm.Barrier()
            sigma_m_sq_St_list     = comm.gather(sigma_m_sq_St, root = 0)
            r_hat_St_list          = comm.gather(r_hat_St,      root = 0) # a list of Nt elements, each element is a np.array of length k_S

        # Zt ------------------------------------------------------------------

        if UPDATE_Z:

            r_hat_Zt = np.full(len(Z_block_idx_dict.keys()), np.nan)

            Z_trace_1t = np.zeros(ADAPT_SIZE * Ns, dtype=np.float64)
            if rank == 0:
                Z_reordered = np.transpose(Z_trace[iter-ADAPT_SIZE+1:iter+1,:,:], (2, 0, 1)) # shape is now (Nt, ADAPT_SIZE, Ns)
                sendbuf     = Z_reordered.reshape(Nt, ADAPT_SIZE*Ns)
            else:
                sendbuf = None
            comm.Scatter(sendbuf, Z_trace_1t, root = 0)
            Z_trace_1t = Z_trace_1t.reshape((ADAPT_SIZE, Ns))

            for i, block_key in enumerate(Z_block_idx_dict.keys()):
                # acceptance ratio
                r_hat                      = num_accepted_Zt[block_key] / ADAPT_SIZE
                num_accepted_Zt[block_key] = 0
                r_hat_Zt[i]                = r_hat
                # scalar
                log_sigma_m_sq_hat         = np.log(sigma_m_sq_Zt[block_key]) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq_Zt[block_key]   = np.exp(log_sigma_m_sq_hat)
                # block covariance
                block_idx             = np.array(Z_block_idx_dict[block_key])
                block_history         = Z_trace_1t[:, block_idx]
                Sigma_0_hat           = np.array(np.cov(block_history.T))
                Sigma_0_Zt[block_key] = Sigma_0_Zt[block_key] + gamma1 * (Sigma_0_hat - Sigma_0_Zt[block_key])
            comm.Barrier()
            sigma_m_sq_Zt_list = comm.gather(sigma_m_sq_Zt, root = 0)
            Sigma_0_Zt_list    = comm.gather(Sigma_0_Zt,    root = 0)
            r_hat_Zt_list      = comm.gather(r_hat_Zt,      root = 0) # a list of Nt elements, each element is a dictionary of length len(Z_block_idx_dict.keys())

            # for i in range(Ns):
            #     r_hat              = num_accepted_Zt[i]/ADAPT_SIZE
            #     num_accepted_Zt[i] = 0
            #     log_sigma_m_sq_hat = np.log(sigma_m_sq_Zt[i]) + gamma2 * (r_hat - R_OPT)
            #     sigma_m_sq_Zt[i]   = np.exp(log_sigma_m_sq_hat)
            # comm.Barrier()
            # sigma_m_sq_Zt_list = comm.gather(sigma_m_sq_Zt, root = 0)

        # gamma_k -------------------------------------------------------------

        if UPDATE_gamma_k:
            if rank == 0:

                r_hat_gamma_k = np.full(k_S, np.nan)

                for i in range(k_S):
                    r_hat                             = num_accepted['gamma_k_vec'][i]/ADAPT_SIZE
                    num_accepted['gamma_k_vec'][i]    = 0
                    r_hat_gamma_k[i]                  = r_hat
                    log_sigma_m_sq_hat                = np.log(sigma_m_sq['gamma_k_vec'][i]) + gamma2 * (r_hat - R_OPT)
                    sigma_m_sq['gamma_k_vec'][i]      = np.exp(log_sigma_m_sq_hat)

        # phi -----------------------------------------------------------------

        if UPDATE_phi:
            if rank == 0:

                r_hat_phi = np.full(len(phi_block_idx_dict.keys()), np.nan)

                for i, key in enumerate(phi_block_idx_dict.keys()):
                    start_idx          = phi_block_idx_dict[key][0]
                    end_idx            = phi_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/ADAPT_SIZE
                    num_accepted[key]  = 0
                    r_hat_phi[i]       = r_hat
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - R_OPT)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(phi_knots_trace[iter-ADAPT_SIZE+1:iter+1, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

        # rho ---------------------------------------------------------------

        if UPDATE_rho:
            if rank == 0:

                r_hat_rho = np.full(len(rho_block_idx_dict.keys()), np.nan)

                for i, key in enumerate(rho_block_idx_dict.keys()):
                    start_idx          = rho_block_idx_dict[key][0]
                    end_idx            = rho_block_idx_dict[key][-1]+1
                    r_hat              = num_accepted[key]/ADAPT_SIZE
                    num_accepted[key]  = 0
                    r_hat_rho[i]       = r_hat
                    log_sigma_m_sq_hat = np.log(sigma_m_sq[key]) + gamma2 * (r_hat - R_OPT)
                    sigma_m_sq[key]    = np.exp(log_sigma_m_sq_hat)
                    Sigma_0_hat        = np.array(np.cov(rho_knots_trace[iter-ADAPT_SIZE+1:iter+1, start_idx:end_idx].T))
                    Sigma_0[key]       = Sigma_0[key] + gamma1 * (Sigma_0_hat - Sigma_0[key])

        # tau -----------------------------------------------------------------

        if UPDATE_tau:
            if rank == 0:

                r_hat               = num_accepted['tau']/ADAPT_SIZE
                num_accepted['tau'] = 0
                log_sigma_m_sq_hat  = np.log(sigma_m_sq['tau']) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq['tau']   = np.exp(log_sigma_m_sq_hat)

                r_hat_tau           = r_hat

        # GPD log(sigma) ------------------------------------------------------

        if UPDATE_GPD_sigma:
            if rank == 0:
                r_hat                         = num_accepted['Beta_logsigma']/ADAPT_SIZE
                num_accepted['Beta_logsigma'] = 0
                log_sigma_m_sq_hat            = np.log(sigma_m_sq['Beta_logsigma']) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq['Beta_logsigma']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat                   = np.array(np.cov(Beta_logsigma_trace[iter-ADAPT_SIZE+1:iter+1].T))
                Sigma_0['Beta_logsigma']      = Sigma_0['Beta_logsigma'] + gamma1 * (Sigma_0_hat - Sigma_0['Beta_logsigma'])

                r_hat_Beta_logsigma           = r_hat

        # GPD xi --------------------------------------------------------------

        if UPDATE_GPD_xi:
            if rank == 0:
                r_hat                   = num_accepted['Beta_xi']/ADAPT_SIZE
                num_accepted['Beta_xi'] = 0
                log_sigma_m_sq_hat      = np.log(sigma_m_sq['Beta_xi']) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq['Beta_xi']   = np.exp(log_sigma_m_sq_hat)
                Sigma_0_hat             = np.array(np.cov(Beta_xi_trace[iter-ADAPT_SIZE+1:iter+1].T))
                Sigma_0['Beta_xi']      = Sigma_0['Beta_xi'] + gamma1 * (Sigma_0_hat - Sigma_0['Beta_xi'])

                r_hat_Beta_xi           = r_hat

        # Regularization ------------------------------------------------------

        if UPDATE_Regularization:

            # Regularization on Beta_logsigma

            if rank == 0:
                r_hat                               = num_accepted['sigma_Beta_logsigma']/ADAPT_SIZE
                num_accepted['sigma_Beta_logsigma'] = 0
                log_sigma_m_sq_hat                  = np.log(sigma_m_sq['sigma_Beta_logsigma']) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq['sigma_Beta_logsigma']   = np.exp(log_sigma_m_sq_hat)

                r_hat_sigma_Beta_logsigma           = r_hat

            # Regularization on Beta_xi

            if rank == 0:
                r_hat                         = num_accepted['sigma_Beta_xi']/ADAPT_SIZE
                num_accepted['sigma_Beta_xi'] = 0
                log_sigma_m_sq_hat            = np.log(sigma_m_sq['sigma_Beta_xi']) + gamma2 * (r_hat - R_OPT)
                sigma_m_sq['sigma_Beta_xi']   = np.exp(log_sigma_m_sq_hat)

                r_hat_sigma_Beta_xi           = r_hat

        # Tracking r_hat history ----------------------------------------------

        if rank == 0:
            adapt_idx = (iter // ADAPT_SIZE) - 1
            if UPDATE_S:              r_hat_S_history[adapt_idx,:,:]           = np.array(r_hat_St_list)
            if UPDATE_Z:              r_hat_Z_history[adapt_idx,:,:]           = np.array(r_hat_Zt_list)
            if UPDATE_phi:            r_hat_phi_history[adapt_idx,:]           = r_hat_phi
            if UPDATE_rho:            r_hat_rho_history[adapt_idx,:]           = r_hat_rho
            if UPDATE_gamma_k:        r_hat_gamma_k_history[adapt_idx,:]       = r_hat_gamma_k
            if UPDATE_tau:            r_hat_tau_history[adapt_idx,:]           = r_hat_tau
            if UPDATE_GPD_sigma:      r_hat_Beta_logsigma_history[adapt_idx,:] = r_hat_Beta_logsigma
            if UPDATE_GPD_xi:         r_hat_Beta_xi_history[adapt_idx,:]       = r_hat_Beta_xi
            if UPDATE_Regularization: r_hat_sigma_Beta_history[adapt_idx,:]    = np.array([r_hat_sigma_Beta_logsigma, r_hat_sigma_Beta_xi])

    comm.Barrier()

    # %% Midway Printing, Drawings, and Savings -----------------------------------------------------------------------
    ##############################################
    ###    Printing, Drawings, Savings       #####
    ##############################################

    if rank == 0:

        if iter % THIN == 0:
            print('iter', iter, 'elapsed: ', round(time.time() - start_time, 1), 'seconds')

        if iter % SAVE_SIZE == 0 or iter == n_iters:

            # Saving ----------------------------------------------------------------------------------------------

            # Save traceplots -------------------------------------------------

            np.save('loglik_trace',              loglik_trace)
            np.save('loglik_detail_trace',       loglik_detail_trace)
            np.save('X_trace',                   X_trace)
            np.save('dX_trace',                  dX_trace)
            if UPDATE_Imputation:     np.save('Y_trace', Y_trace)
            if UPDATE_S:              np.save('S_trace_log',               S_trace_log)
            if UPDATE_Z:              np.save('Z_trace',                   Z_trace)
            if UPDATE_phi:            np.save('phi_knots_trace',           phi_knots_trace)
            if UPDATE_rho:            np.save('rho_knots_trace',           rho_knots_trace)
            if UPDATE_tau:            np.save('tau_trace',                 tau_trace)
            if UPDATE_gamma_k:        np.save('gamma_k_vec_trace',         gamma_k_vec_trace)
            if UPDATE_GPD_sigma:      np.save('Beta_logsigma_trace',       Beta_logsigma_trace)
            if UPDATE_GPD_xi:         np.save('Beta_xi_trace',             Beta_xi_trace)
            if UPDATE_Regularization: np.save('sigma_Beta_logsigma_trace', sigma_Beta_logsigma_trace)
            if UPDATE_Regularization: np.save('sigma_Beta_xi_trace',       sigma_Beta_xi_trace)

            # Save Adaptive tuning history ------------------------------------

            if UPDATE_S:              np.save('r_hat_S_history',             r_hat_S_history)
            if UPDATE_Z:              np.save('r_hat_Z_history',             r_hat_Z_history)
            if UPDATE_phi:            np.save('r_hat_phi_history',           r_hat_phi_history)
            if UPDATE_rho:            np.save('r_hat_rho_history',           r_hat_rho_history)
            if UPDATE_tau:            np.save('r_hat_tau_history',           r_hat_tau_history)
            if UPDATE_gamma_k:        np.save('r_hat_gamma_k_history',       r_hat_gamma_k_history)
            if UPDATE_GPD_sigma:      np.save('r_hat_Beta_logsigma_history', r_hat_Beta_logsigma_history)
            if UPDATE_GPD_xi:         np.save('r_hat_Beta_xi_history',       r_hat_Beta_xi_history)
            if UPDATE_Regularization: np.save('r_hat_sigma_Beta_history',    r_hat_sigma_Beta_history)

            with open('iter.pkl', 'wb')               as file: pickle.dump(iter, file)
            if UPDATE_phi or UPDATE_rho or UPDATE_gamma_k or UPDATE_tau or UPDATE_GPD_sigma or UPDATE_GPD_xi or UPDATE_Regularization:
                with open('sigma_m_sq.pkl', 'wb')         as file: pickle.dump(sigma_m_sq, file)
            if UPDATE_Z or UPDATE_phi or UPDATE_rho or UPDATE_GPD_sigma or UPDATE_GPD_xi:
                with open('Sigma_0.pkl', 'wb')            as file: pickle.dump(Sigma_0, file)
            if UPDATE_S:
                with open('sigma_m_sq_St_list.pkl', 'wb') as file: pickle.dump(sigma_m_sq_St_list, file)
            if UPDATE_Z:
                with open('sigma_m_sq_Zt_list.pkl', 'wb') as file: pickle.dump(sigma_m_sq_Zt_list, file)
                with open('Sigma_0_Zt_list.pkl', 'wb')    as file: pickle.dump(Sigma_0_Zt_list, file)

            # Drawing ---------------------------------------------------------------------------------------------

            adapt_idx = (iter // ADAPT_SIZE) - 1
            xs        = np.arange(iter)
            xs_r_hat  = np.arange(adapt_idx + 1)

            # ---- thinning ----
            xs_thin  = xs[0::THIN] # index 1, 11, 21, ...
            xs_thin2 = np.arange(len(xs_thin)) # index 1, 2, 3, ...
            loglik_trace_thin              = loglik_trace[0:iter:THIN,:]
            loglik_detail_trace_thin       = loglik_detail_trace[0:iter:THIN,:]
            if UPDATE_S:               S_trace_log_thin               = S_trace_log[0:iter:THIN,:,:]
            if UPDATE_Z:               Z_trace_thin                   = Z_trace[0:iter:THIN,:,:]
            if UPDATE_phi:             phi_knots_trace_thin           = phi_knots_trace[0:iter:THIN,:]
            if UPDATE_rho:             rho_knots_trace_thin           = rho_knots_trace[0:iter:THIN,:]
            if UPDATE_tau:             tau_trace_thin                 = tau_trace[0:iter:THIN,:]
            if UPDATE_gamma_k:         gamma_k_vec_trace_thin         = gamma_k_vec_trace[0:iter:THIN,:]
            if UPDATE_GPD_sigma:       Beta_logsigma_trace_thin       = Beta_logsigma_trace[0:iter:THIN,:]
            if UPDATE_GPD_xi:          Beta_xi_trace_thin             = Beta_xi_trace[0:iter:THIN,:]
            if UPDATE_Regularization:  sigma_Beta_logsigma_trace_thin = sigma_Beta_logsigma_trace[0:iter:THIN,:]
            if UPDATE_Regularization:  sigma_Beta_xi_trace_thin       = sigma_Beta_xi_trace[0:iter:THIN,:]

            # ---- log-likelihood ----
            plt.subplots()
            plt.plot(xs_thin2, loglik_trace_thin)
            plt.title('traceplot for log-likelihood')
            plt.xlabel('iter thinned by '+str(THIN))
            plt.ylabel('loglikelihood')
            plt.savefig('MCMC:trace_loglik.pdf')
            plt.close()

            # ---- log-likelihood in detail ----
            plt.subplots()
            labels = ['censored_ll', 'exceed_ll', 'S_ll', 'Z_ll']
            for i in range(4):
                plt.plot(xs_thin2, loglik_detail_trace_thin[:,i], label = labels[i])
                plt.annotate(labels[i], xy=(xs_thin2[-1], loglik_detail_trace_thin[:,i][-1]))
            plt.title('traceplot for detail log likelihood')
            plt.xlabel('iter thinned by '+str(THIN))
            plt.ylabel('log likelihood')
            plt.legend(loc = 'upper left')
            plt.savefig('MCMC:trace_detailed_loglik.pdf')
            plt.close()

            # ---- S_t ----
            if UPDATE_S:
                for t in range(Nt):
                    label_by_knot = ['knot ' + str(knot) for knot in range(k_S)]
                    plt.subplots()
                    plt.plot(xs_thin2, S_trace_log_thin[:,:,t], label = label_by_knot)
                    plt.legend(loc = 'upper left')
                    plt.title('traceplot for log(St) at t=' + str(t))
                    plt.xlabel('iter thinned by '+str(THIN))
                    plt.ylabel('log(St)s')
                    plt.savefig('MCMC:trace_St'+str(t)+'.pdf')
                    plt.close()

                    # r_hat
                    plt.subplots()
                    for i in range(k_S):
                        plt.plot(xs_r_hat, r_hat_S_history[:adapt_idx+1,t,i], label='k'+str(i))
                        plt.annotate('k'+str(i), xy=(xs_r_hat[-1], r_hat_S_history[:adapt_idx+1,t,i][-1]))
                    plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                    plt.title(rf'Adaptation $\hat{{r}}$ for $S_{t}$')
                    plt.xlabel('Adapt Step')
                    plt.ylabel('Acceptance Rate')
                    plt.ylim(0, 1)
                    plt.legend(loc='upper left', fontsize='small', ncol=2)
                    plt.tight_layout()
                    plt.savefig(f'MCMC:adapt_r_hat_S_t{t}.pdf')
                    plt.close()

            # ---- Z_t ---- (some randomly selected sites on each plot)
            if UPDATE_Z:
                for t in range(Nt):
                    selection = np.random.choice(np.arange(Ns), size = 10, replace = False)
                    selection_label = np.array(['site ' + str(s) for s in selection])
                    plt.subplots()
                    plt.plot(xs_thin2, Z_trace_thin[:,selection,t], label = selection_label)
                    plt.legend(loc = 'upper left')
                    plt.title('traceplot for Zt at t=' + str(t))
                    plt.xlabel('iter thinned by '+str(THIN))
                    plt.ylabel('Zt')
                    plt.savefig('MCMC:trace_Zt'+str(t)+'.pdf')
                    plt.close()

                    # r_hat
                    plt.subplots()
                    for block in range(r_hat_Z_history.shape[2]):
                        plt.plot(xs_r_hat, r_hat_Z_history[:adapt_idx+1, t, block], label=f'block {block}')
                    plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                    plt.title(rf'Adaptation $\hat{{r}}$ for $Z_{t}$')
                    plt.xlabel('Adapt Step')
                    plt.ylabel('Acceptance Rate')
                    plt.ylim(0, 1)
                    plt.legend(loc='best', fontsize='small', ncol=2)
                    plt.tight_layout()
                    plt.savefig(f'MCMC:adapt_r_hat_Z_t{t}.pdf')
                    plt.close()

            # ---- gamma ----
            if UPDATE_gamma_k:
                plt.subplots()
                for i in range(k_S):
                    plt.plot(xs_thin2, gamma_k_vec_trace_thin[:,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_thin2[-1], gamma_k_vec_trace_thin[:,i][-1]))
                plt.title(r'traceplot for $\gamma_k$')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel('gamma_k_vec')
                plt.legend(loc = 'upper left')
                plt.savefig('MCMC:trace_gamma.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                for i in range(k_S):
                    plt.plot(xs_r_hat, r_hat_gamma_k_history[:adapt_idx+1,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_r_hat[-1], r_hat_gamma_k_history[:adapt_idx+1,i][-1]))
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\gamma_k$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_gamma_k.pdf')
                plt.close()

            # ---- phi ----
            if UPDATE_phi:
                plt.subplots()
                for i in range(k_phi):
                    plt.plot(xs_thin2, phi_knots_trace_thin[:,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_thin2[-1], phi_knots_trace_thin[:,i][-1]))
                plt.title(rf'traceplot for $\phi$')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel(rf'$\phi$')
                plt.legend(loc = 'upper left')
                plt.savefig('MCMC:trace_phi.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                for i in range(len(phi_block_idx_dict.keys())):
                    plt.plot(xs_r_hat, r_hat_phi_history[:adapt_idx+1,i], label='block'+str(i))
                    plt.annotate('block'+str(i), xy=(xs_r_hat[-1], r_hat_phi_history[:adapt_idx+1,i][-1]))
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\phi$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_phi.pdf')
                plt.close()

            # ---- rho ----
            if UPDATE_rho:
                plt.subplots()
                for i in range(k_rho):
                    plt.plot(xs_thin2, rho_knots_trace_thin[:,i], label='k'+str(i))
                    plt.annotate('k'+str(i), xy=(xs_thin2[-1], rho_knots_trace_thin[:,i][-1]))
                plt.title(rf'traceplot for $\rho$')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel(rf'$\rho$')
                plt.legend(loc = 'upper left')
                plt.savefig('MCMC:trace_rho.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                for i in range(len(rho_block_idx_dict.keys())):
                    plt.plot(xs_r_hat, r_hat_rho_history[:adapt_idx+1,i], label='block'+str(i))
                    plt.annotate('block'+str(i), xy=(xs_r_hat[-1], r_hat_rho_history[:adapt_idx+1,i][-1]))
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\rho$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_rho.pdf')
                plt.close()

            # ---- tau ----
            if UPDATE_tau:
                plt.subplots()
                plt.plot(xs_thin2, tau_trace_thin, label = 'nugget std dev')
                plt.title('tau nugget standard deviation')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel('tau')
                plt.legend(loc='upper left')
                plt.savefig('MCMC:trace_tau.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                plt.plot(xs_r_hat, r_hat_tau_history[:adapt_idx+1], label = 'r_hat')
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\tau$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_tau.pdf')
                plt.close()

            # ---- Beta_logsigma ----
            if UPDATE_GPD_sigma:
                plt.subplots()
                plt.plot(xs_thin2, Beta_logsigma_trace_thin)
                plt.title(r'traceplot for $\beta$ $\log (\sigma)$')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel('Beta_logsigma')
                plt.savefig('MCMC:trace_Beta_logsigma.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                plt.plot(xs_r_hat, r_hat_Beta_logsigma_history[:adapt_idx+1], label = 'r_hat')
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\beta \log(\sigma)$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_Beta_logsigma.pdf')
                plt.close()

            # ---- Beta_xi ----
            if UPDATE_GPD_xi:
                plt.subplots()
                plt.plot(xs_thin2, Beta_xi_trace_thin)
                plt.title(r'traceplot for $\beta$ $\xi$')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel('Beta_xi')
                plt.savefig('MCMC:trace_Beta_xi.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                plt.plot(xs_r_hat, r_hat_Beta_xi_history[:adapt_idx+1], label = 'r_hat')
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for $\beta \xi$')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_Beta_xi.pdf')
                plt.close()

            # ---- regularization ----
            if UPDATE_Regularization:
                plt.subplots()
                plt.plot(xs_thin2, sigma_Beta_logsigma_trace_thin, label = 'sigma_Beta_logsigma')
                plt.plot(xs_thin2, sigma_Beta_xi_trace_thin, label = 'sigma_Beta_xi')
                plt.annotate(r'$\sigma (\beta_{\log \sigma})$', xy=(xs_thin2[-1], sigma_Beta_logsigma_trace_thin[:,0][-1]))
                plt.annotate(r'$\sigma (\beta_\xi)$', xy=(xs_thin2[-1], sigma_Beta_xi_trace_thin[:,0][-1]))
                plt.title('traceplot for regularization')
                plt.xlabel('iter thinned by '+str(THIN))
                plt.ylabel('regularization')
                plt.legend(loc = 'upper left')
                plt.savefig('MCMC:trace_regularization.pdf')
                plt.close()

                # r_hat
                plt.subplots()
                plt.plot(xs_r_hat, r_hat_sigma_Beta_history[:adapt_idx+1,:], label = ['r_hat_sigma_Beta_logsigma', 'r_hat_sigma_Beta_xi'])
                plt.axhline(R_OPT, color='red', linestyle='--', label='R_OPT')
                plt.title(rf'Adaptation $\hat{{r}}$ for regularization')
                plt.xlabel('Adapt Step')
                plt.ylabel('Acceptance Rate')
                plt.ylim(0, 1)
                plt.legend(loc='upper left', fontsize='small', ncol=2)
                plt.tight_layout()
                plt.savefig(f'MCMC:adapt_r_hat_regularization.pdf')
                plt.close()

        if iter == n_iters:
            print(iter)
            end_time = time.time()
            print('elapsed: ', round(end_time - start_time, 1), 'seconds')
            print('FINISHED.')