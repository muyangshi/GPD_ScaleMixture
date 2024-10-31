"""
March 26, 2024
Simulate data using the GPD Scale Mixture Model

May 11
redirect output to ../data/<savefolder>

20240909
Decouple phi and rho knots

20241031
output placed in the current folder, easier to use for coverage analysis
"""

# %%
# imports

# base python -------------------------------------------------------------

import sys
import os
import multiprocessing
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"        # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"        # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"    # export NUMEXPR_NUM_THREADS=1

# packages ----------------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import gstools as gs
import rpy2.robjects as robjects
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

# custom module -----------------------------------------------------------

from utilities import *
print('link function:', norm_pareto, 'Pareto')


# setup -------------------------------------------------------------------

n_processes = 4

try:
    data_seed = int(sys.argv[1])
    data_seed
except:
    data_seed = 2345
finally:
    print('data_seed:', data_seed)

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)



# %%
# Simulate setup

# Configurations --------------------------------------------------------   

np.random.seed(data_seed)

Nt       = 60 # number of time replicates
Ns       = 50 # number of sites/stations
Time     = np.linspace(-Nt/2, Nt/2-1, Nt)/np.std(np.linspace(-Nt/2, Nt/2-1, Nt), ddof=1)

# Knots

N_outer_grid     = 9 # for S and phi
N_outer_grid_rho = 9 # for rho
radius           = 3 # radius of R's Wendland Kernel for R
eff_range        = 3 # range where phi's gaussian kernel drops to 0.05
eff_range_rho    = 3 # range where rho's gaussian kernel drops to 0.05

# gamma

gamma_at_knots = np.repeat(0.5, int(N_outer_grid + 
                                    (np.sqrt(N_outer_grid) - 1)**2))

# phi

scenario_phi = 'nonstatsc2'  # nonstatsc1, nonstatsc2, nonstatsc3, stat_AI, stat_AD

# rho

scenario_rho = 'nonstat' # nonstat, stats

# tau

tau = 10.0

# threshold u

simulation_threshold = 60.0

# sigma

Beta_logsigma = np.array([3.0, 0.0])

# xi

Beta_ksi      = np.array([0.1, 0.0])

# censoring probability

p = 0.9

# save

# savefolder = '../data/simulated' + \
#                         '_seed-'  + str(data_seed) + \
#                         '_t-'     + str(Nt) + \
#                         '_s-'     + str(Ns) + \
#                         '_phi-'   + scenario_phi + \
#                         '_rho-'   + scenario_rho + \
#                         '_tau-'   + str(tau)
savefolder = './simulated' + \
                '_seed-'  + str(data_seed) + \
                '_t-'     + str(Nt) + \
                '_s-'     + str(Ns) + \
                '_phi-'   + scenario_phi + \
                '_rho-'   + scenario_rho + \
                '_tau-'   + str(tau)
Path(savefolder).mkdir(parents=True, exist_ok=True)

# missing indicator matrix ------------------------------------------------

miss_proportion = 0.0
miss_matrix = np.full(shape = (Ns, Nt), fill_value = 0)
for t in range(Nt):
    miss_matrix[:,t] = np.random.choice([0, 1], size = (Ns,), p = [1-miss_proportion, miss_proportion])
miss_matrix = miss_matrix.astype(bool) # matrix of True/False indicating missing, True means missing

# Sites - random unifromly (x,y) generate site locations ------------------

sites_xy = np.random.random((Ns, 2)) * 10
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# Elevation Function ------------------------------------------------------

elev_surf_generator = gs.SRF(gs.Gaussian(dim=2, var = 1, len_scale = 2),seed=data_seed)
elevations = elev_surf_generator((sites_x, sites_y))
elevations += np.abs(np.min(elevations))


# Knots - isometric grid ----------------------------------------------------------------------

# define the lower and upper limits for x and y

# minX, maxX = np.floor(np.min(sites_x)), np.ceil(np.max(sites_x))
# minY, maxY = np.floor(np.min(sites_y)), np.ceil(np.max(sites_y))

minX, maxX = 0.0, 10.0
minY, maxY = 0.0, 10.0

# isometric knot grid -- for R^phi

h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
                                        num = int(2*np.sqrt(N_outer_grid)))
y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
                                        num = int(2*np.sqrt(N_outer_grid)))
x_outer_pos              = x_pos[0::2]
x_inner_pos              = x_pos[1::2]
y_outer_pos              = y_pos[0::2]
y_inner_pos              = y_pos[1::2]
X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
knots_xy                 = knots_xy[knots_id_in_domain]
knots_x                  = knots_xy[:,0]
knots_y                  = knots_xy[:,1]
k                        = len(knots_id_in_domain)

# isometric knot grid -- for rho

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

# Copula Splines ----------------------------------------------------------

# Influence Radius from knots:

# R <- weighted sum of Stable S
radius_from_knots = np.repeat(radius, k)

# phi
bandwidth = eff_range**2/6

# rho
bandwidth_rho = eff_range_rho**2/6

# Weight matrices (matrix expand from knots to sites):

# R, Wendland Kernel

wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix[site_id, :] = weight_from_knots

# phi, Gaussian Kernel

gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix[site_id, :] = weight_from_knots

# rho, Gaussian Kernel

gaussian_weight_matrix_rho = np.full(shape = (Ns, k_rho), fill_value = np.nan)
for site_id in np.arange(Ns):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_rho[site_id, :] = weight_from_knots    


# Marginal Model - GP(sigma, ksi) threshold u ---------------------------------------------------------------------

"""
We no longer need a spline fit surface for mu or any of the like, right?
Because ut(s), the threshold for a site s at time t, will be empiricially
estimated from the actual data.
"""

# Threshold u(t,s) --------------------------------------------------------
u_matrix = np.full(shape = (Ns, Nt), fill_value = simulation_threshold)

# Scale logsigma(s) -------------------------------------------------------

Beta_logsigma_m   = 2 # just intercept and elevation
C_logsigma        = np.full(shape = (Beta_logsigma_m, Ns, Nt), fill_value = np.nan)
C_logsigma[0,:,:] = 1.0 
C_logsigma[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T

# Shape ksi(s) ------------------------------------------------------------

Beta_ksi_m   = 2 # just intercept and elevation
C_ksi        = np.full(shape = (Beta_ksi_m, Ns, Nt), fill_value = np.nan) # ksi design matrix
C_ksi[0,:,:] = 1.0
C_ksi[1,:,:] = np.tile(elevations, reps = (Nt, 1)).T


# Setup For the Copula/Data Model - X = e + X_star = R^phi * g(Z) -------------------------------------------------


# Covariance K for Gaussian Field g(Z)

nu = 0.5 # exponential kernel for matern with nu = 1/2
sigsq = 1.0 # sill for Z
sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

# Scale Mixture R^phi

delta = 0.0 # this is the delta in levy, stays 0
alpha = 0.5

gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                    axis = 1)**(1/alpha) # gamma_bar, axis = 1 to sum over K knots

# -----------------------------------------------------------------------------------------------------------------
# Parameter Configuration For the Model 
# -----------------------------------------------------------------------------------------------------------------

# Censoring probability

# p = 0.9

# Marginal Parameters - GP(sigma, ksi) ----------------------------------------------------------------------------


# Data Model Parameters - X_star = R^phi * g(Z) -------------------------------------------------------------------

# phi
match scenario_phi:
    case 'nonstatsc1': phi_at_knots = 0.65 - np.sqrt((knots_x-3)**2/4 + (knots_y-3)**2/3)/10
    case 'nonstatsc2': phi_at_knots = 0.65 - np.sqrt((knots_x-5.1)**2/5 + (knots_y-5.3)**2/4)/11.6
    case 'nonstatsc3': phi_at_knots = 0.37 + 5*(scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([2.5,3]), cov = 2*np.matrix([[1,0.2],[0.2,1]])) + 
                                            scipy.stats.multivariate_normal.pdf(knots_xy, mean = np.array([7,7.5]), cov = 2*np.matrix([[1,-0.2],[-0.2,1]])))
    case 'stat_AI'   : phi_at_knots = [0.3] * k
    case 'stat_AD'   : phi_at_knots = [0.7] * k
# rho
match scenario_rho:
    case 'nonstat': rho_at_knots = np.sqrt(0.3*knots_x + 0.4*knots_y)/2 # range for spatial Matern Z
    case 'stat'   : rho_at_knots = [1.5] * k_rho                        # for a stationary rho(s)


# %%
# Generate Dataset

np.random.seed(data_seed)

# Transformed Gaussian Process --------------------------------------------

range_vec = gaussian_weight_matrix_rho @ rho_at_knots
K         = ns_cov(range_vec = range_vec, 
                    coords    = sites_xy,
                    sigsq_vec = sigsq_vec,
                    kappa     = nu, cov_model = "matern")
Z         = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),
                                                cov=K,
                                                size=Nt).T
W         = g(Z) 

# Random Scaling Factor ---------------------------------------------------

S_at_knots = np.full(shape = (k, Nt), fill_value = np.nan)
R_phi      = np.full(shape = (Ns, Nt), fill_value = np.nan)
phi_vec    = gaussian_weight_matrix @ phi_at_knots
for t in np.arange(Nt):
    S_at_knots[:,t] = rlevy(n = k, m = delta, s = gamma_at_knots) # generate R at time t, spatially varying k knots
R_at_sites = wendland_weight_matrix @ S_at_knots
for t in np.arange(Nt):
    R_phi[:,t] = np.power(R_at_sites[:,t], phi_vec)

# Nuggets -----------------------------------------------------------------

nuggets = scipy.stats.multivariate_normal.rvs(mean = np.zeros(shape = (Ns,)),
                                                cov  = tau**2,
                                                size = Nt).T

# Hidden Data X -----------------------------------------------------------

X_star       = R_phi * W
X            = X_star + nuggets

# Marginal Observation Y --------------------------------------------------

sigma_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
ksi_matrix   = (C_ksi.T @ Beta_ksi).T

# # single core version
# pX           = np.full(shape = (Ns, Nt), fill_value = np.nan)
# Y            = np.full(shape = (Ns, Nt), fill_value = np.nan)
# for t in np.arange(Nt):
#     # CDF of the generated X
#     pX[:,t] = pRW(X[:,t], phi_vec, gamma_vec, tau)
#     censored_idx = np.where(pX[:,t] <= p)[0]
#     exceed_idx   = np.where(pX[:,t] > p)[0]

#     # censored below
#     Y[censored_idx,t]  = u_matrix[censored_idx,t]

#     # threshold exceeded
#     Y[exceed_idx,t]  = qCGP(pX[exceed_idx, t], p, 
#                             u_matrix[exceed_idx, t], 
#                             sigma_matrix[exceed_idx, t],
#                             ksi_matrix[exceed_idx, t])

# multi core version
def transform_to_Y(args):
    X_vec, phi_vec, gamma_vec, tau, p, u_vec, sigma_vec, ksi_vec = args

    Y_vec      = np.full(shape = (Ns,), fill_value = np.nan)
    pX_vec     = pRW(X_vec, phi_vec, gamma_vec, tau)

    censored_idx        = np.where(pX_vec <= p)[0]
    Y_vec[censored_idx] = u_vec[censored_idx]

    exceed_idx          = np.where(pX_vec > p)[0]
    Y_vec[exceed_idx]   = qCGP(pX_vec[exceed_idx],
                            p, 
                            u_vec[exceed_idx],
                            sigma_vec[exceed_idx],
                            ksi_vec[exceed_idx])
    
    return Y_vec

args_list = []
for t in range(Nt):
    args = (X[:,t], phi_vec, gamma_vec, tau,
            p, u_matrix[:,t], sigma_matrix[:,t], ksi_matrix[:,t])
    args_list.append(args)
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    results = pool.map(transform_to_Y, args_list)

Y_noNA = np.column_stack(results)
Y_wNA   = np.where(miss_matrix, np.nan, Y_noNA)


# %%
# Save Simulated Dataset

# .npy files

np.save(savefolder+'/miss_matrix_bool', miss_matrix)
np.save(savefolder+'/stations',         sites_xy)
np.save(savefolder+'/elev',             elevations)
np.save(savefolder+'/u_matrix',         u_matrix)
np.save(savefolder+'/logsigma_matrix',  np.log(sigma_matrix))
np.save(savefolder+'/ksi_matrix',       ksi_matrix)
np.save(savefolder+'/Y_noNA',           Y_noNA)
np.save(savefolder+'/Y',                Y_wNA)
    
# .RData file

Y_ro = numpy2rpy(Y_wNA)
GP_estimates    = np.column_stack((u_matrix[:,0],
                                    (C_logsigma.T @ Beta_logsigma).T[:,0],
                                    (C_ksi.T @ Beta_ksi).T[:,0]))
GP_estimates_ro = numpy2rpy(GP_estimates)
elev_ro         = numpy2rpy(elevations)
stations_ro     = numpy2rpy(sites_xy)

r.assign('Y', Y_ro)
r.assign('GP_estimates', GP_estimates_ro)
r.assign('elev', elev_ro)
r.assign('stations', stations_ro)

r(f'''
    GP_estimates <- as.data.frame(GP_estimates)
    colnames(GP_estimates) <- c('mu','logsigma','xi')

    stations <- as.data.frame(stations)
    colnames(stations) <- c('x','y')

    elev <- c(elev)

    save(Y, GP_estimates, elev, stations,
        file = '{savefolder}/simulated_data.RData')
''')


# %%
# Check Data Generation

# checking stable variables S ---------------------------------------------

# levy.cdf(S_at_knots[i,:], loc = 0, scale = gamma_at_knots[i]) should look uniform
for i in range(k):
    scipy.stats.probplot(scipy.stats.levy.cdf(S_at_knots[i,:], scale = gamma_at_knots[i]), dist='uniform', fit=False, plot=plt)
    plt.axline((0,0), slope = 1, color = 'black')
    plt.title(f'QQPlot_Stable_knot_{i}')
    plt.savefig(savefolder + '/DataGeneration:QQPlot_Stable_knot_{}.pdf'.format(i))
    plt.show()
    plt.close()
    
# checking Pareto distribution --------------------------------------------

for site_i in range(Ns):
    if site_i % 50 == 0:
        if norm_pareto == 'standard': scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:], b = 1, loc = 0, scale = 1), dist = 'uniform', fit = False, plot=plt)
        if norm_pareto == 'shifted':  scipy.stats.probplot(scipy.stats.pareto.cdf(W[site_i,:]+1, b = 1, loc = 0, scale = 1), dist='uniform', fit=False, plot=plt)
        plt.axline((0,0), slope = 1, color = 'black')
        plt.title(f'QQPlot_Pareto_site_{site_i}')
        plt.savefig(savefolder + '/DataGeneration:QQPlot_Pareto_site_{}.pdf'.format(site_i))
        plt.show()
        plt.close()

# checking model X_star ---------------------------------------------------

for site_i in range(Ns):
    if site_i % 50 == 0:
        unif = RW_inte.pRW_standard_Pareto_vec(X_star[site_i,:], phi_vec[site_i], gamma_vec[site_i])
        scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
        plt.axline((0,0), slope=1, color='black')
        plt.title(f'QQPlot_Xstar_site_{site_i}')
        plt.savefig(savefolder + '/DataGeneration:QQPlot_Xstar_site_{}.pdf'.format(site_i))
        plt.show()
        plt.close()

# checking model X --------------------------------------------------------

for site_i in range(Ns):
    if site_i % 50 == 0:
        unif = pRW(X[site_i,:], phi_vec[site_i], gamma_vec[site_i], tau)
        scipy.stats.probplot(unif, dist="uniform", fit = False, plot=plt)
        plt.axline((0,0), slope=1, color='black')
        plt.title(f'QQPlot_X_site_{site_i}')
        plt.savefig(savefolder + '/DataGeneration:QQPlot_X_site_{}.pdf'.format(site_i))
        plt.show()
        plt.close()

# checking marginal exceedance --------------------------------------------

## all pooled together
pY = np.array([])
for t in range(Nt):
    exceed_idx = np.where(Y_wNA[:,t] > u_matrix[:,t])[0]
    pY = np.append(pY, pGP(Y_wNA[exceed_idx,t],u_matrix[exceed_idx,t],sigma_matrix[exceed_idx,t],ksi_matrix[exceed_idx,t]))
nquant = len(pY)
emp_p = np.linspace(1/nquant, 1-1/nquant, num=nquant)
emp_q = scipy.stats.uniform().ppf(emp_p)
plt.plot(emp_q, np.sort(pY),
            c='blue',marker='o',linestyle='None')
plt.xlabel('Uniform')
plt.ylabel('Observed')
plt.axline((0,0), slope = 1, color = 'black')
plt.xlim((0,1))
plt.ylim((0,1))
plt.title('Generalized Pareto CDF of all exceedance')
plt.savefig(savefolder+'/DataGeneration:QQPlot_Yexceed_all.pdf')
plt.show()
plt.close()

# %% 
# Plot Generated Surfaces 

# 0. Grids for plots
plotgrid_res_x         = 150
plotgrid_res_y         = 175
plotgrid_res_xy        = plotgrid_res_x * plotgrid_res_y
plotgrid_x             = np.linspace(minX,maxX,plotgrid_res_x)
plotgrid_y             = np.linspace(minY,maxY,plotgrid_res_y)
plotgrid_X, plotgrid_Y = np.meshgrid(plotgrid_x, plotgrid_y)
plotgrid_xy            = np.vstack([plotgrid_X.ravel(), plotgrid_Y.ravel()]).T

wendland_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy,k), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
    wendland_weight_matrix_for_plot[site_id, :] = weight_from_knots

gaussian_weight_matrix_for_plot = np.full(shape = (plotgrid_res_xy, k), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                    XB = knots_xy)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
    gaussian_weight_matrix_for_plot[site_id, :] = weight_from_knots

gaussian_weight_matrix_rho_for_plot = np.full(shape = (plotgrid_res_xy, k_rho), fill_value = np.nan)
for site_id in np.arange(plotgrid_res_xy):
    # Compute distance between each pair of the two collections of inputs
    d_from_knots = scipy.spatial.distance.cdist(XA = plotgrid_xy[site_id,:].reshape((-1,2)), 
                                                XB = knots_xy_rho)
    # influence coming from each of the knots
    weight_from_knots = weights_fun(d_from_knots, radius, bandwidth_rho, cutoff = False)
    gaussian_weight_matrix_rho_for_plot[site_id, :] = weight_from_knots

# 1. Station, Knots 

fig, ax = plt.subplots()
fig.set_size_inches(10,8)
ax.set_aspect('equal', 'box')
for i in range(k):
    circle_i = plt.Circle((knots_xy[i,0], knots_xy[i,1]), radius_from_knots[i],
                            color='r', fill=True, fc='lightgrey', ec='grey', alpha = 0.2)
    ax.add_patch(circle_i)
ax.scatter(sites_x, sites_y, marker = '.', c = 'blue', label='sites')
ax.scatter(knots_x, knots_y, marker = '+', c = 'red', label = 'knot', s = 300)
space_rectangle = plt.Rectangle(xy=(minX, minY), width=maxX-minX, height=maxY-minY,
                                fill = False, color = 'black')
ax.add_patch(space_rectangle)
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)    
box = ax.get_position()
legend_elements = [matplotlib.lines.Line2D([0], [0], marker= '.', linestyle='None', color='b', label='Site'),
                matplotlib.lines.Line2D([0], [0], marker='+', linestyle = "None", color='red', label='Knot Center',  markersize=20),
                matplotlib.lines.Line2D([0], [0], marker = 'o', linestyle = 'None', label = 'Knot Radius', markerfacecolor = 'grey', markersize = 20, alpha = 0.2),
                matplotlib.lines.Line2D([], [], color='None', marker='s', linestyle='None', markeredgecolor = 'black', markersize=20, label='Spatial Domain')]
plt.legend(handles = legend_elements, bbox_to_anchor=(1.01,1.01), fontsize = 20)
# plt.subplots_adjust(right=0.6)
plt.savefig(savefolder+'/DataGeneration:stations.pdf',bbox_inches="tight")
plt.show()
plt.close()

# 2. Elevation

fig, ax = plt.subplots()
elev_scatter = ax.scatter(sites_x, sites_y, s=10, c = elevations,
                            cmap = 'RdBu_r')
ax.set_aspect('equal', 'box')
plt.colorbar(elev_scatter)
plt.savefig(savefolder+'/DataGeneration:station_elevation.pdf')
plt.show()
plt.close()       


# 3. phi surface

phi_vec_for_plot = (gaussian_weight_matrix_for_plot @ phi_at_knots).round(3)
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
heatmap = ax.imshow(phi_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x), 
                    vmin = 0.0, vmax = 1.0,
                    cmap ='seismic', interpolation='nearest', 
                    origin = 'lower',extent = [minX, maxX, minY, maxY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
# ax.invert_yaxis()
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
# Plot knots and circles
for i in range(k):
    circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                        color='r', fill=False, fc='None', ec='lightgrey', alpha=0.5)
    ax.add_patch(circle_i)
# Scatter plot for sites and knots
ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
for index, (x, y) in enumerate(knots_xy):
    ax.text(x+0.05, y+0.1, f'{index+1}', fontsize=12, ha='left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('x', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.title(r'True $\phi$ surface', fontsize = 20)
plt.savefig(savefolder + '/DataGeneration:true phi surface.pdf', bbox_inches='tight')
plt.show()
plt.close()



# 4. Plot range surface

range_vec_for_plot = gaussian_weight_matrix_rho_for_plot @ rho_at_knots
vmin = 0.0
vmax = np.ceil(max(range_vec_for_plot))
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(range_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                    vmin = vmin, vmax = vmax, 
                    cmap ='Reds', interpolation='nearest', 
                    origin = 'lower',extent = [minX, maxX, minY, maxY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
# ax.invert_yaxis()
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
# Plot knots and circles
for i in range(k_rho):
    circle_i = plt.Circle((knots_xy_rho[i, 0], knots_xy_rho[i, 1]), radius_from_knots[i],
                        color='r', fill=False, fc='None', ec='lightgrey', alpha=0.5)
    ax.add_patch(circle_i)
# Scatter plot for sites and knots
ax.scatter(knots_x_rho, knots_y_rho, marker='+', c='white', label='knot', s=300)
for index, (x, y) in enumerate(knots_xy_rho):
    ax.text(x+0.05, y+0.1, f'{index+1}', fontsize=12, ha='left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('x', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.title(r'True $\rho$ surface', fontsize = 20)
plt.savefig(savefolder+'/DataGeneration:true rho surface.pdf', bbox_inches='tight')
plt.show()
plt.close()

# 5. Plot gamma_bar surface?

gamma_vec_for_plot = np.sum(np.multiply(wendland_weight_matrix_for_plot, gamma_at_knots)**(alpha), 
                    axis = 1)**(1/alpha)
vmin = 0.0
vmax = np.ceil(max(gamma_vec_for_plot))
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
ax.set_aspect('equal', 'box')
# state_map.boundary.plot(ax=ax, color = 'black')
heatmap = ax.imshow(gamma_vec_for_plot.reshape(plotgrid_res_y,plotgrid_res_x),
                    vmin = vmin, vmax = vmax, 
                    cmap ='Reds', interpolation='nearest', 
                    origin = 'lower', extent = [minX, maxX, minY, maxY])
ax.set_xticks(np.linspace(minX, maxX,num=3))
ax.set_yticks(np.linspace(minY, maxY,num=5))
# ax.invert_yaxis()
cbar = fig.colorbar(heatmap, ax=ax)
cbar.ax.tick_params(labelsize=20)  # Set the fontsize here
# Plot knots and circles
for i in range(k):
    circle_i = plt.Circle((knots_xy[i, 0], knots_xy[i, 1]), radius_from_knots[i],
                        color='r', fill=False, fc='None', ec='lightgrey', alpha=0.5)
    ax.add_patch(circle_i)
# Scatter plot for sites and knots
ax.scatter(knots_x, knots_y, marker='+', c='white', label='knot', s=300)
for index, (x, y) in enumerate(knots_xy):
    ax.text(x+0.05, y+0.1, f'{index+1}', fontsize=12, ha='left')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('x', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.title(r'True $\bar{\gamma}$ surface', fontsize = 20)
plt.savefig(savefolder+'/DataGeneration:true gamma_vec surface.pdf', bbox_inches='tight')
plt.show()
plt.close()
# %%

import shutil
current_file = __file__
shutil.copy(current_file, savefolder)