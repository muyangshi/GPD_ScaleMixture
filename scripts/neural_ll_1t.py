"""
Emulate the likelihood function, 
consisting of 
    - the censored Y likelihood
    - the prior likelihood on S
    - the prior likelihood on Z

We ONLY need to emulate the censored_ll and exceed_ll part of the likelihood!

1. Use scipy rbf smoothing splines to perform the emulation
2. use keras+tensorflow neural network to perform the emulation

# ----------------------------------------------------------------------------------------------------------------------------------- #
def ll_1t(Y, p, u_vec, scale_vec, shape_vec,        # marginal model parameters
          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, # dependence model parameters
          logS_vec, gamma_at_knots, censored_idx, exceed_idx):         # auxilury information
    
    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
    dX     = dRW(X, u_vec, scale_vec, shape_vec)
    
    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])

    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

    # log likelihood of Z
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

    return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)
# ----------------------------------------------------------------------------------------------------------------------------------- #
"""

# %% step 0: imports
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import qmc
from rpy2.robjects import r
from multiprocessing import Pool, cpu_count

# r('load("../data/realdata/JJA_precip_nonimputed.RData")')
# Y                  = np.array(r('Y'))
# GP_estimates       = np.array(r('GP_estimates')).T
# logsigma_estimates = GP_estimates[:,1]
# xi_estimates       = GP_estimates[:,2]
# Notes on range of LHS:
#   looked at data Y to get its range
#   looked at previous run and initial site-level estimates to get scale and shape
#   used reverse CDF to span the R


# %% step 1: generate design points with LHS

# (Y, scale, shape, R, Z, phi, gamma_bar, tau)

N = 1000
d = 8

sampler = qmc.LatinHypercube(d, scramble=False, seed=2345)
lhs_samples = sampler.random(N) # Generate samples in [0,1]^d

l_bounds = [40,   5, -1, 0.01, -5.0, 0.05, 0.05, 0.1]
u_bounds = [800, 60,  1, 0.99,  5.0, 0.95,  5.0, 10.0]

X_lhs = qmc.scale(lhs_samples, l_bounds, u_bounds)
X_lhs[:,3] = scipy.stats.levy(loc=0,scale=0.5).ppf(X_lhs[:,3])
Y_samples, scale_samples, shape_samples, R_samples, Z_samples, phi_samples, gamma_bar_samples, tau_samples = X_lhs.T



# %% step 2a: emulate with scipy rbf smoothing splines

# from scipy.interpolate import Rbf  # Radial Basis Function for interpolation
# from scipy.stats import qmc, levy  # For Latin Hypercube Sampling



# %% step 2b: emulate with keras+tensorflow

# import keras
# from keras import layers
# keras.backend.set_floatx('float64')



# %% step 3: plot and benchmark

