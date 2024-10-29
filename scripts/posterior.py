"""
get posterior covariance matrix
"""
# %%
# imports
import sys
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import geopandas as gpd

np.set_printoptions(threshold=sys.maxsize)


# %% 
# load traceplots

traceplotfolder = '../chains/20241027_SgammaZ_simulated_seed-2345_t-60_s-50_phi-nonstatsc2_rho-nonstat_tau-10.0/'

S_trace_log               = np.load(traceplotfolder + 'S_trace_log.npy')
Z_trace                   = np.load(traceplotfolder + 'Z_trace.npy')
gamma_at_knots_trace      = np.load(traceplotfolder + 'gamma_at_knots_trace.npy')
phi_knots_trace           = np.load(traceplotfolder + 'phi_knots_trace.npy')
range_knots_trace         = np.load(traceplotfolder + 'range_knots_trace.npy')
tau_trace                 = np.load(traceplotfolder + 'tau_trace.npy')
Beta_logsigma_trace       = np.load(traceplotfolder + 'Beta_logsigma_trace.npy')
Beta_xi_trace             = np.load(traceplotfolder + 'Beta_xi_trace.npy')
sigma_Beta_logsigma_trace = np.load(traceplotfolder + 'sigma_Beta_logsigma_trace.npy')
sigma_Beta_xi_trace       = np.load(traceplotfolder + 'sigma_Beta_xi_trace.npy')

k_S             = S_trace_log.shape[1]
k_phi           = phi_knots_trace.shape[1]
k_rho           = range_knots_trace.shape[1]
Nt              = S_trace_log.shape[2]
Ns              = Z_trace.shape[1]
Beta_logsigma_m = Beta_logsigma_trace.shape[1]
Beta_xi_m       = Beta_xi_trace.shape[1]

# %%
# burnins
burnin = 0

S_trace_log               = S_trace_log[burnin:]
Z_trace                   = Z_trace[burnin:]
gamma_at_knots_trace      = gamma_at_knots_trace[burnin:]
phi_knots_trace           = phi_knots_trace[burnin:]
range_knots_trace         = range_knots_trace[burnin:]
Beta_logsigma_trace       = Beta_logsigma_trace[burnin:]
Beta_xi_trace             = Beta_xi_trace[burnin:]
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[burnin:]
sigma_Beta_xi_trace       = sigma_Beta_xi_trace[burnin:]


# %%
# remove unfinished cells

S_trace_log               = S_trace_log[~np.isnan(S_trace_log)].reshape((-1,k_S,Nt))
Z_trace                   = Z_trace[~np.isnan(Z_trace)].reshape((-1, Ns, Nt))
gamma_at_knots_trace      = gamma_at_knots_trace[~np.isnan(gamma_at_knots_trace)].reshape((-1,k_S))
phi_knots_trace           = phi_knots_trace[~np.isnan(phi_knots_trace)].reshape((-1,k_phi))
range_knots_trace         = range_knots_trace[~np.isnan(range_knots_trace)].reshape((-1,k_rho))
Beta_logsigma_trace       = Beta_logsigma_trace[~np.isnan(Beta_logsigma_trace)].reshape((-1,Beta_logsigma_m))
Beta_xi_trace             = Beta_xi_trace[~np.isnan(Beta_xi_trace)].reshape((-1,Beta_xi_m))
sigma_Beta_logsigma_trace = sigma_Beta_logsigma_trace[~np.isnan(sigma_Beta_logsigma_trace)].reshape((-1,1))
sigma_Beta_xi_trace       = sigma_Beta_xi_trace[~np.isnan(sigma_Beta_xi_trace)].reshape((-1,1))


# %%
# posterior covariance matrix
S_log_cov         = np.full(shape=(k_S,k_S,S_trace_log.shape[2]), fill_value = np.nan)
for t in range(S_trace_log.shape[2]):
    S_log_cov[:,:,t] = np.cov(S_trace_log[:,:,t].T)
Z_cov             = np.full(shape = (Ns, Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    Z_cov[:,:,t] = np.cov(Z_trace[:,:,t].T)
gamma_at_knots_cov = np.cov(gamma_at_knots_trace.T)
phi_cov           = np.cov(phi_knots_trace.T)
range_cov         = np.cov(range_knots_trace.T)
Beta_logsigma_cov = np.cov(Beta_logsigma_trace.T)
Beta_xi_cov      = np.cov(Beta_xi_trace.T)
sigma_Beta_logsigma_cov = np.cov(sigma_Beta_logsigma_trace.T)
sigma_Beta_xi_cov = np.cov(sigma_Beta_xi_trace.T)


# %%
# posterior mean

S_log_mean               = np.full(shape=(k_S,S_trace_log.shape[2]), fill_value = np.nan)
for t in range(S_trace_log.shape[2]):
    S_log_mean[:,t] = np.mean(S_trace_log[:,:,t], axis = 0)
Z_mean                   = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    Z_mean[:,t] = np.mean(Z_trace[:,:,t], axis = 0)
gamma_at_knots_mean      = np.mean(gamma_at_knots_trace, axis = 0)
phi_mean                 = np.mean(phi_knots_trace, axis = 0)
range_mean               = np.mean(range_knots_trace, axis = 0)
Beta_logsigma_mean       = np.mean(Beta_logsigma_trace, axis = 0)
Beta_xi_mean            = np.mean(Beta_xi_trace, axis = 0)
sigma_Beta_logsigma_mean = np.mean(sigma_Beta_logsigma_trace, axis = 0)
sigma_Beta_xi_mean      = np.mean(sigma_Beta_xi_trace, axis = 0)


# %%
# posterior median
S_log_median               = np.full(shape=(k_S,S_trace_log.shape[2]), fill_value = np.nan)
for t in range(S_trace_log.shape[2]):
    S_log_median[:,t] = np.median(S_trace_log[:,:,t], axis = 0)
Z_median                   = np.full(shape = (Ns, Nt), fill_value = np.nan)
for t in range(Nt):
    Z_median[:,t] = np.median(Z_trace[:,:,t], axis = 0)
gamma_at_knots_median      = np.median(gamma_at_knots_trace, axis = 0)
phi_median                 = np.median(phi_knots_trace, axis = 0)
range_median               = np.median(range_knots_trace, axis = 0)
Beta_logsigma_median       = np.median(Beta_logsigma_trace, axis = 0)
Beta_xi_median            = np.median(Beta_xi_trace, axis = 0)
sigma_Beta_logsigma_median = np.median(sigma_Beta_logsigma_trace, axis = 0)
sigma_Beta_xi_median      = np.median(sigma_Beta_xi_trace, axis = 0)
