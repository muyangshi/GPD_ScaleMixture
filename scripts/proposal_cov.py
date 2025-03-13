"""
This file defines: 
  Proposal scalr variance for sigma_m:
    - sigma_Betas (regularizaiton variance)
    - S_log_cov, Z_cov are covariance matrices, but will take diagonal as sigma_m
  Proposal Covariance Matrix used to initialize the Sigma_0:
    - phi_cov
    - range_cov
    - Beta_marginal_cov

Note: 
  Only used when starting the chain fresh, as the laster daisychains
  will load the proposal scalar variance and cov from pickle files
"""
# %%
import numpy as np
import os

file_names = [
    # covariance matrices
    'S_log_cov', 
    'Z_cov', 
    'gamma_k_cov', 
    'phi_cov', 
    'range_cov', 
    'Beta_logsigma_cov', 
    'Beta_xi_cov',

    # MUST BE SCALAR VALUES!!!
    #  e.g. np.array(5) is not okay, 
    #  we need np.array(5).item() to get the value
    'tau_cov',
    'sigma_Beta_logsigma_cov', 
    'sigma_Beta_xi_cov'
]

loaded_data = {}

for name in file_names:
    file_path = f"{name}.npy"
    if os.path.exists(file_path):
        loaded_data[name] = np.load(file_path)
        if np.any(np.isnan(loaded_data[name])) or np.any(loaded_data[name] == 0):
            loaded_data[name] = None
    else:
        loaded_data[name] = None

S_log_cov               = loaded_data['S_log_cov']
Z_cov                   = loaded_data['Z_cov']
gamma_k_cov             = loaded_data['gamma_k_cov']
phi_cov                 = loaded_data['phi_cov']
range_cov               = loaded_data['range_cov']
Beta_logsigma_cov       = loaded_data['Beta_logsigma_cov']
Beta_xi_cov             = loaded_data['Beta_xi_cov']
tau_var                 = loaded_data['tau_cov'].item()
sigma_Beta_logsigma_var = loaded_data['sigma_Beta_logsigma_cov'].item()
sigma_Beta_xi_var       = loaded_data['sigma_Beta_xi_cov'].item()
# %%
