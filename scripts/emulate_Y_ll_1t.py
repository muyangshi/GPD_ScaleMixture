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
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)
    
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

# base python

import sys
import os
import time
import multiprocessing
import datetime

from multiprocessing import Pool, cpu_count
from time            import strftime, localtime
from pathlib         import Path

# os.environ["OMP_NUM_THREADS"]        = "1" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"]   = "1" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"]        = "1" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"]    = "1" # export NUMEXPR_NUM_THREADS=1

# packages
import scipy
import scipy.stats
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import gstools           as gs
import geopandas         as gpd
import rpy2.robjects     as robjects
import keras

from keras                  import layers
from keras                  import ops
from scipy.stats            import qmc
from rpy2.robjects          import r
from scipy.interpolate      import RBFInterpolator
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr

# custom modules
from utilities              import *

keras.backend.set_floatx('float64')

print('link function:', norm_pareto, 'Pareto')

random_generator = np.random.RandomState(7)

n_processes = 7 if cpu_count() < 64 else 64

INITIAL_EPOCH = 0
EPOCH         = 100

N             = int(1e8)
N_val         = int(1e6)
d             = 9

# define some helper functions

# censoring log-likelihood on Y

def Y_ll_1t1s(Y, u_vec, scale_vec, shape_vec,
              R_vec, Z_vec, phi_vec, gamma_bar_vec, tau):
    
    p = 0.9

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)

    if Y <= u_vec:
        # log censored likelihood of y on censored sites
        censored_ll = scipy.stats.norm.logcdf((X - X_star)/tau)
        return censored_ll
    else: # if Y > u_vec
        # log censored likelihood of y on exceedance sites
        exceed_ll   = scipy.stats.norm.logpdf(X, loc = X_star, scale = tau) \
                        + np.log(dCGP(Y, p, u_vec, scale_vec, shape_vec)) \
                        - np.log(dX)
        return exceed_ll

def Y_ll_1t1s_par(params):
    """
    calculate the censoring likelihood of Y, at p = 0.9
    This will compute the site-wise log likelihood at one time
    Returning results for one site at one time
    """
    
    p = 0.9

    Y, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, phi_vec, gamma_bar_vec, tau = params

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)

    # if dX < 0: return np.nan

    if Y <= u_vec:
        # log censored likelihood of y on censored sites
        censored_ll = scipy.stats.norm.logcdf((X - X_star)/tau)
        return censored_ll
    else: # if Y > u_vec
        # log censored likelihood of y on exceedance sites
        exceed_ll   = scipy.stats.norm.logpdf(X, loc = X_star, scale = tau) \
                        + np.log(dCGP(Y, p, u_vec, scale_vec, shape_vec)) \
                        - np.log(dX)
        # if np.isnan(exceed_ll):
        #     print(params)
        return exceed_ll
    # return np.sum(censored_ll) + np.sum(exceed_ll)

# Full log-likelihood (Y + S + Z)

# def ll_1t(Y, p, u_vec, scale_vec, shape_vec, \
#           R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
#           logS_vec, gamma_at_knots, censored_idx, exceed_idx):

#     """
#     This will compute the sum of the log like for all sites at 1t
#     The inputs are expected to be vectors
#     """

#     X_star = (R_vec ** phi_vec) * g(Z_vec)
#     X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
#     dX     = dRW(X, phi_vec, gamma_bar_vec, tau)
    
#     # log censored likelihood of y on censored sites
#     censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
#     # log censored likelihood of y on exceedance sites
#     exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
#                     + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
#                     - np.log(dX[exceed_idx])

#     # log likelihood of S
#     S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

#     # log likelihood of Z
#     Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

#     return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)

def ll_1t_par(args): # For use with multiprocessing

    """
    This will compute the sum of the log like for all sites at 1t
    The inputs are expected to be vectors
    """

    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx = args

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_bar_vec, tau)
    dX     = dRW(X, phi_vec, gamma_bar_vec, tau)
    
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


# %% step 1: generate design points with LHS

# # r('load("../data/realdata/JJA_precip_nonimputed.RData")')
# # Y                  = np.array(r('Y'))
# # GP_estimates       = np.array(r('GP_estimates')).T
# # u_vec              = GP_estimates[:,0]
# # logsigma_estimates = GP_estimates[:,1]
# # xi_estimates       = GP_estimates[:,2]
# # Notes on range of LHS:
# #   looked at data Y to get its range
# #   looked at previous run and initial site-level estimates to get scale and shape
# #   used reverse CDF to span the R <- R is NOT levy(0, 0.5)!

# # LHS for the design point X

# sampler     = qmc.LatinHypercube(d, scramble=False, seed=2345)
# lhs_samples = sampler.random(N) # Generate LHS samples in [0,1]^d
# lhs_samples = np.row_stack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary points

# #             pY,    u, scale, shape,    pR,    Z,  phi, gamma_bar,  tau
# l_bounds = [0.001,   30,     5,  -1.0, 1e-2, -5.0, 0.05,       0.5,  1.0]
# u_bounds = [0.999,   80,    60,   1.0,  5e6,  5.0, 0.95,       8.0, 50.0]
# X_lhs    = qmc.scale(lhs_samples, l_bounds, u_bounds)        # scale LHS to specified bounds

# # Note that R is not levy(0, 0.5)
# #   Check the values of gamma_bar, pick the largest, use that to span the sample space
# #   Maybe (0, 8)?
# # X_lhs[:,4] = scipy.stats.levy(loc=0,scale=8.0).ppf(X_lhs[:,4]) # scale the Stables

# # Y assumed to be Generalized Pareto, if pY > 0.9;
# #   otherwise, just return the corresponding threshold u
# X_lhs[:,0] = qCGP(X_lhs[:,0], 0.9, X_lhs[:,1], X_lhs[:,2], X_lhs[:,3])

# print('X_lhs.shape:',X_lhs.shape)

# # Calculate the likelihoods of Y at the design points

# data = [tuple(row) for row in X_lhs]

# start_time = time.time()
# print(rf'start calculating {N} likelihoods using {n_processes} processes:', datetime.datetime.now())
# with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
#     results = pool.map(Y_ll_1t1s_par, data)
# end_time = time.time()
# print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

# # remove the NAs
# noNA = np.where(~np.isnan(results))
# Y_lhs = np.array(results)[noNA]
# X_lhs = X_lhs[noNA]

# print('len(Y_lhs):',len(Y_lhs))   # number of design points retained
# print('proportion not NA:', len(Y_lhs)/N) # proportion of design points retained

# np.save(rf'll_1t_X_{N}.npy', X_lhs)
# np.save(rf'll_1t_Y_{N}.npy', Y_lhs)

# # Generate a set of dedicated validation points

# sampler_val     = qmc.LatinHypercube(d, scramble=False, seed=129)
# lhs_samples_val = sampler_val.random(N_val) # Generate LHS samples in [0,1]^d
# lhs_samples_val = np.row_stack(([0]*d, lhs_samples_val, [1]*d)) # add boundaries
# #                    pY,    u, scale, shape,   pR,    Z,  phi, gamma_bar,  tau
# l_bounds       = [0.001,   30,     5,  -1.0, 1e-2, -5.0, 0.05,       0.5,  1.0]
# u_bounds       = [0.999,   80,    60,   1.0,  5e6,  5.0, 0.95,       8.0, 50.0]
# X_lhs_val      = qmc.scale(lhs_samples_val, l_bounds, u_bounds)        # scale LHS to specified bounds
# # X_lhs_val[:,4] = scipy.stats.levy(loc=0,scale=8.0).ppf(X_lhs_val[:,4]) # scale the Stables
# X_lhs_val[:,0] = qCGP(X_lhs_val[:,0], 0.9, X_lhs_val[:,1], X_lhs_val[:,2], X_lhs_val[:,3])
# print('X_lhs_val.shape:',X_lhs_val.shape)

# data_val = [tuple(row) for row in X_lhs_val]
# start_time = time.time()
# print(rf'start calculating {N_val} likelihoods using {n_processes} processes:', datetime.datetime.now())
# with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
#     results_val = pool.map(Y_ll_1t1s_par, data_val)
# end_time = time.time()
# print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

# noNA      = np.where(~np.isnan(results_val))
# Y_lhs_val = np.array(results_val)[noNA]
# X_lhs_val = X_lhs_val[noNA]

# np.save(rf'll_1t_X_val_{N_val}.npy', X_lhs_val)
# np.save(rf'll_1t_Y_val_{N_val}.npy', Y_lhs_val)

# %% step 2: load design points and train

X_lhs     = np.load(rf'll_1t_X_{N}.npy')
Y_lhs     = np.load(rf'll_1t_Y_{N}.npy')
X_lhs_val = np.load(rf'll_1t_X_val_{N_val}.npy')
Y_lhs_val = np.load(rf'll_1t_Y_val_{N_val}.npy')

# plotting the log-likelihoods:
#   reveals that the logged version can be so negative 
#   that np.min(y_val) = -8.475824028546802e+17

# Train on original scale likelihood

X_train = X_lhs
y_train = np.exp(Y_lhs)
X_val   = X_lhs_val
y_val   = np.exp(Y_lhs_val)

# %% step 2a: emulate with scipy rbf smoothing/interpolating splines

# spline_Y_ll_1t = RBFInterpolator(X_lhs, Y_lhs, 
#                                  kernel='thin_plate_spline', degree=3,
#                                  neighbors=None, epsilon=None,  smoothing=0.0)

# save_pickle_data(spline_Y_ll_1t)

# spline_Y_ll_1t(X_lhs) # took 1.5 seconds

# # LMSE
# np.log(1/N * np.sum(spline_Y_ll_1t(X_lhs) - Y_lhs)**2)


# %% step 2b: emulate with keras+tensorflow

# def log_with_sign(arr):
#     # to account for the "0" in training likelihoods
#     # remember to return 0 if np.exp(predict) = 1
#     return np.where(arr == 0,
#                     0,
#                     np.sign(arr) * np.log(np.absolute(arr)))

# """
# Split train and validation set
# """

# train_size    = 0.9
# indices       = np.arange(X_lhs.shape[0])
# np.random.shuffle(indices)

# split_idx     = int(X_lhs.shape[0] * train_size)
# train_indices = indices[:split_idx]
# test_indices  = indices[split_idx:]

# X_train      = X_lhs[train_indices]
# X_val        = X_lhs[test_indices]
# y_train      = log_with_sign(Y_lhs[train_indices])
# y_val        = log_with_sign(Y_lhs[test_indices])

# """
# Define and fit Keras model
# """

# model = keras.Sequential(
#     [   
#         keras.Input(shape=(d,)),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ]
# )

# initial_learning_rate = 0.001
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
# )

# model.compile(
#     # optimizer='adam',
#     optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule), 
#     loss=keras.losses.mean_squared_error)

# # Fitting Model

# start_time = time.time()
# print('started fitting NN:', datetime.datetime.now())

# checkpoint_filepath = './checkpoint.model.keras' # only saves the best performer seen so far after each epoch 
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
#                                                             monitor='val_loss',
#                                                             mode='min',
#                                                             save_best_only=True)
# history = model.fit(
#     X_train, 
#     y_train, 
#     epochs = EPOCH, 
#     initial_epoch=INITIAL_EPOCH,
#     verbose = 2,
#     validation_data=(X_val, y_val),
#     callbacks=[model_checkpoint_callback])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('MSE loss')
# plt.savefig('./Plot:val_loss.pdf')
# plt.show()
# plt.close()

# bestmodel = keras.models.load_model(checkpoint_filepath)
# bestmodel.save('./Y_ll_1t_NN.keras')

# %% step 2c: train on the original scale likelihood

model = keras.Sequential(
    [
        keras.Input(shape=(d,)),
        # layers.Dense(128,  activation='elu'),
        # layers.Dense(256,  activation='elu'),
        layers.Dense(512,  activation='relu'),
        layers.Dense(512,  activation='relu'),
        layers.Dense(512,  activation='relu'),
        layers.Dense(512,  activation='relu'),
        layers.Dense(512,  activation='relu'),
        # layers.Dense(256,  activation='elu'),
        # layers.Dense(128,  activation='elu'),
        layers.Dense(1)
    ]
)

initial_learning_rate = 1e-5
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 
    decay_steps=5e4, # 100,000,000/batch_size = steps per epoch
    decay_rate=0.96, 
    staircase=False
)

model.compile(
    # optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule), 
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
    loss=keras.losses.mean_squared_error)

# load previously defined model
# model = keras.models.load_model('./checkpoint.model.keras')

# Fitting Model

start_time = time.time()
print('started fitting NN:', datetime.datetime.now())

checkpoint_filepath = './checkpoint.model.keras' # only saves the best performer seen so far after each epoch 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True)
history = model.fit(
    X_train, 
    y_train, 
    initial_epoch=INITIAL_EPOCH,
    epochs = EPOCH, 
    batch_size=8192,
    verbose = 2,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint_callback])

end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.title('validation loss')
plt.savefig(rf'Plot_val_loss_{INITIAL_EPOCH}to{EPOCH}.pdf')
plt.show()
plt.close()

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.title('training loss')
plt.savefig(rf'Plot_train_loss_{INITIAL_EPOCH}to{EPOCH}.pdf')
plt.show()
plt.close()

bestmodel = keras.models.load_model(checkpoint_filepath)
bestmodel.save(rf'./Y_L_1t_NN_{N}.keras')


# %% step 3: Build Prediction Functions
"""
- Try the emulator on a "profile-ish" likelihood for some parameter
"""

# %% Spline emulator -------------------------------------------------------------

# # The emulator one does not work, 
# # at least not on the log-likelihood

# # emulator_spline = read_pickle_data('spline_Y_ll_1t')

# # def ll_1t_spline(Y, p, u_vec, scale_vec, shape_vec,            # marginal model parameters
# #                  R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau,        # dependence model parameters
# #                  logS_vec, gamma_at_knots, censored_idx, exceed_idx,  # auxilury information
# #                  emulator):
    
# #     Y_ll = emulator(np.array([Y, u_vec, scale_vec, shape_vec, R_vec, Z_vec, phi_vec, gamma_bar_vec, np.full_like(Y, tau)]).T)

# #     # log likelihood of S
# #     S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

# #     # log likelihood of Z
# #     Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

# #     return np.sum(Y_ll) + np.sum(S_ll) + np.sum(Z_ll)

# # def ll_1t_spline_par(args):
# #     Y, p, u_vec, scale_vec, shape_vec, \
# #     R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
# #     logS_vec, gamma_at_knots, censored_idx, exceed_idx, emulator = args

# #     Y_ll = emulator(np.array([Y, u_vec, scale_vec, shape_vec, R_vec, Z_vec, phi_vec, gamma_bar_vec, np.full_like(Y, tau)]).T)

# #     # log likelihood of S
# #     S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

# #     # log likelihood of Z
# #     Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

# #     return np.sum(Y_ll) + np.sum(S_ll) + np.sum(Z_ll)



# %% Neural emulator -------------------------------------------------------------

model_nn = keras.models.load_model(rf"Y_L_1t_NN_{N}.keras")

def relu_np(x): 
    # np.maximum(x, 0, x) # changes x IN PLACE! faster than return x * (x > 0)
    return np.where(x > 0, x, 0)

def elu_np(x):
    return np.where(x > 0, 
                    x, 
                    np.exp(x) - 1)

def identity(x):
    return x


Ws, bs, acts = [], [], []
for layer in model_nn.layers:
    W, b = layer.get_weights()
    if layer.get_config()['activation'] == 'relu':
        act = relu_np
    elif layer.get_config()['activation'] == 'elu':
        act = elu_np
    elif layer.get_config()['activation'] == 'linear':
        act = identity
    else:
        print(layer.get_config()['activation'])
        raise NotImplementedError
    Ws.append(W)
    bs.append(b)
    acts.append(act)

# Manually perform matrix multiplication, 
# to bypass the keras parallelization issue

# the output is 1D if X is 1D
#               2D if X is 2D
def Y_ll_1t1s_nn(Ws, bs, activations, X):
    Z = X
    for W, b, activation in zip(Ws, bs, activations):
        Z = Z @ W + b
        Z = activation(Z)
    return np.where(Z > 0, np.log(Z), 0)

def Y_L_1t1s_nn(Ws, bs, activations, X):
    Z = X
    for W, b, activation in zip(Ws, bs, activations):
        Z = Z @ W + b
        Z = activation(Z)
    return Z

# check for extrapolation
def Y_ll_1t1s_nn_2p(Ws, bs, activations, X):
    condition_Y          = (30 <= X[:,0]) & (X[:,0] <= 6020)
    condition_u          = (30 <= X[:,1]) & (X[:,1] <= 80)
    conditiona_scale     = (5 <= X[:,2]) & (X[:,2] <= 60)
    conditiona_shape     = (-1.0 <= X[:,3]) & (X[:,3] <= 1.0)
    conditiona_pR        = (1e-2 <= X[:,4]) & (X[:,4] <= 5e6)
    conditiona_Z         = (-5.0 <= X[:,5]) & (X[:,5] <= 5.0)
    conditiona_phi       = (0.05 <= X[:,6]) & (X[:,6] <= 0.95)
    conditiona_gamma_bar = (0.5 <= X[:,7]) & (X[:,7] <= 8.0)
    conditiona_tau       = (1.0 <= X[:,8]) & (X[:,8] <= 50.0)
    condition            = condition_Y & condition_u & conditiona_scale & conditiona_shape & conditiona_pR & conditiona_Z & conditiona_phi & conditiona_gamma_bar & conditiona_tau
    
    interp_idx = np.where(condition)[0]
    extrap_idx = np.where(~condition)[0]
    
    ll = np.full((len(X),), np.nan)
    ll[interp_idx] = Y_ll_1t1s_nn(Ws, bs, activations, X[interp_idx]).ravel()
    ll[extrap_idx] = [Y_ll_1t1s(*X[idx]) for idx in extrap_idx]

    print("proportion extrapolated:",len(extrap_idx) / len(X))

    return ll

# np.log(1/N * np.sum((Y_ll_1t1s_nn(Ws, bs, acts, X_lhs) - Y_lhs)**2))

# %% loading and setting up for plotting
# imports -------------------------------------------------------------------------------------------------------------


datafolder = './simulated_seed-2345_t-60_s-50_phi-nonstatsc2_rho-nonstat_tau-10.0/'
datafile   = 'simulated_data.RData'
r(f'''
    load("{datafolder}/{datafile}")
''')

# Load from .RData file the following
#   Y, 
#   GP_estimates (u, logsigma, xi), 
#   elev, 
#   stations

Y                  = np.array(r('Y'))
GP_estimates       = np.array(r('GP_estimates')).T
logsigma_estimates = GP_estimates[:,1]
xi_estimates       = GP_estimates[:,2]
elevations         = np.array(r('elev'))
stations           = np.array(r('stations')).T

# this `u_vec` is the threshold, 
# spatially varying but temporally constant
# ie, each site has its own threshold
u_vec              = GP_estimates[:,0]


# Setup (Covariates and Constants) ------------------------------------------------------------------------------------

# Ns, Nt

Ns = Y.shape[0] # number of sites/stations
Nt = Y.shape[1] # number of time replicates
start_year = 1949
end_year   = 2023
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

p        = 0.9
u_matrix = np.full(shape = (Ns, Nt), fill_value = np.nanquantile(Y, p)) # threshold u on Y, i.e. p = Pr(Y <= u)
# u_vec    = u_matrix[:,rank]

# Sites

sites_xy = stations
sites_x = sites_xy[:,0]
sites_y = sites_xy[:,1]

# define the lower and upper limits for x and y
minX, maxX = 0.0, 10.0
minY, maxY = 0.0, 10.0

# Knots - isometric grid of 9 + 4 = 13 knots ----------------------------------

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

# Load/Hardcode parameters --------------------------------------------------------------------------------------------

# True values as intials with the simulation

data_seed = 2345

simulation_threshold = 60.0
Beta_logsigma        = np.array([3.0, 0.0])
Beta_xi              = np.array([0.1, 0.0])
range_at_knots       = np.sqrt(0.3*knots_x_rho + 0.4*knots_y_rho)/2
phi_at_knots         = 0.65 - np.sqrt((knots_x_phi-5.1)**2/5 + (knots_y_phi-5.3)**2/4)/11.6
gamma_k_vec          = np.repeat(0.5, k_S)
tau                  = 10

np.random.seed(data_seed)

# Marginal Model

u_matrix = np.full(shape = (Ns, Nt), fill_value = simulation_threshold)

sigma_Beta_logsigma = 1
sigma_Beta_xi      = 1

# g(Z) Transformed Gaussian Process

range_vec      = gaussian_weight_matrix_rho @ range_at_knots
K              = ns_cov(range_vec = range_vec, sigsq_vec = sigsq_vec,
                        coords = sites_xy, kappa = nu, cov_model = "matern")
Z              = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(Ns,)),cov=K,size=Nt).T
W              = g(Z)

# phi Dependence parameter

phi_vec        = gaussian_weight_matrix_phi @ phi_at_knots

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

Scale_matrix = np.exp((C_logsigma.T @ Beta_logsigma).T)
Shape_matrix = (C_xi.T @ Beta_xi).T

# %% Step 4: plotting marginal likelihood surface
# phi -------------------------------------------------------------------------------------------------------------

# for i in range(k_phi):
for i in range(1):

    print(phi_at_knots[i]) # which phi_k value to plot a "profile" for

    lb = 0.2
    ub = 0.8
    grids = 5 # fast
    # grids = 13
    phi_grid = np.linspace(lb, ub, grids)
    phi_grid = np.sort(np.insert(phi_grid, 0, phi_at_knots[i]))

    # unchanged from above:
    #   - range_vec
    #   - K
    #   - tau
    #   - gamma_bar_vec
    #   - p
    #   - u_matrix
    #   - Scale_matrix
    #   - Shape_matrix

    # Using neural network emulator optimized ------------------------------

    """
    Idea:
        It might be much better to call NN once for a big X
        than call NN for each t separately
    """

    ll_phi_NN_opt = []
    start_time = time.time()
    for phi_x in phi_grid:
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        ll         = []
        input_list = [] # used to calculate all the Y-likelihoods
        for t in range(Nt):
            Y_1t      = Y[:,t]
            u_vec     = u_matrix[:,t]
            Scale_vec = Scale_matrix[:,t]
            Shape_vec = Shape_matrix[:,t]
            R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            logS_vec  = np.log(S_at_knots[:,t])
            # censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            # exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]
            X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec_test, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
            input_list.append(X_input)

            S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
            Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

            ll.append(np.sum(S_ll) + np.sum(Z_ll))

        input_list = np.vstack(input_list)
        Y_ll_all   = Y_ll_1t1s_nn(Ws,bs,acts,input_list)
        Y_ll_split = np.split(Y_ll_all, Nt)
        for t in range(Nt):
            ll[t] += np.sum(Y_ll_split[t])

        ll_phi_NN_opt.append(np.array(ll))

    ll_phi_NN_opt = np.array(ll_phi_NN_opt)
    np.save(rf'll_phi_NN_opt_k{i}', ll_phi_NN_opt)

    # Using neural network emulator optimized 2-piece ----------------------

    """
    Idea:
        It might be much better to call NN once for a big X
        than call NN for each t separately
    """

    ll_phi_NN_opt_2p = []
    start_time = time.time()
    for phi_x in phi_grid:
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        ll         = []
        input_list = [] # used to calculate all the Y-likelihoods
        for t in range(Nt):
            Y_1t      = Y[:,t]
            u_vec     = u_matrix[:,t]
            Scale_vec = Scale_matrix[:,t]
            Shape_vec = Shape_matrix[:,t]
            R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            logS_vec  = np.log(S_at_knots[:,t])
            # censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            # exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]
            X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec_test, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
            input_list.append(X_input)

            S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
            Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

            ll.append(np.sum(S_ll) + np.sum(Z_ll))

        input_list = np.vstack(input_list)
        Y_ll_all   = Y_ll_1t1s_nn_2p(Ws,bs,acts,input_list)
        Y_ll_split = np.split(Y_ll_all, Nt)
        for t in range(Nt):
            ll[t] += np.sum(Y_ll_split[t])

        ll_phi_NN_opt_2p.append(np.array(ll))

    ll_phi_NN_opt_2p = np.array(ll_phi_NN_opt_2p)
    np.save(rf'll_phi_NN_opt_2p_k{i}', ll_phi_NN_opt_2p)

    # %% actual calculation ------------------------------------------------------

    ll_phi     = []
    start_time = time.time()
    for phi_x in phi_grid:
        
        args_list = []
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            u_vec     = u_matrix[:,t]
            Scale_vec = Scale_matrix[:,t]
            Shape_vec = Shape_matrix[:,t]

            # copula process
            R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
            Z_1t      = Z[:,t]

            logS_vec  = np.log(S_at_knots[:,t])

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
                            logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t))

        with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_phi.append(np.array(results))

    ll_phi = np.array(ll_phi, dtype = object)
    np.save(rf'll_phi_k{i}', ll_phi)  

    # Plotting -------------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(phi_grid, np.nansum(ll_phi_NN_opt, axis = 1), 'b.-', label = 'log(NN_opt)')
    ax.plot(phi_grid, np.sum(ll_phi_NN_opt_2p, axis = 1), 'g.-', label = 'log(NN_opt_2p)')
    ax.plot(phi_grid, np.sum(ll_phi, axis = 1), 'k.-', label = 'true log likelihood')

    ax.axvline(x=phi_at_knots[i], color='r', linestyle='--')
    ax.legend(loc = 'lower left')
    ax.set_title(rf'marginal loglike against $\phi_{i}$')
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel('log likelihood')

    ax.set_yscale('symlog')
    if ax.get_yscale() == 'symlog':
        ax.set_ylabel('log likelihood -- plt.yscale("symlog")')

    plt.savefig(rf'profile_ll_phi_k{i}.pdf')
    plt.show()
    plt.close()

    # ax.plot(phi_grid, np.sum(ll_phi_emulator_spline, axis = 1), 'r.-', label = 'spline emulator')
    
    # plot_ll_nn = []
    # float_arr = np.vstack(ll_phi_emulator_nn[:, :]).astype(np.float64)
    # for j in range(len(phi_grid)):
    #     plot_ll_nn.append(np.sum(float_arr[j][np.isfinite(float_arr[j])]))
    # ax.plot(phi_grid, plot_ll_nn, 'b.-', label = 'nn emulator')
    

# %% Using spline emulator ----------------------------------------------------

# ll_phi_emulator_spline = []
# start_time = time.time()
# for phi_x in phi_grid:
#     args_list = []
#     print('elapsed:', round(time.time() - start_time, 3), phi_x)

#     phi_k        = phi_at_knots.copy()
#     phi_k[i]     = phi_x
#     phi_vec_test = gaussian_weight_matrix_phi @ phi_k

#     for t in range(Nt):
#         Y_1t      = Y[:,t]
#         u_vec     = u_matrix[:,t]
#         Scale_vec = Scale_matrix[:,t]
#         Shape_vec = Shape_matrix[:,t]

#         R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
#         Z_1t      = Z[:,t]

#         logS_vec  = np.log(S_at_knots[:,t])

#         censored_idx_1t = np.where(Y_1t <= u_vec)[0]
#         exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

#         args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
#                           R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
#                           logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t, emulator_spline))
        
#     with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
#         results = pool.map(ll_1t_spline_par, args_list)
#     ll_phi_emulator_spline.append(np.array(results))

# ll_phi_emulator_spline = np.array(ll_phi_emulator_spline, dtype = object)
# np.save(rf'll_phi_emulator_spline_k{i}', ll_phi_emulator_spline)

# %% Using neural network emulator --------------------------------------------

# ll_phi_emulator_nn = []
# start_time = time.time()
# for phi_x in phi_grid:
#     args_list = []
#     print('elapsed:', round(time.time() - start_time, 3), phi_x)

#     phi_k        = phi_at_knots.copy()
#     phi_k[i]     = phi_x
#     phi_vec_test = gaussian_weight_matrix_phi @ phi_k

#     ll = []

#     for t in range(Nt):
#         Y_1t      = Y[:,t]
#         u_vec     = u_matrix[:,t]
#         Scale_vec = Scale_matrix[:,t]
#         Shape_vec = Shape_matrix[:,t]

#         R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
#         Z_1t      = Z[:,t]

#         logS_vec  = np.log(S_at_knots[:,t])

#         censored_idx_1t = np.where(Y_1t <= u_vec)[0]
#         exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

#         X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec_test, gamma_bar_vec, np.full_like(Y_1t, tau)]).T

#         Y_ll = Y_ll_1t1s_nn(Ws, bs, acts, X_input)
#         S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
#         Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

#         ll.append(np.sum(Y_ll) + np.sum(S_ll) + np.sum(Z_ll))

#     ll_phi_emulator_nn.append(np.array(ll))

#     #     args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
#     #                       R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
#     #                       logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t, Y_ll_1t1s_nn, Ws, bs, acts))
        
#     # with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
#     #     results = pool.map(ll_1t_nn_par, args_list)
#     # ll_phi_emulator_nn.append(np.array(results))

# ll_phi_emulator_nn = np.array(ll_phi_emulator_nn, dtype = object)
# np.save(rf'll_phi_emulator_nn_k{i}', ll_phi_emulator_nn)



# %% investigate nan output

# ll_phi_NN_opt.shape # returns (len(phi_grid), Nt)

# nan_input_list = np.array(input_list)[np.where(np.isnan(ll_phi_NN_opt[-1]))[0]]
# nan_input_list.shape # returns (len(nan_input_list), Ns, 9)

# nan_ll = Y_ll_1t1s_nn(Ws,bs,acts,nan_input_list)
# nan_t_idx, nan_s_idx = np.where(np.isnan(nan_ll))[0:2]

# nan_inputs = nan_input_list[nan_t_idx, nan_s_idx, :]
# # Y_ll_1t1s_nn(Ws,bs,acts,nan_inputs) # verified that all outputs are nan
# # model_nn.predict(nan_inputs) # doesn't output nan! Oh but they output negative values

# nan_inputs[:,4]
# # #             pY,    u, scale, shape,   pR,    Z,  phi, gamma_bar,  tau
# # l_bounds = [0.001,   30,     5,  -1.0, 0.01, -5.0, 0.05,       0.5,  1.0]
# # u_bounds = [0.999,   80,    60,   1.0, 0.95,  5.0, 0.95,       8.0, 50.0]

# np.logical_or(nan_inputs[:,4] < scipy.stats.levy(loc=0,scale=8.0).ppf(0.01), 
#               nan_inputs[:,4] > scipy.stats.levy(loc=0,scale=8.0).ppf(0.95))


# %% Goodness of Fit Plot

# Goodness of fit plot on the validation dataset ------------------------------

Y_lhs_val

Y_lhs_val_pred = Y_ll_1t1s_nn_2p(Ws,bs,acts,X_lhs_val)

fig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
ax.scatter(np.exp(Y_lhs_val), np.exp(Y_lhs_val_pred))
ax.axline((0, 0), slope=1, color='black', linestyle='--')
ax.set_title(rf'Goodness of Fit Plot on Validation Dataset')
ax.set_xlabel('True exp(log Likelihood)')
ax.set_ylabel('Emulated Likelihood')
plt.savefig(r'GOF_validation.pdf')
plt.show()
plt.close()

# Goodness of fit plot on the simulated dataset -------------------------------

"""
We will need pointwise likelihood both from the emulator and the actual
Use Y_ll_1t1s to get the pointwise true likelihood
"""

# true likelihood -----------------------------------------

args_list = []
for t in range(Nt):
    Y_1t      = Y[:,t]
    u_vec     = u_matrix[:,t]
    Scale_vec = Scale_matrix[:,t]
    Shape_vec = Shape_matrix[:,t]
    R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
    Z_1t      = Z[:,t]
    logS_vec  = np.log(S_at_knots[:,t])

    X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
    args_list.append(X_input)

    # S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
    # Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)
    # ll.append(np.sum(S_ll) + np.sum(Z_ll))

args_list = np.vstack(args_list)

with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
    results = pool.starmap(Y_ll_1t1s, args_list)

# Y_ll_split = np.split(np.array(results), Nt)
Y_ll_Nt_Ns = np.array(results).reshape((Nt, Ns))
np.save(r'Y_ll_Nt_Ns_simulated_dataset.npy', Y_ll_Nt_Ns)

# emulated likelihood -------------------------------------

input_list = [] # used to calculate all the Y-likelihoods
for t in range(Nt):
    Y_1t      = Y[:,t]
    u_vec     = u_matrix[:,t]
    Scale_vec = Scale_matrix[:,t]
    Shape_vec = Shape_matrix[:,t]
    R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
    Z_1t      = Z[:,t]
    logS_vec  = np.log(S_at_knots[:,t])

    X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
    input_list.append(X_input)

input_list = np.vstack(input_list)

results          = Y_ll_1t1s_nn_2p(Ws,bs,acts,input_list)
Y_ll_Nt_Ns_nn_2p = np.array(results).reshape((Nt, Ns))

# plotting ------------------------------------------------

fig, ax = plt.subplots()
ax.set_aspect('equal', 'datalim')
ax.scatter(Y_ll_Nt_Ns, Y_ll_Nt_Ns_nn_2p)
ax.axline((0, 0), slope=1, color='black', linestyle='--')
ax.set_title(rf'Goodness of Fit Plot on Simulated Dataset all t')
ax.set_xlabel('True Log Likelihood')
ax.set_ylabel('log(Emulated Likelihood)')
plt.savefig(r'GOF_simulated_allt.pdf')
plt.show()
plt.close()

for t in range(Nt):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.scatter(Y_ll_Nt_Ns[t,:], Y_ll_Nt_Ns_nn_2p[t,:])
    ax.axline((0, 0), slope=1, color='black', linestyle='--')
    ax.set_title(rf'Goodness of Fit Plot on Simulated Dataset t={t}')
    ax.set_xlabel('True Log Likelihood')
    ax.set_ylabel('log(Emulated Likelihood)')
    # plt.axis('equal')
    plt.savefig(rf'GOF_simulated_t{t}.pdf')
    # plt.show()
    plt.close()

# %% Generating X and Y for Likun, when changing phi_0 

# # truth ---------------------------------------------------

# S_ll_Nt       = []
# Z_ll_Nt       = []

# input_list = [] # used to calculate all the Y-likelihoods
# for t in range(Nt):
#     Y_1t      = Y[:,t]
#     u_vec     = u_matrix[:,t]
#     Scale_vec = Scale_matrix[:,t]
#     Shape_vec = Shape_matrix[:,t]
#     R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
#     Z_1t      = Z[:,t]
#     logS_vec  = np.log(S_at_knots[:,t])

#     X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
#     input_list.append(X_input)

#     S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
#     Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

#     S_ll_Nt.append(np.sum(S_ll))
#     Z_ll_Nt.append(np.sum(Z_ll))

# X_input_Nt_Ns = np.vstack(input_list)

# with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
#     results = pool.starmap(Y_ll_1t1s, X_input_Nt_Ns)
# Y_Nt_Ns = np.array(results)

# np.save('Y_ll_X_input_Nt_Ns.npy', X_input_Nt_Ns)
# np.save('Y_ll_Y_Nt_Ns.npy', Y_Nt_Ns)

# # phi_grid ------------------------------------------------

# i = 0
# lb = 0.2
# ub = 0.8
# grids = 5 # fast
# phi_grid = np.linspace(lb, ub, grids)
# phi_grid = np.sort(np.insert(phi_grid, 0, phi_at_knots[i]))

# Y_Nt_Ns_phi_grid       = []
# X_input_Nt_Ns_phi_grid = []
# S_ll_Nt_phi_grid       = []
# Z_ll_Nt_phi_grid       = []

# for phi_x in phi_grid:

#     X_input_Nt_Ns = []
#     S_ll_Nt       = []
#     Z_ll_Nt       = []

#     phi_k        = phi_at_knots.copy()
#     phi_k[i]     = phi_x
#     phi_vec_test = gaussian_weight_matrix_phi @ phi_k

#     input_list = [] # used to calculate all the Y-likelihoods
#     for t in range(Nt):
#         Y_1t      = Y[:,t]
#         u_vec     = u_matrix[:,t]
#         Scale_vec = Scale_matrix[:,t]
#         Shape_vec = Shape_matrix[:,t]
#         R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
#         Z_1t      = Z[:,t]
#         logS_vec  = np.log(S_at_knots[:,t])

#         X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec_test, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
#         input_list.append(X_input)

#         S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
#         Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

#         S_ll_Nt.append(np.sum(S_ll))
#         Z_ll_Nt.append(np.sum(Z_ll))

#     input_list = np.vstack(input_list)
#     X_input_Nt_Ns_phi_grid.append(input_list)

#     S_ll_Nt_phi_grid.append(S_ll_Nt)
#     Z_ll_Nt_phi_grid.append(Z_ll_Nt)

# np.array(X_input_Nt_Ns_phi_grid).shape
# np.array(S_ll_Nt_phi_grid).shape
# np.array(Z_ll_Nt_phi_grid).shape

# for X_input in X_input_Nt_Ns_phi_grid:
#     with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
#         results = pool.starmap(Y_ll_1t1s, X_input)
#     Y_Nt_Ns_phi_grid.append(np.array(results))

# X_input_Nt_Ns_phi_grid = np.array(X_input_Nt_Ns_phi_grid)
# S_ll_Nt_phi_grid       = np.array(S_ll_Nt_phi_grid)
# Z_ll_Nt_phi_grid       = np.array(Z_ll_Nt_phi_grid)
# Y_Nt_Ns_phi_grid       = np.array(Y_Nt_Ns_phi_grid)

# plt.plot(phi_grid, np.sum(S_ll_Nt_phi_grid, axis = 1))
# plt.plot(phi_grid, np.sum(Z_ll_Nt_phi_grid, axis = 1))
# plt.plot(phi_grid, np.sum(Y_Nt_Ns_phi_grid, axis = 1))

# plt.plot(phi_grid,
#          np.sum(Y_Nt_Ns_phi_grid, axis = 1) + np.sum(Z_ll_Nt_phi_grid, axis = 1) + np.sum(S_ll_Nt_phi_grid, axis = 1))

# plt.plot(phi_grid, np.sum(Y_Nt_Ns_phi_grid, axis = 1))

# np.save('phi_grid.npy', phi_grid)
# np.save('Y_ll_X_input_Nt_Ns_phi_grid.npy', X_input_Nt_Ns_phi_grid)
# np.save('Y_ll_Y_Nt_Ns_phi_grid.npy', Y_Nt_Ns_phi_grid)