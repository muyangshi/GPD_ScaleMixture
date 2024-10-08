"""
Overview:

    Write a "non-optimized" ll function (calculate qRW inside ll function)
    LHS Design X
    Calculate Design Y <- Y is calculated from reversing GP scale and shape
    NN train on (X,Y)

"""

# %% imports and utils
# base python
import os
import time
import datetime
import pickle
import multiprocessing # Note on Arm Macs, default is spawn. get_context('fork') to use on laptop
                       # with multiprocessing.get_context('fork').Pool(n_processes)
from pathlib import Path
# packages
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit, float64, float32
import keras
from keras import layers
keras.backend.set_floatx('float64')
# custom modules
from utilities import *

"""
Constants:
    p         - threshold probability set to 0.9 (constant) set to 0.9
    u_vec     - the corresponding threshold quantile (on Y) set to 0
    

Marginal Parameter:
    scale_vec (sigma)  - 5 to 30 (looking back at the logsigma of GEV chains)
    shape_vec (ksi)    - -0.5 to 0.8
    Y <- generate GP using scale and shape?


Copula Parameter:
    R         - 0.01 to qlevy(0.95)
    Z         - -5.0 to 5.0 shall suffice
    phi_vec   - 0.05 to 0.9
    gamma_vec - 0.4 to 4
    tau       - 0.5 to 50
"""


# %% non-optimized ll function ----------------------------------------------------------------------------------------
def Y_ll_1t(Y, p, u_vec, scale_vec, shape_vec,
            R_vec, Z_vec, phi_vec, gamma_vec, tau):

    X_star       = (R_vec ** phi_vec) * g(Z_vec)
    X            = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_vec, tau)
    dX           = dRW(X, phi_vec, gamma_vec, tau)
    censored_idx = np.where(Y <= u_vec)[0]
    exceed_idx   = np.where(Y  > u_vec)[0]

    if(isinstance(Y, (int, np.int64, float))): 
        Y = np.array([Y], dtype='float64')

    # log likelihood of the censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)

    # log likelihood of the exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])
    
    ll = np.empty(len(Y))
    ll[censored_idx] = censored_ll
    ll[exceed_idx]   = exceed_ll

    return ll

def Y_ll_1t_par(args): # wrapper to put Y_ll_1t for multiprocessing
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, phi_vec, gamma_vec, tau = args
    return Y_ll_1t(Y, p, u_vec, scale_vec, shape_vec, R_vec, Z_vec, phi_vec, gamma_vec, tau)

# %%
# Design points X -----------------------------------------------------------------------------------------------------

n_samples = int(1e6)
savefolder = '../data/ll1t_LHS_'+ str(n_samples)
Path(savefolder).mkdir(parents=True, exist_ok=True)

# scale, shape, R, Z, phi, gamma, tau
n_param = 7
LHSampler = scipy.stats.qmc.LatinHypercube(d = n_param,
                                           scramble = True, seed = 2345)
LHSamples = LHSampler.random(n_samples-2)
LHSamples = np.row_stack(([0]*n_param, [1]*n_param, LHSamples)) # need to manually codein the bounds

l_bounds = np.array([5.0,     #scale
                     -0.5,  #shape
                     0.01,   #R
                     -5.0,  #Z
                     0.05,  #phi
                     0.4,   #gamma
                     0.5    #tau
                     ])
u_bounds = np.array([30.0,     #scale
                     0.8,   #shape
                     qlevy(0.95),  #R
                     5.0,   #Z
                     0.9,   #phi
                     4.0,   #gamma
                     50     #tau
                     ])

LHS_tmp = scipy.stats.qmc.scale(LHSamples, 
                                l_bounds = l_bounds,
                                u_bounds = u_bounds, 
                                reverse  = False)

# Use the generated sigma and ksi to draw Y
Y     = scipy.stats.genpareto.rvs(c = LHS_tmp[:,1], loc = 0, scale = LHS_tmp[:,0], random_state = 2345)
LHS_X = np.column_stack((Y, [0.9]*n_samples, [0.0]*n_samples, LHS_tmp))
np.save(savefolder + '/LHS_X', LHS_X)


# %%
# Calculate the Design points Y (log likelihoods) ---------------------------------------------------------------------
nprocesses = 50


start_time = time.time()
print('start design point Y calculation:', datetime.datetime.now())

# inside ll_1t, some variable requires vector
tasks = np.array([[np.array([item]) for item in LHS_X[i,:]] for i in range(n_samples)])

with multiprocessing.get_context('fork').Pool(processes=nprocesses) as pool:
    LHS_Y = pool.map(Y_ll_1t_par, list(tasks))
LHS_Y = np.array(LHS_Y)

np.save(savefolder + '/LHS_Y', LHS_Y)
print('done: took', round(time.time() - start_time, 3), 'seconds')

# %%
# Training and Validation Dataset for NN ------------------------------------------------------------------------------

LHS_X = np.load(savefolder + '/LHS_X.npy')
LHS_Y = np.load(savefolder + '/LHS_Y.npy')

train_size    = 0.8
indices       = np.arange(LHS_X.shape[0])
np.random.shuffle(indices)
split_idx     = int(LHS_X.shape[0] * train_size)
train_indices = indices[:split_idx]
test_indices  = indices[split_idx:]

X_train = LHS_X[train_indices]
X_val   = LHS_X[test_indices]
y_train = LHS_Y[train_indices]
y_val   = LHS_Y[test_indices]

def log_with_sign(arr):
    log_value = np.log(np.absolute(arr))
    return np.sign(arr) * log_value

y_train = log_with_sign(y_train)
y_val   = log_with_sign(y_val)


# %%
# Keras Model ---------------------------------------------------------------------------------------------------------

model = keras.Sequential(
    [   
        keras.Input(shape=(10,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting Model

start_time = time.time()
print('started fitting NN:', datetime.datetime.now())

checkpoint_filepath = savefolder + '/checkpoint.model.keras' # only saves the best performer seen so far after each epoch 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor='val_loss',
                                                            mode='max',
                                                            save_best_only=True)
history = model.fit(X_train, y_train, epochs = 500, 
                    verbose = 2,
                    validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.savefig(savefolder + '/Plot:val_loss.pdf')
plt.show()
plt.close()

bestmodel = keras.models.load_model(checkpoint_filepath)
bestmodel.save(savefolder + '/ll_1t_NN.keras')

# %%
# Make example ll plots

# %%
