"""
emulate the qRW function on p between (0.9, 0.9999)
    - The rest of the domain will rely on numerical integral
    - remmeber to check the speed of prediction -- might be better to 
        - do all the (Ns x Nt) predictions all at once
        - rather than do Ns predictions Nt time
    - potentially need to train on log(qRW) otherwise MSE overflow
        - can also try training on LMSE?
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

N = int(1e8)
N_val = int(1e6)
d = 4

INITIAL_EPOCH = 150
EPOCH = 200

"""
Notes on coding:

    - pRW(1e16, 1, 4, 50) yields array(0.99999999)

    - This is incredibly slow -- much better to directly pass X_2d to model.predict
        def qRW_NN(x, phi, gamma, tau):
            return model.predict(np.array([[x, phi, gamma, tau]]), verbose=0)[0]
        qRW_NN_vec = np.vectorize(qRW_NN)

    - 400,000 qRW() evals in 5 minutes, 30 processes

    - Windows/Mac need to explicitly use 'fork':
        from multiprocessing import get_context
        p = get_context("fork").Pool(4)
        results = p.map(pRW_par, inputs)
        p.close()
"""

# %% Step 1: LHS design for the parameter of dRW(X, phi, gamma, tau)

lhs_sampler = qmc.LatinHypercube(d, scramble = False, seed = 1031)
lhs_samples = lhs_sampler.random(N) # doesn't include the boundary
lhs_samples = np.row_stack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary

"""
reasoning for choosing the bounds:
    l_bounds used in qRW emluation [0.9,    0.05,       0.5,   1]
    u_bounds used in qRW emulation [0.9999, 0.95,         5, 100]
    qRW(0.9, 0.05, 0.5, 1)    = 0.457873773698486
    qRW(0.9999, 0.95, 5, 100) = 402114519.4524983
"""
#            X,     phi, gamma_bar, tau
l_bounds = [10,    0.05,       0.5,   1]
u_bounds = [403e6, 0.95,         5, 100] 

X_lhs = qmc.scale(lhs_samples, l_bounds, u_bounds)

# Calculate the design points

start_time = time.time()
print("start calculate dRW()'s:", datetime.datetime.now())

with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    Y_lhs = pool.starmap(dRW, list(X_lhs))
Y_lhs = np.array(Y_lhs)

end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

np.save(rf'dRW_X_{N}.npy', X_lhs)
np.save(rf'dRW_Y_{N}.npy', Y_lhs)

# Caluclate a set of validation points

lhs_sampler_val = qmc.LatinHypercube(d, scramble = False, seed = 122)
lhs_samples_val = lhs_sampler_val.random(N_val) # doesn't include the boundary
lhs_samples_val = np.row_stack(([0]*d, lhs_samples_val, [1]*d)) # manually add the boundary
#             p,     phi, gamma_bar, tau
l_bounds  = [10,    0.05,       0.5,   1]
u_bounds  = [403e6, 0.95,         5, 100] 
X_lhs_val = qmc.scale(lhs_samples_val, l_bounds, u_bounds)

start_time = time.time()
print('start calculating validation qRW()s:', datetime.datetime.now())
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    Y_lhs_val = pool.starmap(dRW, list(X_lhs_val))
Y_lhs_val = np.array(Y_lhs_val)
end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

np.save(rf'dRW_X_val_{N_val}.npy', X_lhs_val)
np.save(rf'dRW_Y_val_{N_val}.npy', Y_lhs_val)

# %% Step 2: Emulator

X_lhs     = np.load(rf'dRW_X_{N}.npy')
Y_lhs     = np.load(rf'dRW_Y_{N}.npy')
X_lhs_val = np.load(rf'dRW_X_val_{N_val}.npy')
Y_lhs_val = np.load(rf'dRW_Y_val_{N_val}.npy')

# Use separate sets of training and validation points

X_train = X_lhs
X_val   = X_lhs_val

# to train on original scale
y_train = Y_lhs
y_val   = Y_lhs_val

# %% Defining model

model = keras.Sequential(
    [   
        keras.Input(shape=(d,)),
        layers.Dense(32,  activation = 'relu'),
        layers.Dense(64,  activation = 'relu'),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(64,  activation = 'relu'),
        layers.Dense(32,  activation = 'relu'),
        layers.Dense(1,   activation = 'linear')
    ]
)

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.98, staircase=True
)

model.compile(
    # optimizer='adam',
    optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule), 
    # loss=keras.losses.mean_squared_error
    loss = keras.losses.mean_squared_logarithmic_error)

# %% Load Previously Defined Model (For additional training)

"""
If want to train an existing model for additional epochs:
"""

model = keras.models.load_model('./checkpoint.model.keras')

# %% # Fitting Model

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
    epochs = EPOCH, 
    initial_epoch = INITIAL_EPOCH,
    batch_size = 1024,
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
bestmodel.save(rf'./qRW_NN_{N}.keras')

# saving the "best" model

# bestmodel = keras.models.load_model(checkpoint_filepath)
# Ws, bs, acts = [], [], []
# for layer in bestmodel.layers:
#     W, b = layer.get_weights()
#     act  = layer.get_config()['activation']
#     Ws.append(W)
#     bs.append(b)
#     acts.append(act)
# bestmodel.save('/qRW_NN.keras')
# # Note that numpy cannot save inhomogeneous shaped array
# #      therefore we use pickle dump
# with open('/qRW_NN_Ws.pkl',   'wb') as file: pickle.dump(Ws,   file)
# with open('/qRW_NN_bs.pkl',   'wb') as file: pickle.dump(bs,   file)
# with open('/qRW_NN_acts.pkl', 'wb') as file: pickle.dump(acts, file)


# %% Step 3: Prediction

# qRW_NN_keras = keras.models.load_model('checkpoint.model.keras')

qRW_NN_keras = keras.models.load_model(rf'qRW_NN_{N}.keras')

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
for layer in qRW_NN_keras.layers:
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

def qRW_NN(X, weights = Ws, biases = bs, activations = acts):
    Z = X.copy()
    for W, b, activation in zip(weights, biases, activations):
        Z = Z @ W + b
        Z = activation(Z)
    return Z
    # return np.exp(Z) # if training on log scale

def qRW_NN_2p(X, weights = Ws, biases = bs, activations = acts):
    # to store the outputs
    outputs = np.full((len(X),), fill_value=np.nan)

    # X is a 2D array of shape (N, 4) of (p, phi, gamma, tau)
    # use the columns of X to determine where to use NN where to use qRW
    condition_p      = (0.9  <= X[:,0]) & (X[:,0] <= 0.999)
    condition_phi    = (0.05 <= X[:,1]) & (X[:,1] <= 0.95)
    conditiona_gamma = (0.5  <= X[:,2]) & (X[:,2] <= 5)
    conditiona_tau   = (1    <= X[:,3]) & (X[:,3] <= 100)
    combined_condition = condition_p & condition_phi & conditiona_gamma & conditiona_tau

    emulate_idx   = np.where(combined_condition)[0]
    calculate_idx = np.where(~combined_condition)[0]

    outputs[emulate_idx] = qRW_NN(X[emulate_idx], Ws, bs, acts).ravel()
    outputs[calculate_idx] = qRW(X[calculate_idx,0], X[calculate_idx,1], X[calculate_idx,2], X[calculate_idx,3])
    
    return outputs

# %% Step 4: Evaluation
# Make example qRW plots

phi   = 0.5
gamma = 0.5
tau   = 1
ps    = np.linspace(0.9, 0.999, 100)
tasks = np.array([[p, phi, gamma, tau] for p in ps])
plt.plot(ps, qRW(ps, phi, gamma, tau), 'k.-', label = 'truth')
# plt.plot(ps, qRW_NN_keras.predict(tasks, verbose = 0).ravel(), label = 'NN')
plt.plot(ps, qRW_NN(tasks, Ws, bs, acts), 'b.-', label = 'qRW NN')
plt.legend(loc = 'upper left')
plt.xlabel('p')
plt.ylabel('quantile')
plt.xticks(np.linspace(0.9, 0.999, 5))
plt.title(rf'qRW(...) along p with $\phi$={phi} $\gamma$={gamma} $\tau$={tau}')
plt.savefig(rf'Plot_qRW_{INITIAL_EPOCH}to{EPOCH}.pdf')
plt.show()
plt.close()

phi   = 0.7
gamma = 5
tau   = 25
ps    = np.linspace(0.9, 0.999, 100)
tasks = np.array([[p, phi, gamma, tau] for p in ps])
plt.plot(ps, qRW(ps, phi, gamma, tau), 'k.-', label = 'truth')
# plt.plot(ps, qRW_NN_keras.predict(tasks, verbose = 0).ravel(), label = 'NN')
plt.plot(ps, qRW_NN(tasks, Ws, bs, acts), 'b.-', label = 'qRW NN')
plt.legend(loc = 'upper left')
plt.xlabel('p')
plt.ylabel('quantile')
plt.xticks(np.linspace(0.9, 0.999, 5))
plt.title(rf'qRW(...) along p with $\phi$={phi} $\gamma$={gamma} $\tau$={tau}')
plt.savefig(rf'Plot_qRW2_{INITIAL_EPOCH}to{EPOCH}.pdf')
plt.show()
plt.close()

# Marginal Likelihood


# # Number of rows
# rows = 1000
# # Generate individual ranges for each column
# col1 = np.linspace(0.8, 0.9999, rows)
# col2 = np.linspace(0.01, 0.99, rows)
# col3 = np.linspace(0.4, 10, rows)
# col4 = np.linspace(1, 100, rows)
# # Shuffle each column independently to create the mix-and-match effect
# np.random.shuffle(col1)
# np.random.shuffle(col2)
# np.random.shuffle(col3)
# np.random.shuffle(col4)
# # Combine the columns into a 2D array
# X = np.column_stack((col1, col2, col3, col4))

# %%
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
# %% the likelihood functions ----------------------------------------------------------------------------------------

def ll_1t_par(args):
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

def ll_1t_par_NN_2p(args):
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx, \
    Ws, bs, acts = args

    # calculate X using qRW_NN_2p
    X_star = (R_vec ** phi_vec) * g(Z_vec)
    pY     = pCGP(Y, p, u_vec, scale_vec, shape_vec)
    X      = qRW_NN_2p(np.column_stack((pY, phi_vec, gamma_bar_vec, np.full((len(Y),), tau))),
                  Ws, bs, acts)
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

def ll_1t_par_NN_2p_opt(args):
    Y, p, u_vec, scale_vec, shape_vec, \
    R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, \
    logS_vec, gamma_at_knots, censored_idx, exceed_idx, \
    X, Ws, bs, acts = args

    X_star = (R_vec ** phi_vec) * g(Z_vec)
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

# %%
# phi -------------------------------------------------------------------------------------------------------------

# for i in range(k_phi):
for i in [0]:

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
    
    # %% Using qRW_NN_2p ---------------------------------------------------------

    ll_phi_NN_2p = []
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
                            logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t,
                            Ws, bs, acts))
        
        with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
            results = pool.map(ll_1t_par_NN_2p, args_list)
        ll_phi_NN_2p.append(np.array(results))

    ll_phi_NN_2p = np.array(ll_phi_NN_2p)
    np.save(rf'll_phi_NN_2p_k{i}', ll_phi_NN_2p)

    # %% Optimized using qRW_NN_2p -----------------------------------------------

    """
    Idea:
        It might be much better to call NN once for a big X
        than call NN for each t separately
    """

    ll_phi_NN_2p_opt = []
    start_time = time.time()
    for phi_x in phi_grid:
        print('elapsed:', round(time.time() - start_time, 3), phi_x)

        phi_k        = phi_at_knots.copy()
        phi_k[i]     = phi_x
        phi_vec_test = gaussian_weight_matrix_phi @ phi_k

        # Calculate the X all at once
        input_list = [] # used to calculate X
        for t in range(Nt):
            pY_t = pCGP(Y[:,t], p, u_matrix[:,t], Scale_matrix[:,t], Shape_matrix[:,t])
            X_t = np.column_stack((pY_t, phi_vec_test, gamma_bar_vec, np.full((len(pY_t),), tau)))
            input_list.append(X_t)

        X_nn = qRW_NN_2p(np.vstack(input_list), Ws, bs, acts)

        # Split the X to each t, and use the 
        # calculated X to calculate likelihood
        X_nn = X_nn.reshape(Nt, Ns).T

        args_list = []

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

            X_1t      = X_nn[:,t]

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, K, phi_vec_test, gamma_bar_vec, tau,
                            logS_vec, gamma_k_vec, censored_idx_1t, exceed_idx_1t,
                            X_1t, Ws, bs, acts))
        
        with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
            results = pool.map(ll_1t_par_NN_2p_opt, args_list)
        ll_phi_NN_2p_opt.append(np.array(results))

    ll_phi_NN_2p_opt = np.array(ll_phi_NN_2p_opt)
    np.save(rf'll_phi_NN_2p_opt_k{i}', ll_phi_NN_2p_opt)


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

    plt.plot(phi_grid, np.sum(ll_phi, axis = 1), 'b.-', label = 'actual')
    plt.plot(phi_grid, np.sum(ll_phi_NN_2p, axis = 1), 'r.-', label = 'qRW_NN_2p emulator')
    plt.yscale('symlog')
    plt.axvline(x=phi_at_knots[i], color='r', linestyle='--')
    plt.legend(loc = 'upper left')
    plt.title(rf'marginal loglike against $\phi_{i}$')
    plt.xlabel(r'$\phi$')
    plt.ylabel('log likelihood')
    plt.savefig(rf'profile_ll_phi_k{i}.pdf')
    plt.show()
    plt.close()
# %%
