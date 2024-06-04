"""
Overview:

    Calculate Design points for qRW()
    Create a Keras-Tensorflow Neural Network emulating the qRW(p, phi, gamma, tau) function:
        note: training on log(qRW) otherwise MSE overflow

File Structures:

    - For grid desing:
        Save to [savefolder]: data/qRW_p#_phi#_gamma#_tau#/
            - the grid of design points
            - .keras model

    - For space filling design:
        Save to [savefolder]: data/qRW_LHS_'+ str(n_samples)
            - the grid of design points
            - .keras model

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
# %% imports
# base python
import os
import time
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

# # %% set up and helper functions --------------------------------------------------------------------------------------

# p
lb_p = 0.9
ub_p = 0.9999

# phi
lb_phi = 0.05
ub_phi = 0.99

# gamma
lb_gamma = 0.4 # we chose gamma_k to be 0.5. This won't be smaller than 0.5
ub_gamma = 4

# tau
lb_tau = 0.1
ub_tau = 50

def qRW_par(args): # wrapper to put qRW for multiprocessing
    p, phi, gamma, tau = args
    return(qRW(p, phi, gamma, tau))

# %% Grid Design Points ---------------------------------------------------------------------------------------------

# n_p        = 100
# n_phi      = 20
# n_gamma    = 10
# n_tau      = 20
# savefolder = '../data/qRW_grid'+                \
#                      '_p'      + str(n_p)     + \
#                      '_phi'    + str(n_phi)   + \
#                      '_gamma'  + str(n_gamma) + \
#                      '_tau'    + str(n_tau)
# Path(savefolder).mkdir(parents=True, exist_ok=True)

# p_samples     = 2 - np.geomspace(2-ub_p, 2-lb_p, n_p)[::-1]
# phi_samples   = np.linspace(lb_phi, ub_phi, n_phi)
# gamma_samples = np.linspace(lb_gamma, ub_gamma, n_gamma)
# tau_samples   = np.linspace(lb_tau, ub_tau, n_tau)

# P, Phi, Gamma, Tau = np.meshgrid(p_samples, phi_samples, gamma_samples, tau_samples, indexing='ij')
# P_flat             = P.ravel()
# Phi_flat           = Phi.ravel()
# Gamma_flat         = Gamma.ravel()
# Tau_flat           = Tau.ravel()
# inputs             = np.column_stack((P_flat, Phi_flat, Gamma_flat, Tau_flat))

# %% LatinHypercube Design points -------------------------------------------------------------------------------------

n_samples = int(1e7)
savefolder = '../data/qRW_LHS_'+ str(n_samples)
Path(savefolder).mkdir(parents=True, exist_ok=True)

LHSampler = scipy.stats.qmc.LatinHypercube(d = 4, scramble = True, seed = 2345)
LHSamples = LHSampler.random(n_samples-2)
LHSamples = np.row_stack(([0,0,0,0], [1,1,1,1], LHSamples)) # need to manually codein the bounds
inputs    = scipy.stats.qmc.scale(LHSamples, 
                                  l_bounds = np.array([lb_p, lb_phi, lb_gamma, lb_tau]),
                                  u_bounds = np.array([ub_p, ub_phi, ub_gamma, ub_tau]), 
                                  reverse  = False)

# %% Calculate and Save the training data -----------------------------------------------------------------------------

nprocesses = 50

start_time = time.time()
print('start x calculation:', start_time)

with multiprocessing.get_context('fork').Pool(processes=nprocesses) as pool:
    x_samples = pool.map(qRW_par, list(inputs))
x_samples = np.array(x_samples)

end_time = time.time()

print('done:', round(end_time - start_time, 3))
print('processes:', str(nprocesses))
np.save(savefolder + '/inputs',    inputs)
np.save(savefolder + '/x_samples', x_samples)

# %% Training and Validation Dataset ----------------------------------------------------------------------------------

"""
Split the dataset we just calculated into training and validation chunks
"""

train_size    = 0.8
indices       = np.arange(inputs.shape[0])
np.random.shuffle(indices)
split_idx     = int(inputs.shape[0] * train_size)
train_indices = indices[:split_idx]
test_indices  = indices[split_idx:]

X_train = inputs[train_indices]
X_val   = inputs[test_indices]
y_train = x_samples[train_indices]
y_val   = x_samples[test_indices]

y_train = np.log(y_train)
y_val   = np.log(y_val)


"""
Load previously calculated dataset(s) as training and validation
"""

# # training (load and shuffle)
# inputs    = np.load('../data/qRW_LHS_50000000/inputs.npy')
# x_samples = np.load('../data/qRW_LHS_50000000/x_samples.npy')
# indices   = np.arange(inputs.shape[0])
# np.random.shuffle(indices)
# X_train   = inputs[indices]
# y_train   = np.log(x_samples[indices])

# # validation (load)
# loadfolder = '../data/qRW_LHS_5000000'
# X_val = np.load(loadfolder        + '/inputs.npy')
# y_val = np.log(np.load(loadfolder + '/x_samples.npy'))



# %% Setup Keras model ------------------------------------------------------------------------------------------------

model = keras.Sequential(
    [   
        keras.Input(shape=(4,)),
        # layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

# %% Fitting Model ----------------------------------------------------------------------------------------------------

start_time = time.time()
print('started fitting model:', start_time)

# only saves the best performer seen so far after each epoch 
checkpoint_filepath = savefolder + '/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor='val_loss',
                                                            mode='max',
                                                            save_best_only=True)
history = model.fit(X_train, y_train, epochs=200, 
                    verbose = 2,
                    validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.savefig(savefolder + '/Plot:val_loss.pdf')
plt.show()
plt.close()

# saving the "best" model

bestmodel = keras.models.load_model(checkpoint_filepath)
Ws, bs, acts = [], [], []
for layer in bestmodel.layers:
    W, b = layer.get_weights()
    act  = layer.get_config()['activation']
    Ws.append(W)
    bs.append(b)
    acts.append(act)
bestmodel.save(savefolder + '/qRW_NN.keras')
# Note that numpy cannot save inhomogeneous shaped array
#      therefore we use pickle dump
with open(savefolder + '/qRW_NN_Ws.pkl',   'wb') as file: pickle.dump(Ws,   file)
with open(savefolder + '/qRW_NN_bs.pkl',   'wb') as file: pickle.dump(bs,   file)
with open(savefolder + '/qRW_NN_acts.pkl', 'wb') as file: pickle.dump(acts, file)

end_time = time.time()
print('training took:', round(end_time - start_time, 3))

# %% Make example qRW plots

ps = np.linspace(0.9, 0.999, 100)
tasks = np.array([[p, 0.5, 0.5, 1] for p in ps])

plt.plot(ps, qRW(ps, 0.5, 0.5, 1), label = 'numerical integral')
plt.plot(ps, np.exp(bestmodel.predict(tasks, verbose = 0).ravel()), label = 'NN')
plt.legend(loc = 'upper left')
plt.xlabel('p')
plt.ylabel('quantile')
plt.xticks(np.linspace(0.9, 0.999, 5))
plt.title(r'qRW(...) along p with $\phi$=0.5 $\gamma$=0.5 $\tau$=1.0')
plt.savefig(savefolder+'/Plot:qRW.pdf')
plt.show()
plt.close()

# %%
