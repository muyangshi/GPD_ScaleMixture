"""
Calculate Design points for qRW()
Create a Keras-Tensorflow Neural Network emulating the qRW(p, phi, gamma, tau) function
Save to [savefolder]: data/qRW_p#_phi#_gamma#_tau#/
    - the grid of design points
    - .keras model
"""
# %% imports
# base python
import os
import time
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

# %% set up and helper functions

# Note pRW(1e16, 1, 4, 50) yields array(0.99999999)
# p
lb_p = 0.8
ub_p = 0.999
n_p  = 100

# phi
lb_phi = 0.05
ub_phi = 0.99
n_phi  = 20

# gamma
lb_gamma = 0.1
ub_gamma = 4
n_gamma  = 10

# tau
lb_tau = 0.1
ub_tau = 50
n_tau  = 20

savefolder = '../data/qRW'    +                \
                     '_p'     + str(n_p)     + \
                     '_phi'   + str(n_phi)   + \
                     '_gamma' + str(n_gamma) + \
                     '_tau'   + str(n_tau)
Path(savefolder).mkdir(parents=True, exist_ok=True)

def qRW_par(args): # wrapper to put qRW for multiprocessing
    p, phi, gamma, tau = args
    return(qRW(p, phi, gamma, tau))

# This is incredibly slow -- much better to directly pass X_2d to model.predict
# def qRW_NN(x, phi, gamma, tau):
#     return model.predict(np.array([[x, phi, gamma, tau]]), verbose=0)[0]
# qRW_NN_vec = np.vectorize(qRW_NN)


# %% Generate Design Points

p_samples     = 2 - np.geomspace(2-ub_p, 2-lb_p, n_p)[::-1]
phi_samples   = np.linspace(lb_phi, ub_phi, n_phi)
gamma_samples = np.linspace(lb_gamma, ub_gamma, n_gamma)
tau_samples   = np.linspace(lb_tau, ub_tau, n_tau)

P, Phi, Gamma, Tau = np.meshgrid(p_samples, phi_samples, gamma_samples, tau_samples, indexing='ij')
P_flat             = P.ravel()
Phi_flat           = Phi.ravel()
Gamma_flat         = Gamma.ravel()
Tau_flat           = Tau.ravel()
inputs             = np.column_stack((P_flat, Phi_flat, Gamma_flat, Tau_flat))

# x_samples = qRW(inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3])

with multiprocessing.get_context('fork').Pool(processes=30) as pool:
    x_samples = pool.map(qRW_par, list(inputs))
x_samples = np.array(x_samples)

# 400,000 qRW() evals in 5 minutes, 30 processes

# from multiprocessing import get_context
# p = get_context("fork").Pool(4)
# results = p.map(pRW_par, inputs)
# p.close()

print('done')
np.save(savefolder + 'inputs',    inputs)
np.save(savefolder + 'x_samples', x_samples)


# %% Define Keras model

model = keras.Sequential(
    [   
        keras.Input(shape=(4,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

# %% Spliting training and validation
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

# %% Fitting Model

history = model.fit(X_train, y_train, epochs= 100, validation_data=(X_val, y_val))

Ws, bs, acts = [], [], []
for layer in model.layers:
    W, b = layer.get_weights()
    act  = layer.get_config()['activation']
    Ws.append(W)
    bs.append(b)
    acts.append(act)

model.save(savefolder + 'qRW_NN.keras')
np.save(savefolder + 'qRW_NN_Ws',   Ws)
np.save(savefolder + 'qRW_NN_bs',   bs)
np.save(savefolder + 'qRW_NN_acts', acts)

plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.show()
plt.savefig(savefolder + 'Plot:val_loss.pdf')
plt.close()


# %% Make example qRW plots

ps = np.linspace(0.9, 0.999, 100)
tasks = np.array([[p, 0.5, 0.5, 1] for p in ps])

plt.plot(ps, qRW(ps, 0.5, 0.5, 1), label = 'numerical integral')
plt.plot(ps, np.exp(model.predict(tasks, verbose = 0).ravel()), label = 'NN')
plt.legend(loc = 'upper left')
plt.xlabel('p')
plt.ylabel('quantile')
plt.title(r'qRW(...) along p with $\phi$=0.5 $\gamma$=0.5 $\tau$=1.0')
plt.show()
plt.savefig(savefolder+'Plot:qRW.pdf')
plt.close()

# %%
