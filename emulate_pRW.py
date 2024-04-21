# %%
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from utilities import *
import multiprocessing
# Note on Arm Macs, default is spawn. get_context('fork') to use on laptop

import keras
from keras import layers


# %%
# Generate Design Points

# pRW(1e16, 1, 0.5, 50) yields array(0.99999999)
"""
Plan to emulate 
X: (-tau * (?), 1e16)
phi: (0, 1)
gamma: (0 <- won't be smaller than 0.5(?), 4)
tau: (0 <- causes trouble. use small value, 50)
"""

x_samples = np.linspace(0, 1e4, 1000)
phi_samples = np.linspace(0.05, 1, 10)
gamma_samples = np.linspace(0.1, 4, 10)
tau_samples = np.linspace(1e-3, 50, 10)
# 14 seconds for total of 100000

X, Phi, Gamma, Tau = np.meshgrid(x_samples, phi_samples, gamma_samples, tau_samples, indexing='ij')
X_flat = X.ravel()
Phi_flat = Phi.ravel()
Gamma_flat = Gamma.ravel()
Tau_flat = Tau.ravel()

inputs = np.column_stack((X_flat, Phi_flat, Gamma_flat, Tau_flat))

# p_samples = pRW(inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3])


def pRW_par(args):
    x, phi, gamma, tau = args
    return(pRW(x, phi, gamma, tau))

with multiprocessing.get_context('fork').Pool(processes=4) as pool:
    p_samples = pool.map(pRW_par, list(inputs))
p_samples = np.array(p_samples)


# from multiprocessing import get_context
# p = get_context("fork").Pool(4)
# results = p.map(pRW_par, inputs)
# p.close()


# %%
# Define Keras model
model = keras.Sequential(
    [   
        keras.Input(shape=(4,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Spliting training and validation
train_size    = 0.8
indices       = np.arange(inputs.shape[0])
np.random.shuffle(indices)
split_idx     = int(inputs.shape[0] * train_size)
train_indices = indices[:split_idx]
test_indices  = indices[split_idx:]

X_train = inputs[train_indices]
X_val   = inputs[test_indices]
y_train = p_samples[train_indices]
y_val   = p_samples[test_indices]


# %%
# Fitting Model
model.fit(X_train, y_train, epochs= 100, validation_data=(X_val, y_val))

# %%
def pRW_NN(x, phi, gamma, tau):
    return model.predict(np.array([[x, phi, gamma, tau]]), verbose=0)[0]
pRW_NN_vec = np.vectorize(pRW_NN)

# %%
x = np.linspace(0, 100, 2000)
plt.plot(x, pRW(x, 0.5, 0.5, 1), label = 'incomplete gamma')
plt.plot(x, pRW_NN_vec(x, 0.5, 0.5, 1), label = 'NN')
plt.legend(loc = 'upper left')
# %%
