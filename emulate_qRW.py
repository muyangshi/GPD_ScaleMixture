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

# pRW(1e16, 1, 4, 50) yields array(0.99999999)

p_samples = 2 - np.geomspace(1-1e-3, 2-0.8, 25)[::-1]
phi_samples = np.linspace(0.05, 1, 10)
gamma_samples = np.linspace(0.1, 4, 10)
tau_samples = np.linspace(1e-3, 50, 10)


# 14 seconds for total of 100000

P, Phi, Gamma, Tau = np.meshgrid(p_samples, phi_samples, gamma_samples, tau_samples, indexing='ij')

P_flat     = P.ravel()
Phi_flat   = Phi.ravel()
Gamma_flat = Gamma.ravel()
Tau_flat   = Tau.ravel()
inputs     = np.column_stack((P_flat, Phi_flat, Gamma_flat, Tau_flat))

# x_samples = qRW(inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3])


def qRW_par(args):
    p, phi, gamma, tau = args
    return(qRW(p, phi, gamma, tau))

with multiprocessing.get_context('fork').Pool(processes=4) as pool:
    x_samples = pool.map(qRW_par, list(inputs))
x_samples = np.array(x_samples)


# from multiprocessing import get_context
# p = get_context("fork").Pool(4)
# results = p.map(pRW_par, inputs)
# p.close()


# %%
# Define Keras model
model = keras.Sequential(
    [   
        keras.Input(shape=(4,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
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
y_train = x_samples[train_indices]
y_val   = x_samples[test_indices]

y_train = np.log(y_train)
y_val = np.log(y_val)


# %%
# Fitting Model
model.fit(X_train, y_train, epochs= 100, validation_data=(X_val, y_val))

# %%
def qRW_NN(x, phi, gamma, tau):
    return model.predict(np.array([[x, phi, gamma, tau]]), verbose=0)[0]
qRW_NN_vec = np.vectorize(qRW_NN)

# %%
p = np.linspace(0.8, 0.999, 100)
plt.plot(p, qRW(p, 0.5, 0.5, 1), label = 'incomplete gamma')
plt.plot(p, np.exp(qRW_NN_vec(p, 0.5, 0.5, 1)), label = 'NN')
plt.legend(loc = 'upper left')
# %%
