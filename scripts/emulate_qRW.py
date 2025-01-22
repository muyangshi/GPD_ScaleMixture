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

# %% Step 1: LHS design for the parameter of qRW(p, phi, gamma, tau)

lhs_sampler = qmc.LatinHypercube(d, scramble = False, seed = 1031)
lhs_samples = lhs_sampler.random(N) # doesn't include the boundary
lhs_samples = np.row_stack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary

"""
reasoning for choosing the bounds:
    - gamma_bar: we chose gamma_k as 0.5, gamma_bar won't be smaller than this
    - upper bound on tau is large because tau is heavily over-estimated
"""
#             p,     phi, gamma_bar, tau
l_bounds = [0.9,    0.05,       0.5,   1]
u_bounds = [0.9999, 0.95,         5, 100] 

X_lhs = qmc.scale(lhs_samples, l_bounds, u_bounds)


# Calculate the design points

def qRW_par(args): # wrapper to put qRW for multiprocessing
    p, phi, gamma, tau = args
    return(qRW(p, phi, gamma, tau))

start_time = time.time()
print('start calculate qRW()s:', datetime.datetime.now())

with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    Y_lhs = pool.map(qRW_par, list(X_lhs))
Y_lhs = np.array(Y_lhs)

end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

np.save(rf'qRW_X_{N}.npy', X_lhs)
np.save(rf'qRW_Y_{N}.npy', Y_lhs)

# Caluclate a set of validation points

lhs_sampler_val = qmc.LatinHypercube(d, scramble = False, seed = 122)
lhs_samples_val = lhs_sampler_val.random(N_val) # doesn't include the boundary
lhs_samples_val = np.row_stack(([0]*d, lhs_samples_val, [1]*d)) # manually add the boundary
#              p,     phi, gamma_bar, tau
l_bounds  = [0.9,    0.05,       0.5,   1]
u_bounds  = [0.9999, 0.95,         5, 100] 
X_lhs_val = qmc.scale(lhs_samples_val, l_bounds, u_bounds)

start_time = time.time()
print('start calculating validation qRW()s:', datetime.datetime.now())
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    Y_lhs_val = pool.map(qRW_par, list(X_lhs_val))
Y_lhs_val = np.array(Y_lhs_val)
end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

np.save(rf'qRW_X_val_{N_val}.npy', X_lhs_val)
np.save(rf'qRW_Y_val_{N_val}.npy', Y_lhs_val)

# %% Step 2: Emulator

X_lhs     = np.load(rf'qRW_X_{N}.npy')
Y_lhs     = np.load(rf'qRW_Y_{N}.npy')
X_lhs_val = np.load(rf'qRW_X_val_{N_val}.npy')
Y_lhs_val = np.load(rf'qRW_Y_val_{N_val}.npy')

# Use part of the design points as training/validation

train_size    = 0.9
indices       = np.arange(X_lhs.shape[0])
np.random.shuffle(indices)

split_idx     = int(X_lhs.shape[0] * train_size)
train_indices = indices[:split_idx]
test_indices  = indices[split_idx:]

X_train       = X_lhs[train_indices]
X_val         = X_lhs[test_indices]

# # to train on log scale
# y_train       = np.log(Y_lhs[train_indices])
# y_val         = np.log(Y_lhs[test_indices])

# to train on original scale
y_train       = Y_lhs[train_indices]
y_val         = Y_lhs[test_indices]

# Use separate sets of training and validation points

X_train = X_lhs
X_val   = X_lhs_val

# to train on log scale
# y_train = np.log(Y_lhs)
# y_val   = np.log(Y_lhs_val)

# to train on original scale
y_train = Y_lhs
y_val   = Y_lhs_val

# Defining model

model = keras.Sequential(
    [   
        keras.Input(shape=(d,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ]
)

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(
    # optimizer='adam',
    optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule), 
    loss=keras.losses.mean_squared_error)

# %% # Fitting Model

start_time = time.time()
print('started fitting NN:', datetime.datetime.now())

checkpoint_filepath = './checkpoint.model.keras' # only saves the best performer seen so far after each epoch 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            monitor='val_loss',
                                                            mode='max',
                                                            save_best_only=True)
history = model.fit(
    X_train, 
    y_train, 
    epochs = 100, 
    verbose = 2,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint_callback])

end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.savefig('Plot_val_loss.pdf')
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

qRW_NN_keras = keras.models.load_model(rf'qRW_NN_{N}.keras')

def relu_np(x): # changes x IN PLACE! faster than return x * (x > 0)
    np.maximum(x, 0, x)

def identity(x):
    pass

Ws, bs, acts = [], [], []
for layer in qRW_NN_keras.layers:
    W, b = layer.get_weights()
    act  = relu_np if layer.get_config()['activation'] == 'relu' else identity
    Ws.append(W)
    bs.append(b)
    acts.append(act)

def qRW_NN(X, weights = Ws, biases = bs, activations = acts):
    Z = X.copy()
    for W, b, activation in zip(weights, biases, activations):
        Z = Z @ W + b
        activation(Z)
    return np.exp(Z)

# %% Step 4: Evaluation
# Make example qRW plots

ps    = np.linspace(0.9, 0.999, 100)
tasks = np.array([[p, 0.5, 0.5, 1] for p in ps])
plt.plot(ps, qRW(ps, 0.5, 0.5, 1), 'k.-', label = 'truth')
# plt.plot(ps, np.exp(bestmodel.predict(tasks, verbose = 0).ravel()), label = 'NN')
plt.plot(ps, qRW_NN(tasks, Ws, bs, acts), 'b.-', label = 'qRW NN')
plt.legend(loc = 'upper left')
plt.xlabel('p')
plt.ylabel('quantile')
plt.xticks(np.linspace(0.9, 0.999, 5))
plt.title(r'qRW(...) along p with $\phi$=0.5 $\gamma$=0.5 $\tau$=1.0')
plt.savefig('Plot_qRW.pdf')
plt.show()
plt.close()

# Marginal Likelihood
