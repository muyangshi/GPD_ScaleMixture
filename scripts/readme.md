This note documents how the design points (X and Y) are generated.

## Quantiles

**Files**:
- `X_new_sub` (200 design points)
- `Y_new_sub` (the corresponding 200 quantiles from the `X_new_sub`)

**Generating Code**:

```
x1        = np.linspace(0.9, 0.999, num=200, dtype=np.float32)
x2        = np.full(shape=(200,), fill_value=0.6, dtype=np.float32)
x3        = np.full(shape=(200,), fill_value=1.0, dtype=np.float32)
x4        = np.full(shape=(200,), fill_value=2.0, dtype=np.float32)
X_new_sub = np.column_stack([x1, x2, x3, x4])

Y_new_sub = qRW(x1, x2, x3, x4)

np.save('Y_new_sub.npy', Y_new_sub)
np.save('X_new_sub.npy', X_new_sub)
```

**Files**:

- `qRW_X_100000000.npy` (100 million design points)
- `qRW_Y_100000000.npy` (the corresponding 100 million quantiles from the design points)
- `qRW_X_1000000.npy` (1 million design points)
- `qRW_Y_1000000.npy` (the corresponding 1 million quantiles from the design points)

**Generating Code**:

```
N # <- the number of design points

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


def qRW_par(args): # wrapper to put qRW for multiprocessing
    p, phi, gamma, tau = args
    return(qRW(p, phi, gamma, tau))

with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    Y_lhs = pool.map(qRW_par, list(X_lhs))
Y_lhs = np.array(Y_lhs)

np.save(rf'qRW_X_{N}.npy', X_lhs)
np.save(rf'qRW_Y_{N}.npy', Y_lhs)
```

## Log Likelihoods (Archived, because range on R is not sufficient)

**Files**:

- `ll_1t_X_lhs_100000000.npy` (100 million design points)
- `ll_1t_Y_lhs_100000000.npy` (the corresponding 100 million log-likelihoods to the design points)
- `ll_1t_X_lhs_val_1000000.npy` (1 million design points)
- `ll_1t_Y_lhs_val_1000000.npy` (the corresponding 1 million log-likelihoods to the design points)

**Generating Code**:

```
N # <- the number of design points

sampler     = qmc.LatinHypercube(d, scramble=False, seed=2345)
lhs_samples = sampler.random(N) # Generate LHS samples in [0,1]^d
lhs_samples = np.row_stack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary points

#             pY,    u, scale, shape,   pR,    Z,  phi, gamma_bar,  tau
l_bounds = [0.001,   30,     5,  -1.0, 0.01, -5.0, 0.05,       0.5,  1.0]
u_bounds = [0.999,   80,    60,   1.0, 0.95,  5.0, 0.95,       8.0, 50.0]
X_lhs    = qmc.scale(lhs_samples, l_bounds, u_bounds)        # scale LHS to specified bounds

# Note that R is not levy(0, 0.5)
#   Check the values of gamma_bar, pick the largest, use that to span the sample space
#   Maybe (0, 8)?
X_lhs[:,4] = scipy.stats.levy(loc=0,scale=8.0).ppf(X_lhs[:,4]) # scale the Stables

# Y assumed to be Generalized Pareto, if pY > 0.9;
#   otherwise, just return the corresponding threshold u
X_lhs[:,0] = qCGP(X_lhs[:,0], 0.9, X_lhs[:,1], X_lhs[:,2], X_lhs[:,3])

# %% Calculate the log likelihoods at the design points

def Y_ll_1t(params): # dependence model parameters)
    """
    calculate the censoring likelihood of Y, at p = 0.9
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

data = [tuple(row) for row in X_lhs]
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    results = pool.map(Y_ll_1t, data)

# %% remove the NAs
noNA = np.where(~np.isnan(results))
Y_lhs = np.array(results)[noNA]
X_lhs = X_lhs[noNA]

np.save(rf'll_1t_X_lhs_{N}.npy', X_lhs)
np.save(rf'll_1t_Y_lhs_{N}.npy', Y_lhs)
```

##Log Likelihood Training and Validation Data

Files:
    - `ll_1t_X_100000000.npy`
    - `ll_1t_Y_100000000.npy` (log likelihood)
    - `ll_1t_X_val_1000000.npy`
    - `ll_1t_Y_val_1000000.npy` (log likelihood)

Generating Code:
```
# LHS for the design point X

sampler     = qmc.LatinHypercube(d, scramble=False, seed=2345)
lhs_samples = sampler.random(N) # Generate LHS samples in [0,1]^d
lhs_samples = np.row_stack(([0]*d, lhs_samples, [1]*d)) # manually add the boundary points

#             pY,    u, scale, shape,    pR,    Z,  phi, gamma_bar,  tau
l_bounds = [0.001,   30,     5,  -1.0, 1e-2, -5.0, 0.05,       0.5,  1.0]
u_bounds = [0.999,   80,    60,   1.0,  5e6,  5.0, 0.95,       8.0, 50.0]
X_lhs    = qmc.scale(lhs_samples, l_bounds, u_bounds)        # scale LHS to specified bounds

# Note that R is not levy(0, 0.5)
#   Check the values of gamma_bar, pick the largest, use that to span the sample space
#   Maybe (0, 8)?
# X_lhs[:,4] = scipy.stats.levy(loc=0,scale=8.0).ppf(X_lhs[:,4]) # scale the Stables

# Y assumed to be Generalized Pareto, if pY > 0.9;
#   otherwise, just return the corresponding threshold u
X_lhs[:,0] = qCGP(X_lhs[:,0], 0.9, X_lhs[:,1], X_lhs[:,2], X_lhs[:,3])

print('X_lhs.shape:',X_lhs.shape)

# %% Calculate the likelihoods of Y at the design points

data = [tuple(row) for row in X_lhs]

start_time = time.time()
print(rf'start calculating {N} likelihoods using {n_processes} processes:', datetime.datetime.now())
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    results = pool.map(Y_ll_1t1s_par, data)
end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

# remove the NAs
noNA = np.where(~np.isnan(results))
Y_lhs = np.array(results)[noNA]
X_lhs = X_lhs[noNA]

print('len(Y_lhs):',len(Y_lhs))   # number of design points retained
print('proportion not NA:', len(Y_lhs)/N) # proportion of design points retained

np.save(rf'll_1t_X_{N}.npy', X_lhs)
np.save(rf'll_1t_Y_{N}.npy', Y_lhs)

# %% Generate a set of dedicated validation points

sampler_val     = qmc.LatinHypercube(d, scramble=False, seed=129)
lhs_samples_val = sampler_val.random(N_val) # Generate LHS samples in [0,1]^d
lhs_samples_val = np.row_stack(([0]*d, lhs_samples_val, [1]*d)) # add boundaries
#                    pY,    u, scale, shape,   pR,    Z,  phi, gamma_bar,  tau
l_bounds       = [0.001,   30,     5,  -1.0, 1e-2, -5.0, 0.05,       0.5,  1.0]
u_bounds       = [0.999,   80,    60,   1.0,  5e6,  5.0, 0.95,       8.0, 50.0]
X_lhs_val      = qmc.scale(lhs_samples_val, l_bounds, u_bounds)        # scale LHS to specified bounds
# X_lhs_val[:,4] = scipy.stats.levy(loc=0,scale=8.0).ppf(X_lhs_val[:,4]) # scale the Stables
X_lhs_val[:,0] = qCGP(X_lhs_val[:,0], 0.9, X_lhs_val[:,1], X_lhs_val[:,2], X_lhs_val[:,3])
print('X_lhs_val.shape:',X_lhs_val.shape)

data_val = [tuple(row) for row in X_lhs_val]
start_time = time.time()
print(rf'start calculating {N_val} likelihoods using {n_processes} processes:', datetime.datetime.now())
with multiprocessing.get_context('fork').Pool(processes=n_processes) as pool:
    results_val = pool.map(Y_ll_1t1s_par, data_val)
end_time = time.time()
print('done:', round(end_time - start_time, 3), 'using processes:', str(n_processes))

noNA      = np.where(~np.isnan(results_val))
Y_lhs_val = np.array(results_val)[noNA]
X_lhs_val = X_lhs_val[noNA]

np.save(rf'll_1t_X_val_{N_val}.npy', X_lhs_val)
np.save(rf'll_1t_Y_val_{N_val}.npy', Y_lhs_val)
```

## Marginal Log Likelihood over phi grid

Files:
- `phi_grid.npy` (0.2 to 0.8, plus truth of $\phi_0$) array([0.2      , 0.35     , 0.4064064, 0.5      , 0.65     , 0.8      ])
- `Y_ll_X_input_Nt_Ns_phi_grid_X.npy` (shaped (|phi_grid| = 6, Nt*Ns = 3000, d = 9))
- `Y_ll_Y_Nt_Ns_phi_grid.npy` (shaped (|phi_grid| = 6, Nt*Ns = 3000))

Generating Code:
```
i = 0
lb = 0.2
ub = 0.8
grids = 5 # fast
phi_grid = np.linspace(lb, ub, grids)
phi_grid = np.sort(np.insert(phi_grid, 0, phi_at_knots[i]))

Y_Nt_Ns_phi_grid       = []
X_input_Nt_Ns_phi_grid = []
S_ll_Nt_phi_grid       = []
Z_ll_Nt_phi_grid       = []

for phi_x in phi_grid:

    X_input_Nt_Ns = []
    S_ll_Nt       = []
    Z_ll_Nt       = []

    phi_k        = phi_at_knots.copy()
    phi_k[i]     = phi_x
    phi_vec_test = gaussian_weight_matrix_phi @ phi_k

    input_list = [] # used to calculate all the Y-likelihoods
    for t in range(Nt):
        Y_1t      = Y[:,t]
        u_vec     = u_matrix[:,t]
        Scale_vec = Scale_matrix[:,t]
        Shape_vec = Shape_matrix[:,t]
        R_vec     = wendland_weight_matrix_S @ S_at_knots[:,t]
        Z_1t      = Z[:,t]
        logS_vec  = np.log(S_at_knots[:,t])

        X_input = np.array([Y_1t, u_vec, Scale_vec, Shape_vec, R_vec, Z_1t, phi_vec_test, gamma_bar_vec, np.full_like(Y_1t, tau)]).T
        input_list.append(X_input)

        S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
        Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

        S_ll_Nt.append(np.sum(S_ll))
        Z_ll_Nt.append(np.sum(Z_ll))

    input_list = np.vstack(input_list)
    X_input_Nt_Ns_phi_grid.append(input_list)

    S_ll_Nt_phi_grid.append(S_ll_Nt)
    Z_ll_Nt_phi_grid.append(Z_ll_Nt)

np.array(X_input_Nt_Ns_phi_grid).shape
np.array(S_ll_Nt_phi_grid).shape
np.array(Z_ll_Nt_phi_grid).shape

for X_input in X_input_Nt_Ns_phi_grid:
    with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
        results = pool.starmap(Y_ll_1t1s, X_input)
    Y_Nt_Ns_phi_grid.append(np.array(results))

X_input_Nt_Ns_phi_grid = np.array(X_input_Nt_Ns_phi_grid)
S_ll_Nt_phi_grid       = np.array(S_ll_Nt_phi_grid)
Z_ll_Nt_phi_grid       = np.array(Z_ll_Nt_phi_grid)
Y_Nt_Ns_phi_grid       = np.array(Y_Nt_Ns_phi_grid)

plt.plot(phi_grid, np.sum(S_ll_Nt_phi_grid, axis = 1))
plt.plot(phi_grid, np.sum(Z_ll_Nt_phi_grid, axis = 1))
plt.plot(phi_grid, np.sum(Y_Nt_Ns_phi_grid, axis = 1))

plt.plot(phi_grid, np.sum(Y_Nt_Ns_phi_grid, axis = 1))

plt.plot(phi_grid,
         np.sum(Y_Nt_Ns_phi_grid, axis = 1) + np.sum(Z_ll_Nt_phi_grid, axis = 1) + np.sum(S_ll_Nt_phi_grid, axis = 1))

np.save('Y_ll_X_input_Nt_Ns_phi_grid.npy', X_input_Nt_Ns_phi_grid)
np.save('Y_ll_Y_Nt_Ns_phi_grid.npy', Y_Nt_Ns_phi_grid)
```

## Marginal Log Likelihood at the truth

Files:
    - `Y_ll_X_input_Nt_Ns.npy'
    - `Y_ll_Y_Nt_Ns.npy`

Generating Code:
```
S_ll_Nt       = []
Z_ll_Nt       = []

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

    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_k_vec) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

    S_ll_Nt.append(np.sum(S_ll))
    Z_ll_Nt.append(np.sum(Z_ll))

X_input_Nt_Ns = np.vstack(input_list)

with multiprocessing.get_context('fork').Pool(processes = n_processes) as pool:
    results = pool.starmap(Y_ll_1t1s, X_input_Nt_Ns)
Y_Nt_Ns = np.array(results)

np.save('Y_ll_X_input_Nt_Ns.npy', X_input_Nt_Ns)
np.save('Y_ll_Y_Nt_Ns.npy', Y_Nt_Ns)
```