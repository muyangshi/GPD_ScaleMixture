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

## Likelihoods

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