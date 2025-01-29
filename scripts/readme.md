This note documents how the design points (X and Y) are generated.

## Quantiles

Files:
- `X_new_sub` (200 design points)
- `Y_new_sub` (the corresponding 200 quantiles from the `X_new_sub`)

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

Files:
- `qRW_X_100000000.npy` (100 million design points)
- `qRW_Y_100000000.npy` (the corresponding 100 million quantiles from the design points)
- `qRW_X_1000000.npy` (1 million design points)
- `qRW_Y_1000000.npy` (the corresponding 1 million quantiles from the design points)

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

TBD