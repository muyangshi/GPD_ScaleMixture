"""
April 24, 2024
Take a simulated dataset
Plot the ll marginally against each parameter
Use as a benchmark for future emulation objects

April 25, 2024
A tensorflow session cannot be shared across processes, 
each child process needs to be in charge of its own keras environment:
    - (via local imports)
    - tensorflow session
    - loaded model
A key is to never import Tensorflow in the main process
i.e. don't do import Keras within the main document, only under a function
see: 
    - https://stackoverflow.com/questions/74540699/keras-multiprocessing-model-prediction
    - https://github.com/keras-team/keras/issues/9964
"""
# %% imports
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from utilities import *
import pickle
from time import strftime, localtime
import time
import multiprocessing
from pathlib import Path
# import keras

# model = keras.models.load_model("qRW_100_20_10_20.keras")

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)
    
def ll_1t_par(args):
    Y_1t, p, u_vec, Scale_vec, Shape_vec,                   \
    R_vec, Z_1t, phi_vec, gamma_vec, tau,                   \
    X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,        \
    K = args

    if X_1t is None:
        X_1t      = qRW(pCGP(Y_1t, p, u_vec, Scale_vec, Shape_vec), phi_vec, gamma_vec, tau)
    if X_star_1t is None:
        X_star_1t = (R_vec ** phi_vec) * g(Z_1t)
    
    dX_1t = dRW(X_1t, phi_vec, gamma_vec, tau)
    
    censored_ll_1t = Y_censored_ll_1t(Y_1t, p, u_vec, Scale_vec, Shape_vec,
                                        R_vec, Z_1t, phi_vec, gamma_vec, tau,
                                        X_1t, X_star_1t, dX_1t, censored_idx_1t, exceed_idx_1t)
    gaussian_joint = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

    return censored_ll_1t + gaussian_joint

def ll_1t_par_NN(args):
    Y_1t, p, u_vec, Scale_vec, Shape_vec,                   \
    R_vec, Z_1t, phi_vec, gamma_vec, tau,                   \
    X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,        \
    K = args

    Ns = phi_vec.shape[0]

    import keras
    model = keras.models.load_model("../data/qRW_100_20_10_20/qRW_100_20_10_20.keras")

    if X_1t is None:
        coord = np.column_stack((pCGP(Y_1t, p, u_vec, Scale_vec, Shape_vec), 
                                    phi_vec,
                                    gamma_vec,
                                    np.full((Ns,), tau)))
        X_1t  = np.exp(model.predict(coord, verbose = 0).ravel())
    if X_star_1t is None:
        X_star_1t = (R_vec ** phi_vec) * g(Z_1t)
    
    dX_1t = dRW(X_1t, phi_vec, gamma_vec, tau)
    
    censored_ll_1t = Y_censored_ll_1t(Y_1t, p, u_vec, Scale_vec, Shape_vec,
                                        R_vec, Z_1t, phi_vec, gamma_vec, tau,
                                        X_1t, X_star_1t, dX_1t, censored_idx_1t, exceed_idx_1t)
    gaussian_joint = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

    return censored_ll_1t + gaussian_joint

def ll_1t_par_NN_mod(args):
    Y_1t, p, u_vec, Scale_vec, Shape_vec,                   \
    R_vec, Z_1t, phi_vec, gamma_vec, tau,                   \
    X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,        \
    K = args

    Ns = phi_vec.shape[0]

    import keras
    model = keras.models.load_model("qRW_100_20_10_20.keras")

    if X_1t is None:
        coord = np.column_stack((pCGP(Y_1t, p, u_vec, Scale_vec, Shape_vec), 
                                    phi_vec,
                                    gamma_vec,
                                    np.full((Ns,), tau)))
        X_1t  = np.where(coord[:,0] > 0.999, qRW(coord[:,0], coord[:,1], coord[:,2], coord[:,3]),
                                                np.exp(model.predict(coord, verbose = 0).ravel()))
    if X_star_1t is None:
        X_star_1t = (R_vec ** phi_vec) * g(Z_1t)
    
    dX_1t = dRW(X_1t, phi_vec, gamma_vec, tau)
    
    censored_ll_1t = Y_censored_ll_1t(Y_1t, p, u_vec, Scale_vec, Shape_vec,
                                        R_vec, Z_1t, phi_vec, gamma_vec, tau,
                                        X_1t, X_star_1t, dX_1t, censored_idx_1t, exceed_idx_1t)
    gaussian_joint = scipy.stats.multivariate_normal.logpdf(Z_1t, mean = None, cov = K)

    return censored_ll_1t + gaussian_joint

# %% Configuration

data_seed    = 2345
Ns           = 500
Nt           = 32
p            = 0.9 # theshold proability (across space time)
radius       = 2
bandwidth    = radius**2/6
N_outer_grid = 16
phi_truth    = 0.3
rho_truth    = 1.0
datafolder   = '../data/stationary_seed'+str(data_seed)+\
                            '_t'+str(Nt)+'_s'+str(Ns)+\
                            '_phi'+str(phi_truth)+\
                            '_rho'+str(rho_truth)+'/'
resultfolder = datafolder+'RESULTS/'
Path(resultfolder).mkdir(parents=True, exist_ok=True)

# %% Calculate and save the likelihood
if __name__ == "__main__":

    # %% Setup

    # Load Simulated Dataset ---------------------------------------------------------------------------------------
    Y                  = np.load(datafolder + 'Y.npy')
    Z                  = np.load(datafolder + 'Z.npy')
    X                  = np.load(datafolder + 'X.npy')
    X_star             = np.load(datafolder + 'X_star.npy')
    S_at_knots         = np.load(datafolder + 'S_at_knots.npy')
    logsigma_matrix    = np.load(datafolder + 'logsigma_matrix.npy')
    ksi_matrix         = np.load(datafolder + 'ksi_matrix.npy')
    stations           = np.load(datafolder + 'sites_xy.npy')
    elevations         = np.load(datafolder + 'elevations.npy')
    
    # Sites -----------------------------------------------------------------------------------------------------------
    
    sites_xy = stations
    sites_x = sites_xy[:,0]
    sites_y = sites_xy[:,1]

    # Knots -----------------------------------------------------------------------------------------------------------

    # define the lower and upper limits for x and y
    minX, maxX = 0.0, 10.0
    minY, maxY = 0.0, 10.0
    # isometric knot grid
    h_dist_between_knots     = (maxX - minX) / (int(2*np.sqrt(N_outer_grid))-1)
    v_dist_between_knots     = (maxY - minY) / (int(2*np.sqrt(N_outer_grid))-1)
    x_pos                    = np.linspace(minX + h_dist_between_knots/2, maxX + h_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    y_pos                    = np.linspace(minY + v_dist_between_knots/2, maxY + v_dist_between_knots/2, 
                                           num = int(2*np.sqrt(N_outer_grid)))
    x_outer_pos              = x_pos[0::2]
    x_inner_pos              = x_pos[1::2]
    y_outer_pos              = y_pos[0::2]
    y_inner_pos              = y_pos[1::2]
    X_outer_pos, Y_outer_pos = np.meshgrid(x_outer_pos, y_outer_pos)
    X_inner_pos, Y_inner_pos = np.meshgrid(x_inner_pos, y_inner_pos)
    knots_outer_xy           = np.vstack([X_outer_pos.ravel(), Y_outer_pos.ravel()]).T
    knots_inner_xy           = np.vstack([X_inner_pos.ravel(), Y_inner_pos.ravel()]).T
    knots_xy                 = np.vstack((knots_outer_xy, knots_inner_xy))
    knots_id_in_domain       = [row for row in range(len(knots_xy)) if (minX < knots_xy[row,0] < maxX and minY < knots_xy[row,1] < maxY)]
    knots_xy                 = knots_xy[knots_id_in_domain]
    knots_x                  = knots_xy[:,0]
    knots_y                  = knots_xy[:,1]
    k                        = len(knots_id_in_domain)

    # Copula  ---------------------------------------------------------------------------------------------------------
    
    radius_from_knots = np.repeat(radius, k) # influence radius from a knot

    # Generate the weight matrices
    # Weight matrix generated using Gaussian Smoothing Kernel
    gaussian_weight_matrix = np.full(shape = (Ns, k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = weights_fun(d_from_knots, radius, bandwidth, cutoff = False)
        gaussian_weight_matrix[site_id, :] = weight_from_knots

    # Weight matrix generated using wendland basis
    wendland_weight_matrix = np.full(shape = (Ns,k), fill_value = np.nan)
    for site_id in np.arange(Ns):
        # Compute distance between each pair of the two collections of inputs
        d_from_knots = scipy.spatial.distance.cdist(XA = sites_xy[site_id,:].reshape((-1,2)), 
                                        XB = knots_xy)
        # influence coming from each of the knots
        weight_from_knots = wendland_weights_fun(d_from_knots, radius_from_knots)
        wendland_weight_matrix[site_id, :] = weight_from_knots
    
    # Covariance K for Gaussian Field g(Z) 
    nu = 0.5 # exponential kernel for matern with nu = 1/2
    sigsq = 1.0 # sill for Z
    sigsq_vec = np.repeat(sigsq, Ns) # hold at 1

    # Scale Mixture R^phi
    gamma = 0.5 # this is the gamma that goes in rlevy, gamma_at_knots
    delta = 0.0 # this is the delta in levy, stays 0
    alpha = 0.5
    gamma_at_knots = np.repeat(gamma, k)
    gamma_vec = np.sum(np.multiply(wendland_weight_matrix, gamma_at_knots)**(alpha), 
                       axis = 1)**(1/alpha) # bar{gamma}, axis = 1 to sum over K knots
    
    phi_at_knots   = np.array([phi_truth] * k)
    range_at_knots = np.array([rho_truth] * k)

    # Marginal --------------------------------------------------------------------------------------------------------

    # threshold probability and quantile
    u_matrix = np.full(shape = (Ns, Nt), fill_value = np.nanquantile(Y, p)) # threshold u on Y, i.e. p = Pr(Y <= u)

    scale_matrix = np.exp(logsigma_matrix)
    shape_matrix = ksi_matrix.copy()


    # %% Likelihood ---------------------------------------------------------------------------------------------------


    # %%
    # Truth -----------------------------------------------------------------------------------------------------------
    args_list = []
    for t in range(Nt):
        # marginal process
        Y_1t      = Y[:,t]
        p         = 0.9
        u_vec     = u_matrix[:,t]
        Scale_vec = scale_matrix[:,t]
        Shape_vec = shape_matrix[:,t]

        censored_idx_1t = np.where(Y_1t <= u_vec)[0]
        exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

        # copula process
        R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
        Z_1t      = Z[:,t]
        phi_vec   = gaussian_weight_matrix @ phi_at_knots
        tau       = 10
        range_vec = gaussian_weight_matrix @ range_at_knots
        K         = ns_cov(range_vec = range_vec,
                           sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

        X_1t      = X[:,t]      # if not changing, we place it here
        X_star_1t = X_star[:,t] # if changing, place under ll_1t_par, so qRW calculated in parallel

        args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                          R_vec, Z_1t, phi_vec, gamma_vec, tau,
                          X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                          K))

    with multiprocessing.Pool(processes = Nt) as pool:
        results = pool.map(ll_1t_par, args_list)

    with multiprocessing.Pool(processes = Nt) as pool:
        results_NN = pool.map(ll_1t_par_NN, args_list)

    # %%
    # phi -------------------------------------------------------------------------------------------------------------
    lb = 0.1
    ub = 0.8
    grids = 29
    phi_grid = np.linspace(lb, ub, grids)
    ll_phi = []
    for phi_x in phi_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = scale_matrix[:,t]
            Shape_vec = shape_matrix[:,t]

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ np.array([phi_x] * k)
            tau       = 10
            range_vec = gaussian_weight_matrix @ range_at_knots
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_phi.append(sum(results))

    print(ll_phi)
    plt.plot(np.linspace(lb, ub, grids), ll_phi, 'b.-')
    plt.yscale('symlog')
    plt.title(r'marginall loglike against $\phi$')
    plt.xlabel(r'$\phi$')
    plt.ylabel('log likelihood')
    plt.savefig(resultfolder+'ll_phi.pdf')
    np.save(resultfolder+'ll_phi', ll_phi)

    # %%
    # phi with qRW Newral Network emulated ----------------------------------------------------------------------------
    lb = 0.1
    ub = 0.8
    grids = 8
    phi_grid = np.linspace(lb, ub, grids)
    ll_phi_NNqRW = []
    for phi_x in phi_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = scale_matrix[:,t]
            Shape_vec = shape_matrix[:,t]

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ np.array([phi_x] * k)
            tau       = 10
            range_vec = gaussian_weight_matrix @ range_at_knots
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            # coord = np.column_stack((pCGP(Y_1t, p, u_vec, Scale_vec, Shape_vec), 
            #                 phi_vec,
            #                 gamma_vec,
            #                 np.full((Ns,), tau)))
            # X_1t  = np.exp(model.predict(coord, verbose = 0).ravel())
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par_NN_mod, args_list)
        ll_phi_NNqRW.append(sum(results))

    print(ll_phi_NNqRW)
    plt.plot(np.linspace(lb, ub, grids), ll_phi_NNqRW, 'b.-')
    plt.yscale('symlog')
    plt.savefig(resultfolder+'ll_phi_NNqRW.pdf')
    np.save(resultfolder+'ll_phi_NNqRW', ll_phi_NNqRW)

    # 50s    for ll_1t_par_NN
    # 2m     for ll_1t_par
    # 2m 26s for ll_1t_par_NN_mod

    # %%
    # rho -------------------------------------------------------------------------------------------------------------
    lb = 0.5
    ub = 1.5
    grids = 5
    rho_grid = np.linspace(lb, ub, grids)
    ll_rho = []
    for rho_x in rho_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = scale_matrix[:,t]
            Shape_vec = shape_matrix[:,t]

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ phi_at_knots
            tau       = 10
            range_vec = gaussian_weight_matrix @ np.array([rho_x] * k)
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_rho.append(sum(results))

    print(ll_rho)
    plt.plot(np.linspace(lb, ub, grids), ll_rho, 'b.-')
    plt.savefig(resultfolder+'ll_rho.pdf')
    np.save(resultfolder+'ll_rho', ll_rho)

    # %%
    # tau -------------------------------------------------------------------------------------------------------------
    lb = 8
    ub = 12
    grids = 5
    tau_grid = np.linspace(lb, ub, grids)
    ll_tau = []
    for tau_x in tau_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = scale_matrix[:,t]
            Shape_vec = shape_matrix[:,t]

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ phi_at_knots
            tau       = tau_x
            range_vec = gaussian_weight_matrix @ range_at_knots
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_tau.append(sum(results))

    print(ll_tau)
    plt.plot(np.linspace(lb, ub, grids), ll_tau, 'b.-')
    plt.savefig(resultfolder+'ll_tau.pdf')
    np.save(resultfolder+'ll_tau', ll_tau)


    # %%
    # scale -----------------------------------------------------------------------------------------------------------
    lb = 0.8
    ub = 1.2
    grids = 5
    scale_grid = np.linspace(lb, ub, grids)
    ll_scale = []
    for scale_x in scale_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = np.full((Ns,), scale_x)
            Shape_vec = shape_matrix[:,t]

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ phi_at_knots
            tau       = 10
            range_vec = gaussian_weight_matrix @ range_at_knots
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_scale.append(sum(results))

    print(ll_scale)
    plt.plot(np.linspace(lb, ub, grids), ll_scale, 'b.-')
    plt.savefig(resultfolder+'ll_scale.pdf')
    np.save(resultfolder+'ll_scale', ll_scale)


    # %%
    # shape -----------------------------------------------------------------------------------------------------------
    lb = 0.2
    ub = 0.3
    grids = 5
    shape_grid = np.linspace(lb, ub, grids)
    ll_shape = []
    for shape_x in shape_grid:
        args_list = []
        for t in range(Nt):
            # marginal process
            Y_1t      = Y[:,t]
            p         = 0.9
            u_vec     = u_matrix[:,t]
            Scale_vec = scale_matrix[:,t]
            Shape_vec = np.full((Ns,), shape_x)

            censored_idx_1t = np.where(Y_1t <= u_vec)[0]
            exceed_idx_1t   = np.where(Y_1t  > u_vec)[0]

            # copula process
            R_vec     = wendland_weight_matrix @ S_at_knots[:,t]
            Z_1t      = Z[:,t]
            phi_vec   = gaussian_weight_matrix @ phi_at_knots
            tau       = 10
            range_vec = gaussian_weight_matrix @ range_at_knots
            K         = ns_cov(range_vec = range_vec,
                            sigsq_vec = sigsq_vec, coords = sites_xy, kappa = nu, cov_model = 'matern')

            X_1t      = None
            X_star_1t = None

            args_list.append((Y_1t, p, u_vec, Scale_vec, Shape_vec,
                            R_vec, Z_1t, phi_vec, gamma_vec, tau,
                            X_1t, X_star_1t, censored_idx_1t, exceed_idx_1t,
                            K))

        with multiprocessing.Pool(processes = Nt) as pool:
            results = pool.map(ll_1t_par, args_list)
        ll_shape.append(sum(results))

    print(ll_shape)
    plt.plot(np.linspace(lb, ub, grids), ll_shape, 'b.-')
    plt.yscale('symlog')
    plt.savefig(resultfolder+'ll_shape.pdf')
    np.save(resultfolder+'ll_shape', ll_shape)

# %%
