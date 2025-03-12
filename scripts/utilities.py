"""
# combine utilitlies helpful to MCMC sampler
# grabbed and copied useful functions from Likun's model_sim.py, ns_cov.py
# Require:
#   - RW_inte.py, RW_inte_cpp.cpp & RW_inte.cpp.so
#   - qRW_NN_weights_and_biases.pkl, qRW_NN_X_min.npy, qRW_NN_X_max.npy
# March 5, 2025
# Use Neural Network to emulate qRW
    # Put dRW and qRW outside of the likelihood function, reduce the number of times they are involved
"""
# %% check dependencies

import os

required_files = [
    "RW_inte.py",
    "RW_inte_cpp.cpp",
    "RW_inte_cpp.so",
    "qRW_NN_weights_and_biases.pkl",
    "qRW_NN_X_min.npy",
    "qRW_NN_X_max.npy"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}. Please ensure all dependencies are available.")

# %%
# general imports and ubiquitous utilities
import sys
import numpy as np
import scipy
import scipy.special as sc
from scipy.spatial import distance
import pickle
import RW_inte

norm_pareto = 'standard'

def save_pickle_data(dataname,data_to_save):
    with open(dataname + '.pickle', 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def read_pickle_data(dataname):
    with open(dataname + '.pickle', 'rb') as f:
        read_data = pickle.load(f)
    f.close()
    return read_data

# %% spatial covariance functions copied from ns_cov
# spatial covariance functions copied from ns_cov

## -------------------------------------------------------------------------- ##
##               Implement the Matern correlation function (stationary)
## -------------------------------------------------------------------------- ##
def cov_spatial(r, cov_model = "exponential", cov_pars = np.array([1,1]), kappa = 0.5):
    # Input from a matrix of pairwise distances and a vector of parameters
    if type(r).__module__!='numpy' or isinstance(r, np.float64):
        r = np.array(r)
    if np.any(r<0):
        sys.exit('Distance argument must be nonnegative.')
    r[r == 0] = 1e-10
    
    if cov_model != "matern" and cov_model != "gaussian" and cov_model != "exponential" :
        sys.exit("Please specify a valid covariance model (matern, gaussian, or exponential).")
    
    if cov_model == "exponential":
        C = np.exp(-r)
    
    if cov_model == "gaussian" :
        C = np.exp(-(r^2))
  
    if cov_model == "matern" :
        range = 1
        nu = kappa
        part1 = 2 ** (1 - nu) / sc.gamma(nu)
        part2 = (r / range) ** nu
        part3 = sc.kv(nu, r / range)
        C = part1 * part2 * part3
    return C
## -------------------------------------------------------------------------- ##

## -------------------------------------------------------------------------- ##
##               Calculate a locally isotropic spatial covariance
## -------------------------------------------------------------------------- ##
def ns_cov(range_vec, sigsq_vec, coords, kappa = 0.5, cov_model = "matern"):
    ## Arguments:
    ##    range_vec = N-vector of range parameters (one for each location) 
    ##    sigsq_vec = N-vector of marginal variance parameters (one for each location)
    ##    coords = N x 2 matrix of coordinates
    ##    cov.model = "matern" --> underlying covariance model: "gaussian", "exponential", or "matern"
    ##    kappa = 0.5 --> Matern smoothness, scalar
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
        range_vec = np.array(range_vec)
        sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N: 
        sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) --- 
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones  
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones 
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = cov_spatial(Dist_mat, cov_model = cov_model, cov_pars = np.array([1,1]), kappa = kappa)
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec)) 
    return(Spatial_cov)
    

def ns_cov_interp(range_vec, sigsq_vec, coords, tck):
    # Using the grid of values to interpolate because sc.special.kv is computationally expensive
    # tck is the output function of sc.interpolate.pchip (Contains information about roughness kappa)
    # ** Has to be Matern model **
    if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
        range_vec = np.array(range_vec)
        sigsq_vec = np.array(sigsq_vec)
    
    N = range_vec.shape[0] # Number of spatial locations
    if coords.shape[0]!=N:
        sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
    # Scale matrix
    arg11 = range_vec
    arg22 = range_vec
    arg12 = np.repeat(0,N)
    ones = np.repeat(1,N)
    det1  = arg11*arg22 - arg12**2
  
    ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) ---
    mat11_1 = np.reshape(arg11, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
    mat11_2 = np.reshape(ones, (N, 1)) * arg11
    ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
    mat22_1 = np.reshape(arg22, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
    mat22_2 = np.reshape(ones, (N, 1)) * arg22
    ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
    mat12_1 = np.reshape(arg12, (N, 1)) * ones
    ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
    mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
    mat11 = 0.5*(mat11_1 + mat11_2)
    mat22 = 0.5*(mat22_1 + mat22_2)
    mat12 = 0.5*(mat12_1 + mat12_2)
  
    det12 = mat11*mat22 - mat12**2
  
    Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
    # Distance matrix
    inv11 = mat22/det12
    inv22 = mat11/det12
    inv12 = -mat12/det12
  
    dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
    dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
    temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
    temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
    temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
    temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
    sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
    sgn_mat1[~sgn_mat1] = -1
    sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
    sgn_mat2[~sgn_mat2] = -1
  
    dists1_sq = dists1**2
    dists2_sq = dists2**2
    dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
    Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
    Dist_mat = np.zeros(Dist_mat_sqd.shape)
    Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
    # Combine
    Unscl_corr = np.ones(Dist_mat_sqd.shape)
    Unscl_corr[Dist_mat_sqd>0] = tck(Dist_mat[Dist_mat_sqd>0])
    NS_corr = Scale_mat*Unscl_corr
  
    Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec))
    return(Spatial_cov)
## -------------------------------------------------------------------------- ##

#########################################################################################
# Write my own covariance function ######################################################
#########################################################################################
# note: gives same result as Likun's
#       paremeterization is same, up to a constant in specifying the range

# def matern_correlation(d, range, nu):
#     # using wikipedia definition
#     part1 = 2**(1-nu)/scipy.special.gamma(nu)
#     part2 = (np.sqrt(2*nu) * d / range)**nu
#     part3 = scipy.special.kv(nu, np.sqrt(2*nu) * d / range)
#     return(part1*part2*part3)
# matern_correlation_vec = np.vectorize(matern_correlation, otypes=[float])

# # pairwise_distance = scipy.spatial.distance.pdist(sites_xy)
# # matern_correlation_vec(pairwise_distance, 1, nu) # gives same result as skMatern(sites_xy)
# # tri = np.zeros((4,4))
# # tri[np.triu_indices(4,1)] = matern_correlation_vec(pairwise_distance, 1, 1)
# # tri + tri.T + np.identity(4)

# # Note:
# #       K[i,j] (row i, col j) means the correlation between site_i and site_j
# #       np.mean(np.round(K,3) == np.round(K_current, 3)) # 1.0, meaning they are the same. 
# K = np.full(shape = (Ns, Ns), fill_value = 0.0)
# for i in range(Ns):
#     for j in range(i+1, Ns):
#         site_i = sites_xy[i,]
#         site_j = sites_xy[j,]
#         d = scipy.spatial.distance.pdist([site_i, site_j])
#         rho_i = range_vec[i]
#         rho_j = range_vec[j]
#         sigma_i = sigsq_vec[i]
#         sigma_j = sigsq_vec[j]
#         M = matern_correlation(d/np.sqrt((rho_i + rho_j)/2), 1, 0.5)
#         C = sigma_i * sigma_j * (np.sqrt(rho_i*rho_j)) * (1/((rho_i + rho_j)/2)) * M
#         K[i,j] = C[0]
# K = K + K.T + sigsq * np.identity(Ns)



# %%
# utility functions ---------------------------------------------------------------------------------------------------

# Gaussian Smoothing Kernel
def weights_fun(d,radius,h=1, cutoff=True):
    # When d > fit radius, the weight will be zero
    # h is the bandwidth parameter
    if(isinstance(d, (int, np.int64, float))): 
        d=np.array([d])
    tmp = np.exp(-d**2/(2*h))
    if cutoff: 
        tmp[d>radius] = 0
    return tmp/np.sum(tmp)

# Wendland compactly-supported basis
def wendland_weights_fun(d, theta, k=0, dimension=2, derivative=0):
    # fields_Wendland(d, theta = 1, dimension, k, derivative=0, phi=NA)
    # theta: the range where the basis value is non-zero, i.e. [0, theta]
    # dimension: dimension of locations 
    # k: smoothness of the function at zero.
    if(isinstance(d, (int, np.int64, float))): 
        d=np.array([d])      
    d = d/theta
    l = np.floor(dimension/2) + k + 1
    if (k==0): 
        res = np.where(d < 1, (1-d)**l, 0)
    if (k==1):
        res = np.where(d < 1, (1-d)**(l+k) * ((l+1)*d + 1), 0)
    if (k==2):
        res = np.where(d < 1, (1-d)**(l+k) * ((l**2+4*l+3)*d**2 + (3*l+6) * d + 3), 0)
    if (k==3):
        res = np.where(d < 1, (1-d)**(l+k) * ((l**3+9*l**2+23*l+15)*d**3 + 
                                            (6*l**2+36*l+45) * d**2 + (15*l+45) * d + 15), 0)
    if (k>3):
        sys.exit("k must be less than 4")
    return res/np.sum(res)

# generate levy random samples
def rlevy(n, m = 0, s = 1):
    if np.any(s < 0):
        sys.exit("s must be positive")
    return s/scipy.stats.norm.ppf(1-scipy.stats.uniform.rvs(0,1,n)/2)**2 + m

def qlevy(p, m = 0, s = 1):
    return m + s/(2*(scipy.special.erfcinv(p))**2)

# Generalized Extreme Value distribution
# note negative shape parametrization in scipy.genextreme
def dgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return scipy.stats.genextreme.logpdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return scipy.stats.genextreme.pdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
def pgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return scipy.stats.genextreme.logcdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return scipy.stats.genextreme.cdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
def qgev(p, Loc, Scale, Shape):
    if type(p).__module__!='numpy':
        p = np.array(p)  
    return scipy.stats.genextreme.ppf(p, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

# Generalized Pareto (GP) distribution
def dGP(y, loc, scale, shape):
    return scipy.stats.genpareto.pdf(y, c=shape, loc=loc, scale=scale)
def pGP(y, loc, scale, shape):
    return scipy.stats.genpareto.cdf(y, c=shape, loc=loc, scale=scale)
def qGP(p, loc, scale, shape):
    return scipy.stats.genpareto.ppf(p, c=shape, loc=loc, scale=scale)

# Censored Generalized Pareto (CGP) distribution, censored at probability p
# the loc should be the threshold u: p = P(Y <= u)
# pX is the CDF of X: pX = P(X <= x)
# p_thres is the censoring threshold probability 
def pCGP(y, p, loc, scale, shape):
    return np.where(y <= loc, p,
                              p + (1-p) * pGP(y, loc, scale, shape))
    # return p + (1-p) * pGP(y, loc, scale, shape)
def dCGP(y, p, loc, scale, shape):
    return np.where(y <= loc, 0.0,
                              (1-p) * dGP(y, loc, scale, shape))
    # return (1-p) * dGP(y, loc, scale, shape)
def qCGP(pX, p_thres, loc, scale, shape):
    return np.where(pX <= p_thres, loc,
                                   qGP((pX - p_thres)/(1-p_thres), loc, scale, shape))
    # return qGP((pX - p_thres)/(1-p_thres), loc, scale, shape)


# Half-t distribution with nu degrees of freedom
def dhalft(y, nu, mu=0, sigma=1):
    if y >= mu:
        return 2*scipy.stats.t.pdf(y, nu, mu, sigma)
    else: # y < mu
        return 0
def phalft(y, nu, mu=0, sigma=1):
    if y >= mu:
        return 2*scipy.stats.t.cdf(y, nu, mu, sigma) - 1
    else: # y < mu
        return 0
def rhalft(nu, mu=0, sigma=1):
    return mu + np.abs(scipy.stats.t.rvs(nu, 0, sigma))

# transformation to standard Pareto
def norm_to_stdPareto(Z):
    pNorm = scipy.stats.norm.cdf(x = Z)
    return(scipy.stats.pareto.ppf(q = pNorm, b = 1))
norm_to_stdPareto_vec = np.vectorize(norm_to_stdPareto)
def stdPareto_to_Norm(W):
    pPareto = scipy.stats.pareto.cdf(x = W, b = 1)
    return(scipy.stats.norm.ppf(q = pPareto))
stdPareto_to_Norm_vec = np.vectorize(stdPareto_to_Norm)

# transformation to shifted Pareto
def norm_to_Pareto1(z):
    if(isinstance(z, (int, np.int64, float))): 
        z=np.array([z])
    tmp = scipy.stats.norm.cdf(z)
    if np.any(tmp==1): 
        tmp[tmp==1]=1-1e-9
    return 1/(1-tmp)-1
def pareto1_to_Norm(W):
    if(isinstance(W, (int, np.int64, float))): 
        W=np.array([W])
    tmp = 1-1/(W+1)
    return scipy.stats.norm.ppf(tmp)

# %%
# specify g(Z) and RW distribution functions --------------------------------------------------------------------------

if norm_pareto == 'standard':

    g    = norm_to_stdPareto_vec
    ginv = stdPareto_to_Norm_vec

    dRW = RW_inte.dRW_standard_Pareto_nugget_vec
    pRW = RW_inte.pRW_standard_Pareto_nugget_vec
    qRW = RW_inte.qRW_standard_Pareto_nugget_vec

if norm_pareto == 'shifted':
    g    = norm_to_Pareto1
    ginv = pareto1_to_Norm

    dRW = print('2D Integral No Implementation!' )
    pRW = print('2D Integral No Implementation!' )
    qRW = print('2D Integral No Implementation!' )

# neural network emulators for distribution functions -----------------------------------------------------------------

with open('qRW_NN_weights_and_biases.pkl', 'rb') as f:
    weights_and_biases = pickle.load(f)
w0, b0, w1, b1, w2, b2, w3, b3 = weights_and_biases

qRW_NN_X_min = np.load('qRW_NN_X_min.npy', allow_pickle=True)
qRW_NN_X_max = np.load('qRW_NN_X_max.npy', allow_pickle=True)

def NN_forward_pass(X_scaled):
        """
        Applies a 4-layer MLP with tanh on hidden layers and linear on output:
        1) Dense(512, tanh)
        2) Dense(512, tanh)
        3) Dense(512, tanh)
        4) Dense(1, linear)

        Parameters
        ----------
        X_scaled : (N, d) array, already min-max scaled
        w0, b0   : weights, bias for layer 1
        w1, b1   : for layer 2
        w2, b2   : for layer 3
        w3, b3   : for layer 4

        Returns
        -------
        (N, ) array of predictions
        """
        # Layer 1
        Z = X_scaled @ w0 + b0
        Z = np.tanh(Z)
        # Layer 2
        Z = Z @ w1 + b1
        Z = np.tanh(Z)
        # Layer 3
        Z = Z @ w2 + b2
        Z = np.tanh(Z)
        # Layer 4
        Z = Z @ w3 + b3
        return Z.ravel()

def qRW_NN(p_vec, phi_vec, gamma_vec, tau_vec):
    inputs = np.column_stack((p_vec, phi_vec, gamma_vec, tau_vec))
    inputs = (inputs - qRW_NN_X_min) / (qRW_NN_X_max - qRW_NN_X_min)
    return np.exp(NN_forward_pass(inputs))

def qRW_NN_2p(p_vec, phi_vec, gamma_vec, tau_vec):
    p_vec     = np.atleast_1d(p_vec)
    phi_vec   = np.atleast_1d(phi_vec)
    gamma_vec = np.atleast_1d(gamma_vec)
    tau_vec   = np.atleast_1d(tau_vec)

    if not (len(p_vec) == len(phi_vec) == len(gamma_vec) == len(tau_vec)):
        max_length = np.max([len(p_vec), len(phi_vec), len(gamma_vec), len(tau_vec)])
        
        if len(p_vec)     == 1: p_vec     = np.full(max_length, p_vec[0])
        if len(phi_vec)   == 1: phi_vec   = np.full(max_length, phi_vec[0])
        if len(gamma_vec) == 1: gamma_vec = np.full(max_length, gamma_vec[0])
        if len(tau_vec)   == 1: tau_vec   = np.full(max_length, tau_vec[0])
    
    if not (len(p_vec) == len(phi_vec) == len(gamma_vec) == len(tau_vec)):
        raise ValueError("Cannot broadcast with different lengths.")

    outputs             = np.full((len(p_vec),), fill_value=np.nan)
    condition_p         = (0.95  <= p_vec)     & (p_vec <= 0.9995)
    condition_phi       = (0.05 <= phi_vec)    & (phi_vec <= 0.95)
    condition_gamma     = (0.5  <= gamma_vec)  & (gamma_vec <= 5)    
    condition_tau       = (1 <= tau_vec)       & (tau_vec <= 100)
    condition           = condition_p & condition_phi & condition_gamma & condition_tau

    if np.mean(condition) < 0.99:
        print('Proportion interpolated:',       np.mean(condition))
        print('Proportion p interpolated:',     np.mean(condition_p))
        print('Proportion phi interpolated:',   np.mean(condition_phi))
        print('Proportion gamma interpolated:', np.mean(condition_gamma))
        print('Proportion tau interpolated:',   np.mean(condition_tau))

    outputs[condition]  = qRW_NN(p_vec[condition], phi_vec[condition], gamma_vec[condition], tau_vec[condition])
    outputs[~condition] = qRW(p_vec[~condition], phi_vec[~condition], gamma_vec[~condition], tau_vec[~condition])
    return outputs.ravel()


# %% Likelihood Not Simplified

def ll_1t(Y, p, u_vec, scale_vec, shape_vec,        # marginal model parameters
          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau, # dependence model parameters
          logS_vec, gamma_at_knots, censored_idx, exceed_idx):         # auxilury information
    
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

def ll_1t_detail(Y, p, u_vec, scale_vec, shape_vec,
          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau,
          logS_vec, gamma_at_knots, censored_idx, exceed_idx):
    
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
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec

    # log conditional likelihood of Z
    Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)

    return (np.sum(censored_ll),np.sum(exceed_ll), np.sum(S_ll), np.sum(Z_ll))

def ll_1t_qRWdRWout(Y, p, u_vec, scale_vec, shape_vec,
          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau,
          logS_vec, gamma_at_knots, censored_idx, exceed_idx,
          qRW_vec, dRW_vec, MVN_frozen):
    
    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW_vec
    dX     = dRW_vec

    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])

    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec # 0.5 here is the gamma_k, not \bar{\gamma}

    # log likelihood of Z
    # Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)
    Z_ll = MVN_frozen.logpdf(Z_vec)

    return np.sum(censored_ll) + np.sum(exceed_ll) + np.sum(S_ll) + np.sum(Z_ll)

def ll_1t_qRWdRWout_detail(Y, p, u_vec, scale_vec, shape_vec,
          R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau,
          logS_vec, gamma_at_knots, censored_idx, exceed_idx,
          qRW_vec, dRW_vec, MVN_frozen):

    X_star = (R_vec ** phi_vec) * g(Z_vec)
    X      = qRW_vec
    dX     = dRW_vec

    # log censored likelihood of y on censored sites
    censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)
    # log censored likelihood of y on exceedance sites
    exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
                    + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
                    - np.log(dX[exceed_idx])
    
    # log likelihood of S
    S_ll = scipy.stats.levy.logpdf(np.exp(logS_vec),  scale = gamma_at_knots) + logS_vec

    # log conditional likelihood of Z
    # Z_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = None, cov = K)
    Z_ll = MVN_frozen.logpdf(Z_vec)

    return (np.sum(censored_ll),np.sum(exceed_ll), np.sum(S_ll), np.sum(Z_ll))
# %%
# imputation of missing values
#   returns (Z_miss, Y_miss)
#   needs to modify the censored and exceedance sites after imputation in the sampler
def impute_ZY_1t(p, u_vec, scale_vec, shape_vec,
              R_vec, Z_vec, K, phi_vec, gamma_bar_vec, tau,
              obs_idx, miss_idx):
    
    # conditional gaussian draw
    K11       = K[miss_idx,:][:,miss_idx]
    K12       = K[miss_idx,:][:,obs_idx]
    K21       = K[obs_idx,:][:,miss_idx]
    K22       = K[obs_idx,:][:,obs_idx]
    K22_inv   = np.linalg.inv(K22)
    cond_mean = K12 @ K22_inv @ Z_vec[obs_idx]
    cond_cov  = K11 - K12 @ K22_inv @ K21
    Z_miss    = scipy.stats.multivariate_normal.rvs(mean = cond_mean, cov = cond_cov)

    # the smooth process X_star
    X_star_miss = (R_vec[miss_idx] ** phi_vec[miss_idx]) * g(Z_miss)
    
    # random draw nugget for the nuggeted process X
    X_miss      = X_star_miss + scipy.stats.norm.rvs(loc = 0, scale = tau, size = len(miss_idx))

    # marginal transform to Y
    Y_miss = qCGP(pRW(X_miss, phi_vec[miss_idx], gamma_bar_vec[miss_idx], tau), 
                  p,
                  u_vec[miss_idx], scale_vec[miss_idx], shape_vec[miss_idx])
    
    return(Z_miss, X_miss, Y_miss)

# %% Likelihood
# Likelihood

# marginal censored (log) likelihood of Y at 1 time
# def Y_censored_ll_1t(Y, p, u_vec, scale_vec, shape_vec,         # marginal observation and parameter
#                      R_vec, Z_vec, phi_vec, gamma_vec, tau,     # coupla model parameter
#                      X, X_star, dX, censored_idx, exceed_idx):  # things to facilitate computation
#     # Note: 
#     #   X_star = (R_vec ** phi_vec) * g(Z_vec)
#     #   X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_vec, tau)
#     #   censored_idx = np.where(Y <= u_vec)[0]
#     #   exceed_idx   = np.where(Y > u_vec)[0]
#     #   If necessary, 
#     #       dRW can be optimized too (by passing a dedicate argument for it)
#     if(isinstance(Y, (int, np.int64, float))): 
#         Y = np.array([Y], dtype='float64')
    
#     # log likelihood of the censored sites
#     censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)

#     # log likelihood of the exceedance sites
#     exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
#                     + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
#                     - np.log(dX[exceed_idx])

#     return np.sum(censored_ll) + np.sum(exceed_ll)

# full conditional likelihood of smooth process X_star
# def X_star_conditional_ll_1t(X_star, R_vec, phi_vec, K, # original Pr(X_star | R_vec, phi_vec, K)
#                              Z_vec):                    # things to facilitate computation
#     # Note:
#     #   Z_vec = ginv(X_star/R_vec**phi_vec)

#     D = len(Z_vec)

#     # log of the D-dimensional joint gaussian density
#     D_gauss_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = np.zeros(D), cov=K) # log D-dimensional joint gaussian density

#     # log of the (determinant of) Jacobian
#     # log_J      = (D/2)*np.log(2*np.pi) + 0.5*np.sum(Z_vec**2) + np.sum(-phi_vec*np.log(R_vec) - 2*np.log(g(Z_vec)))
#     log_J      = (D/2)*np.log(2*np.pi) + 0.5*np.sum(Z_vec**2) + np.sum(phi_vec * np.log(R_vec)) - 2*np.sum(np.log(X_star))

#     return D_gauss_ll + log_J

# marginal censored (log) likelihood of Y at 1 time
# def Y_censored_ll_1t_detail(Y, p, u_vec, scale_vec, shape_vec,         # marginal observation and parameter
#                      R_vec, Z_vec, phi_vec, gamma_vec, tau,     # coupla model parameter
#                      X, X_star, dX, censored_idx, exceed_idx):  # things to facilitate computation
#     # Note: 
#     #   X_star = (R_vec ** phi_vec) * g(Z_vec)
#     #   X      = qRW(pCGP(Y, p, u_vec, scale_vec, shape_vec), phi_vec, gamma_vec, tau)
#     #   censored_idx = np.where(Y <= u_vec)[0]
#     #   exceed_idx   = np.where(Y > u_vec)[0]
#     #   If necessary, 
#     #       dRW can be optimized too (by passing a dedicate argument for it)
#     if(isinstance(Y, (int, np.int64, float))): 
#         Y = np.array([Y], dtype='float64')
    
#     # log likelihood of the censored sites
#     censored_ll = scipy.stats.norm.logcdf((X[censored_idx] - X_star[censored_idx])/tau)

#     # log likelihood of the exceedance sites
#     exceed_ll   = scipy.stats.norm.logpdf(X[exceed_idx], loc = X_star[exceed_idx], scale = tau) \
#                     + np.log(dCGP(Y[exceed_idx], p, u_vec[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) \
#                     - np.log(dX[exceed_idx])

#     return (np.sum(censored_ll), np.sum(exceed_ll))

# full conditional likelihood of smooth process X_star
# def X_star_conditional_ll_1t_detail(X_star, R_vec, phi_vec, K, # original Pr(X_star | R_vec, phi_vec, K)
#                                     Z_vec):                    # things to facilitate computation
#     # Note:
#     #   Z_vec = ginv(X_star/R_vec**phi_vec)

#     D = len(Z_vec)

#     # log of the D-dimensional joint gaussian density
#     D_gauss_ll = scipy.stats.multivariate_normal.logpdf(Z_vec, mean = np.zeros(D), cov=K) # log D-dimensional joint gaussian density

#     # log of the (determinant of) Jacobian
#     log_J_1 = (D/2)*np.log(2*np.pi) + 0.5*np.sum(np.square(Z_vec)) 
#     # log_J_2      = np.sum(-phi_vec*np.log(R_vec) - 2*np.log(g(Z_vec)))
#     log_J_2 = np.sum(phi_vec * np.log(R_vec)) - 2*np.sum(np.log(X_star))

#     return (D_gauss_ll, log_J_1, log_J_2)

# %% Imputation for missing data
# imputaiton for missing data -----------------------------------------------------------------------------------------

# def impute_1t_(miss_index, obs_index, 
#               R_vec, Z_vec, phi_vec, gamma_vec, tau, K, # Ingredients
#               p, u_vec, sigma_vec, ksi_vec,             # Marginal data parameters
#               random_generator):                        # for generating epsilons

#     if len(miss_index) == 0:
#         return (None, None)

#     # Calculate conditional mean and covariance
#     Z_obs     = Z_vec[obs_index]
#     K11       = K[miss_index,:][:,miss_index] # shape(miss, miss)
#     K12       = K[miss_index,:][:,obs_index]  # shape(miss, obs)
#     K21       = K[obs_index,:][:,miss_index]  # shape(obs, miss)
#     K22       = K[obs_index,:][:,obs_index]   # shape(obs, obs)
#     K22_inv   = np.linalg.inv(K22)
#     cond_mean = K12 @ K22_inv @ Z_obs
#     cond_cov  = K11 - K12 @ K22_inv @ K21

#     # Make Kriging Prediciton
#     phi_vec_miss   = phi_vec[miss_index]
#     gamma_vec_miss = gamma_vec[miss_index]
#     R_vec_miss     = R_vec[miss_index]
#     u_vec_miss     = u_vec[miss_index]
#     sigma_vec_miss = sigma_vec[miss_index]
#     ksi_vec_miss   = ksi_vec[miss_index]
#     Z_miss = scipy.stats.multivariate_normal.rvs(mean = cond_mean, cov = cond_cov)

#     # Generate X and Y
#     X_star_miss = R_vec_miss**phi_vec_miss * g(Z_miss)
#     X_miss      = X_star_miss + scipy.stats.norm.rvs(loc = 0, scale = tau, size = len(miss_index), random_state = random_generator)
#     Y_miss = qCGP(pRW(X_miss, phi_vec_miss, gamma_vec_miss, tau), 
#                   p,
#                   u_vec_miss, sigma_vec_miss, ksi_vec_miss)

#     return (Z_miss, X_star_miss, X_miss, Y_miss)

# def impute_Y_1t(miss_idx, obs_idx,
#                 R_vec, Z_vec, phi_vec, gamma_vec, tau, K,
#                 p, u_vec, sigma_vec, ksi_vec,
#                 random_generator):
#     # initial imputation is coded inside the sampler
#     # this is for during the MCMC updates (i.e. when K changes)
    
#     pass