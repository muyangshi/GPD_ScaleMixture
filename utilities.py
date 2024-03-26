# combine utilitlies helpful to MCMC sampler
# grabbed and copied useful functions from Likun's model_sim.py, ns_cov.py
# Require:
#   - RW_inte.py, RW_inte_cpp.cpp & RW_inte.cpp.so
# %%
# general imports and ubiquitous utilities
import sys
import numpy as np
import scipy
import scipy.special as sc
from scipy.spatial import distance
import RW_inte
norm_pareto = 'standard'

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
def pCGP(y, p, loc, scale, shape):
    return p + (1-p) * pGP(y, loc, scale, shape)
def dCGP(y, p, loc, scale, shape):
    return (1-p) * dGP(y, loc, scale, shape)

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
def rhalft(y, nu, mu=0, sigma=1):
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

    norm_to_Pareto = norm_to_stdPareto_vec
    pareto_to_Norm = stdPareto_to_Norm_vec

    dRW = RW_inte.dRW_standard_Pareto_nugget_vec
    pRW = RW_inte.pRW_standard_Pareto_nugget_vec
    qRW = RW_inte.qRW_standard_Pareto_nugget_vec


# %% Likelihood
# Likelihood

# marginal censored (log) likelihood at 1 time
def marg_log_likelihood_1t(Y, X, tau, u, p, scale_vec, shape_vec, phi_vec, gamma_vec):
    # other than tau and p,
    # all parameters are (Ns,) vectors, the sites at time t
    if(isinstance(Y, (int, np.int64, float))): 
        Y = np.array([Y], dtype='float64')

    # censored
    censored_idx = np.where(Y <= u)[0]
    censored_loglik = scipy.stats.norm.logcdf((qRW(p, 
                                                   phi_vec[censored_idx], 
                                                   gamma_vec[censored_idx], 
                                                   tau) - X[censored_idx])/tau)

    # threshold exceeded
    exceed_idx = np.where(Y > u)[0]
    qX_pY = qRW(pCGP(Y[exceed_idx],
                     p,
                     u[exceed_idx],
                     scale_vec[exceed_idx],
                     shape_vec[exceed_idx]), 
                phi_vec[exceed_idx],
                gamma_vec[exceed_idx],
                tau)
    exceed_loglik = scipy.stats.norm.logpdf(qX_pY, loc = X[exceed_idx], scale = tau) + \
                        np.log(dCGP(Y[exceed_idx], p, u[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx])) - \
                        dRW(qX_pY, phi_vec[exceed_idx], gamma_vec[exceed_idx], tau)

    return np.sum(censored_loglik) + np.sum(exceed_loglik)

# marginal censored (log) likelihood by parts, for plot and debug
def marg_log_likelihood_1t(Y, X, tau, u, p, scale_vec, shape_vec, phi_vec, gamma_vec):
    # other than tau and p,
    # all parameters are (Ns,) vectors, the sites at time t
    if(isinstance(Y, (int, np.int64, float))): 
        Y = np.array([Y], dtype='float64')

    # censored
    censored_idx = np.where(Y <= u)[0]
    censored_loglik = scipy.stats.norm.logcdf((qRW(p, 
                                                   phi_vec[censored_idx], 
                                                   gamma_vec[censored_idx], 
                                                   tau) - X[censored_idx])/tau)

    # threshold exceeded
    exceed_idx = np.where(Y > u)[0]
    qX_pY = qRW(pCGP(Y[exceed_idx],
                     p,
                     u[exceed_idx],
                     scale_vec[exceed_idx],
                     shape_vec[exceed_idx]), 
                phi_vec[exceed_idx],
                gamma_vec[exceed_idx],
                tau)
    exceed_loglik_1 = scipy.stats.norm.logpdf(qX_pY, loc = X[exceed_idx], scale = tau)
    exceed_loglik_2 = np.log(dCGP(Y[exceed_idx], p, u[exceed_idx], scale_vec[exceed_idx], shape_vec[exceed_idx]))
    exceed_loglik_3 = - dRW(qX_pY, phi_vec[exceed_idx], gamma_vec[exceed_idx], tau)

    return (np.sum(censored_loglik), np.sum(exceed_loglik_1), np.sum(exceed_loglik_2), np.sum(exceed_loglik_3))


# %% Imputation for missing data
# imputaiton for missing data -----------------------------------------------------------------------------------------

def impute_1t(miss_index, obs_index, Y_vec, X_vec, mu_vec, sigma_vec, ksi_vec, phi_vec, gamma_vec, R_vec, K):

    if len(miss_index) == 0:
        return (None, None)

    phi_vec_obs = phi_vec[obs_index]
    R_vec_obs   = R_vec[obs_index]
    X_obs       = X_vec[obs_index]
    Z_obs       = pareto_to_Norm(X_obs/R_vec_obs**phi_vec_obs)

    K11       = K[miss_index,:][:,miss_index] # shape(miss, miss)
    K12       = K[miss_index,:][:,obs_index]  # shape(miss, obs)
    K21       = K[obs_index,:][:,miss_index]  # shape(obs, miss)
    K22       = K[obs_index,:][:,obs_index]   # shape(obs, obs)
    K22_inv   = np.linalg.inv(K22)
    
    cond_mean = K12 @ K22_inv @ Z_obs
    cond_cov  = K11 - K12 @ K22_inv @ K21

    phi_vec_miss      = phi_vec[miss_index]
    gamma_vec_miss    = gamma_vec[miss_index]
    R_vec_miss        = R_vec[miss_index]
    mu_vec_miss       = mu_vec[miss_index]
    sigma_vec_miss    = sigma_vec[miss_index]
    ksi_vec_miss      = ksi_vec[miss_index]

    Z_miss = scipy.stats.multivariate_normal.rvs(mean = cond_mean, cov = cond_cov)
    X_miss = R_vec_miss**phi_vec_miss * norm_to_Pareto(Z_miss)
    Y_miss = qgev(pRW(X_miss, phi_vec_miss, gamma_vec_miss), 
                    mu_vec_miss, sigma_vec_miss, ksi_vec_miss)

    return (X_miss,Y_miss)