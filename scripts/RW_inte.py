# %%
import numpy as np
# import scipy
# from scipy.integrate import quad
# from mpmath import mp
# import model_sim
# from numba import jit
import os, ctypes
RW_lib = ctypes.CDLL(os.path.abspath('./RW_inte_cpp.so'))

# STANDARD PARETO -------------------------------------------------------------

# %% STANDARD PARETO NO NUGGET

"""
- Since no nugget, this should be used with GEV block maxima data
- Small scale ~20 simulation study in STAT 600 shows this not working
    - Suspect the mismatch in the bulk of the data mess with inference
    - "Super fast" since only involves gamma functions, no numerical integral
"""

RW_lib.pRW_standard_Pareto_C.restype  = ctypes.c_double
RW_lib.pRW_standard_Pareto_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
RW_lib.dRW_standard_Pareto_C.restype  = ctypes.c_double
RW_lib.dRW_standard_Pareto_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
RW_lib.qRW_standard_Pareto_C_brent.restype  = ctypes.c_double
RW_lib.qRW_standard_Pareto_C_brent.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

dRW_standard_Pareto_vec = np.vectorize(RW_lib.dRW_standard_Pareto_C, otypes=[float])
pRW_standard_Pareto_vec = np.vectorize(RW_lib.pRW_standard_Pareto_C, otypes=[float])
qRW_standard_Pareto_vec = np.vectorize(RW_lib.qRW_standard_Pareto_C_brent, otypes=[float])


# %% STANDARD PARETO WITH NUGGET

"""
- Convolution with a Gaussian(0, var = tau^2) nugget
- Since nugget, this should be used with GP threshold exceedance data
- One dimensional numerical integration for convolution, rest is gamma function
"""

RW_lib.dRW_standard_Pareto_nugget_C.restype  = ctypes.c_double
RW_lib.dRW_standard_Pareto_nugget_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.pRW_standard_Pareto_nugget_C.restype  = ctypes.c_double
RW_lib.pRW_standard_Pareto_nugget_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)

RW_lib.qRW_standard_Pareto_nugget_C_brent.restype  = ctypes.c_double
RW_lib.qRW_standard_Pareto_nugget_C_brent.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)

dRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.dRW_standard_Pareto_nugget_C, otypes=[float])
pRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.pRW_standard_Pareto_nugget_C, otypes=[float])
qRW_standard_Pareto_nugget_vec = np.vectorize(RW_lib.qRW_standard_Pareto_nugget_C_brent, otypes=[float])

# explicit parameters for plotting
# RW_lib.pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot.restype = ctypes.c_double
# RW_lib.pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
# pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot_vec = np.vectorize(RW_lib.pRW_standard_Pareto_nugget_upper_gamma_integrand_forplot, otypes=[float])

# transformation on s = (1/t)/t to (0,1) bound gives bad result. 
# Don't do it!
# RW_lib.pRW_standard_Pareto_nugget_transform_C.restype  = ctypes.c_double
# RW_lib.pRW_standard_Pareto_nugget_transform_C.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
# pRW_standard_Pareto_nugget_transform_vec               = np.vectorize(RW_lib.pRW_standard_Pareto_nugget_transform_C, otypes=[float])

# %% STANDARD PARETO WITH NUGGET using FFT
"""
- Convolution with a Gaussian(0, var = tau^2) nugget
- To avoid numerical integration, use FFT for the convolution
"""
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from math import sqrt, pi
import matplotlib.pyplot as plt
import scipy

"""
return 
    the grid: x
    the CDF: F(x)
Use those to construct a interpolator for repeated use
"""
def pRW_standard_Pareto_nugget_FFT(x_val, phi, gamma, tau, n = 2**10):

    # 1. Determine the domain via quantile function:
    x_min = qRW_standard_Pareto_vec(0.1, phi, gamma)
    x_max = qRW_standard_Pareto_vec(0.99999, phi, gamma)
    x = np.linspace(x_min, x_max, n)
    dx = x[1] - x[0]

    # 2. Use middle padding
    n_pad        = 2 * n
    # start_idx = (n_pad - n)//2
    # end_idx   = start_idx + n
    mid       = n_pad // 2
    start_idx = mid - n//2
    end_idx   = mid + n//2

    # 3. Evaluate F_{X^*}(x)
    F_X_star_padded = np.zeros(n_pad)
    F_X_star        = pRW_standard_Pareto_vec(x, phi, gamma)
    F_X_star_padded[start_idx:end_idx] = F_X_star

    # 4. Fourier transform of f_ε(x) and F_{X^*}(x)
    
    # Analytical Fourier transform of Gaussian f_epsilon
    # f_epsilon(x) = (1/(tau*sqrt(2*pi))) * exp(-x^2/(2*tau^2))
    # FT of f_epsilon(x) w.r.t x:
    # F{f_epsilon}(f) = exp(-2 * (pi^2) * (tau^2) * (f^2))
    freqs         = fftfreq(n_pad, dx)
    f_epsilon_fft = np.exp(-2*(pi**2)*(tau**2)*(freqs**2))

    # Numerical Fourier transform of F_{X^*}(x)
    F_X_star_fft = fft(F_X_star_padded)
    
    # 5. Convolution
    conv_padded = ifft(F_X_star_fft * f_epsilon_fft)*dx
    F_X_padded  = np.real(conv_padded)
    F_X         = F_X_padded[start_idx:end_idx]

    # Ensure monotonicity and bounds [0,1], just in case of minor numerical issues:
    # F_X = np.clip(F_X, 0.0, 1.0)

    return scipy.interpolate.interp1d(x, F_X)
    # return np.interp(x_val, x, F_X)

# def qX(u):
#     return np.interp(u, F_X, x)

x_plot = np.linspace(qRW_standard_Pareto_nugget_vec(0.8, 0.5, 1, 10), qRW_standard_Pareto_nugget_vec(0.999, 0.5, 1, 10), 100)
f_20 = pRW_standard_Pareto_nugget_FFT(0, 0.5, 1, 10, 2**20)
f_21 = pRW_standard_Pareto_nugget_FFT(0, 0.5, 1, 10, 2**21)
f_22 = pRW_standard_Pareto_nugget_FFT(0, 0.5, 1, 10, 2**22)
plt.plot(x_plot, pRW_standard_Pareto_nugget_vec(x_plot, 0.5, 1, 10), label = 'exact')
plt.plot(x_plot, f_20(x_plot), label = 'FFT 2**20')
plt.plot(x_plot, f_21(x_plot), label = 'FFT 2**21')
plt.plot(x_plot, f_22(x_plot), label = 'FFT 2**22')
plt.legend(loc='lower right')



# SHIFTED PARETO --------------------------------------------------------------

# %% SHIFTED PARETO NO NUGGET
"""
- This has been used for GEV block maxima data
- 3 x 50 simulation study showed it works well
"""

RW_lib.pRW_transformed.restype = ctypes.c_double
RW_lib.pRW_transformed.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
RW_lib.dRW_transformed.restype = ctypes.c_double
RW_lib.dRW_transformed.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
RW_lib.qRW_transformed_brent.restype = ctypes.c_double
RW_lib.qRW_transformed_brent.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)

pRW_transformed_cpp = np.vectorize(RW_lib.pRW_transformed, otypes=[float])
dRW_transformed_cpp = np.vectorize(RW_lib.dRW_transformed, otypes=[float])
qRW_transformed_cpp = np.vectorize(RW_lib.qRW_transformed_brent, otypes=[float])

# # no gain in accuracy
# RW_lib.pRW_transformed_2piece.restype = ctypes.c_double
# RW_lib.pRW_transformed_2piece.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
# pRW_transformed_2piece_cpp = np.vectorize(RW_lib.pRW_transformed_2piece, otypes=[float])

# scipy's solver is really sketchy, DON'T USE IT
# it can really give a WRONG ANSWER!!! and it is also SLOW!!!
# hence use C as much as possible
# def qRW_transformed_using_cpp(p, phi, gamma):
#     try:
#         return scipy.optimize.root_scalar(lambda x: pRW_transformed_cpp(x, phi, gamma) - p,
#                                         bracket=[0.1,1e12],
#                                         fprime = lambda x: dRW_transformed_cpp(x, phi, gamma),
#                                         x0 = 10,
#                                         method='ridder').root
#     except Exception as e:
#         print(e)
#         print('p=',p,',','phi=',phi,',','gamma',gamma)
# qRW_transformed_cpp = np.vectorize(qRW_transformed_using_cpp, otypes=[float])


# %% SHIFTED PARETO WITH NUGGET
"""
- Convolution with a Gaussian(0, var = tau^2) nugget
- Since nugget, this should be used with GP threshold exceedance data
- Two dimensional numerical integration for both convolution and original distribution
"""

# see p_cubature.py

