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

