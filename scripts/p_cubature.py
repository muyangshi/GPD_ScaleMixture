# %%
# imports
import numpy as np
from cubature import cubature
from scipy import stats
from scipy.optimize import root_scalar
import scipy
import matplotlib.pyplot as plt
import numpy as np
from mpmath import gammainc

# %%
def part_1_integrand(x_array, *args, **kwargs):
    epsilon = x_array
    tau = args
    # print('epsilon:', epsilon)
    # print('tau:', tau)
    return stats.norm.pdf(epsilon, scale = tau)

def F_X_part_1(x, tau):
    LB = [x]
    UB = [39*tau]
    val, err = cubature(part_1_integrand, ndim = 1, fdim = 1, xmin = LB, xmax = UB, args = (tau,),
                        adaptive='p')
    # print('part 1 err: ', err)
    return val[0]

def part_2_integrand(x_array, *args, **kwargs):
    t, epsilon = x_array
    x, phi, gamma, tau = args

    cst = np.sqrt(gamma/(2*np.pi))
    gaussian_density = stats.norm.pdf(epsilon, scale = tau)
    ratio_numerator = np.power((1-t)/t, phi-1.5)
    ratio_denominator = x - epsilon + np.power((1-t)/t, phi)
    exponential_term = np.exp(-gamma/(2*((1-t)/t)))
    jacobian = 1/(t**2)
    return cst * gaussian_density * (ratio_numerator/ratio_denominator) * exponential_term * jacobian

def F_X_part_2(x, phi, gamma, tau):
    LB_outer = -39*tau
    LB_inner = 0
    UB_outer = min(x, 39*tau)
    UB_inner = 1
    xmin = [LB_inner, LB_outer]
    xmax = [UB_inner, UB_outer]
    val, err = cubature(part_2_integrand, ndim = 2, fdim = 1, xmin = xmin, xmax = xmax, args = (x, phi, gamma, tau),
                        adaptive='h', abserr=1e-12, relerr=1e-12)
    # print('part 2 err: ', err)
    return val[0]

def F_X(x, phi, gamma, tau):
    return 1 - F_X_part_1(x, tau) - F_X_part_2(x, phi, gamma, tau)


# Asymptotic Approximation version
# Error near phi = 0.5
C_alpha = scipy.special.gamma(0.5) * np.sin(0.5*np.pi/2) / np.pi
def F_X_approx(x, phi, gamma):
    if phi > 0.5:
        # print('phi > 0.5')
        survival = (2 * C_alpha * np.power(gamma, 0.5) / (1 - 0.5/phi)) * np.power(x, -0.5/phi)
        return 1 - survival
    elif phi == 0.5:
        # print('phi = 0.5')
        survival = 2 * C_alpha * np.power(gamma, 0.5) * (1/x) * np.log(x)
        return 1 - survival
    else:
        # print('0 <= phi < 0.5')
        # survival = np.power(gamma, phi) * np.power(np.cos(np.pi*0.5/2), -phi/0.5) * \
        #             (scipy.special.gamma(1 - phi/0.5) / scipy.special.gamma(1-phi)) * (1/x)
        # return 1 - survival
        survival = np.power(gamma, phi) * np.power(np.cos(np.pi*0.5/2), -phi/0.5) * \
                    np.exp(scipy.special.loggamma(1 - phi/0.5) - scipy.special.loggamma(1-phi)) * (1/x)
        return 1 - survival


# Smooth Approximation
# CDF of the smooth process
# def F_X_smooth(x, phi, gamma):
#     lower_gamma = scipy.special.gamma(0.5) * scipy.special.gammainc(0.5, np.power(gamma, 0.5)/(2*np.power(x, 1/phi)))

#     if phi >= 0.5:
#         survival = np.sqrt(1/np.pi) * lower_gamma
#         return 1 - survival
#     else: # phi < 0.5
#         upper_gamma = scipy.special.gamma(0.5 - phi) * scipy.special.gammaincc(0.5 - phi, np.power(gamma, 0.5)/(2*np.power(x, 1/phi)))
#         survival = np.sqrt(1/np.pi) * lower_gamma + \
#             (1/x) * np.sqrt(1/np.pi) * np.power(np.power(gamma, 0.5)/2, phi) * upper_gamma
        
#     return 1 - survival

# Smooth Approximation
# CDF of the smooth process
def F_X_smooth_mpmath(x, phi, gamma):
    lower_gamma = gammainc(0.5, b = np.power((gamma/(2*x)), (1/phi)))
    upper_gamma = gammainc(0.5 - phi, a = np.power((gamma/(2*x)), (1/phi)))
    # lower_gamma = gammainc(0.5, b=np.power(gamma, 1)/(2*np.power(x, 1/phi)))
    # upper_gamma = gammainc(0.5-phi, a = np.power(gamma, 1)/(2*np.power(x, 1/phi)))
    survival = np.sqrt(1/np.pi) * lower_gamma + \
        (1/x) * np.sqrt(1/np.pi) * np.power(np.power(gamma, 1)/2, phi) * upper_gamma
    return 1 - survival

# Numerical Integration of the Nuggeted Process
# Convolve the "closed form smooth process" with gaussian

import scipy.integrate as integrate
import scipy.special as special

def survival_smooth(x, phi, gamma):
    return 1 - F_X_smooth_mpmath(x, phi, gamma)

def F_X_1D_integrand(epsilon, x, phi, gamma, tau):
    return survival_smooth(x - epsilon, phi, gamma) * stats.norm.pdf(epsilon, scale = tau)

def F_X_1D(x, phi, gamma, tau):
    part_1 = stats.norm.sf(x, scale = tau)
    part_2 = integrate.quad(F_X_1D_integrand, -39*tau, x, args=(x, phi, gamma, tau), epsabs = 1e-10, epsrel = 1e-10, limit = 100)
    return 1 - part_1 - part_2[0]


# %%
# Density f_X using cubature numerical integration
def f_X_integrand(x_array, *args, **kwargs):
    t, epsilon = x_array
    x, phi, gamma, tau = args
    cst = np.sqrt(gamma/(2*np.pi))
    gaussian_density = stats.norm.pdf(epsilon, scale = tau)
    ratio_numerator = np.power((1-t)/t, phi-1.5)
    ratio_denominator = (x - epsilon + np.power((1-t)/t, phi))**2
    exponential_term = np.exp(-gamma/(2*((1-t)/t)))
    jacobian = 1/(t**2)
    return cst * gaussian_density * (ratio_numerator/ratio_denominator) * exponential_term * jacobian

def f_X(x, phi, gamma, tau):
    LB_outer = -39*tau
    LB_inner = 0
    UB_outer = min(x, 39*tau)
    UB_inner = 1
    xmin = [LB_inner, LB_outer]
    xmax = [UB_inner, UB_outer]
    val, err = cubature(f_X_integrand, ndim = 2, fdim = 1, xmin = xmin, xmax = xmax, args = (x, phi, gamma, tau))
    return val[0]

# %%
# Root Finding for Quantile Function

# %%
# def f(x, coef):
#     return (coef*x**3 - 1)

# def fprim(x, coef):
#     return coef*3*x**2

# sol = root_scalar(f, args = (1,), bracket = [2,3], method='brentq')

# Actual Quantile Function from Numerical Integrated CDF
def function_to_solve(x, p, phi, gamma, tau):
    return F_X(x, phi, gamma, tau) - p
def quantile_F_X(p, phi, gamma, tau):
    sol = root_scalar(function_to_solve, args = (p, phi, gamma, tau), 
                        bracket = [0, 1e9], method = 'bisect') # ridder will fail
    return sol.root

def log_quantile_F_X(p, phi, gamma, tau):
    return np.log(quantile_F_X(p, phi, gamma, tau))

# Quantile Function from Asymptotic Approximated CDF
def function_to_solve_approx(x, p, phi, gamma):
    return F_X_approx(x, phi, gamma) - p
def quantile_F_X_approx(p, phi, gamma):
    sol = root_scalar(function_to_solve_approx, args = (p, phi, gamma), 
                        bracket = [10, 1e9], method = 'bisect') # ridder will fail
    return sol.root

# def function_to_solve_smooth(x, p, phi, gamma):
#     return F_X_smooth(x, phi, gamma) - p
# def quantile_F_X_smooth(p, phi, gamma):
#     sol = root_scalar(function_to_solve_smooth, args = (p, phi, gamma), 
#                         bracket = [10, 1e9], method = 'bisect') # ridder will fail
#     return sol.root

# Quantile Function of the Smooth Process
def function_to_solve_smooth_mpmath(x, p, phi, gamma):
    return F_X_smooth_mpmath(x, phi, gamma) - p
def quantile_F_X_smooth_mpmath(p, phi, gamma):
    sol = root_scalar(function_to_solve_smooth_mpmath, args = (p, phi, gamma), 
                        bracket = [10, 1e11], method = 'bisect') # ridder will fail
    return sol.root

# Quantile function from the 1D numerical integration
# Convolve the "closed form smooth process" with gaussian
def function_to_solve_1D(x, p, phi, gamma, tau):
    return F_X_1D(x, phi, gamma, tau) - p
def quantile_F_X_1D(p, phi, gamma, tau):
    sol = root_scalar(function_to_solve_1D, args = (p, phi, gamma, tau), 
                        bracket = [1, 1e11], method = 'bisect')
    return sol.root

# root_scalar(function_to_solve, args = (0.9,0.475,1,1), bracket = [0,1e8], method = 'ridder')
#
# %%
# p_train = 0.9592486902182072
# phi_train = 0.7777777777777777
# gamma_train = 1
# tau_train = np.geomspace(0.1,100, 20)
# p_train_g, phi_train_g, gamma_train_g, tau_train_g = np.meshgrid(p_train, phi_train, gamma_train, tau_train, indexing='ij')
# X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g, tau_train_g]).reshape(4,-1).T
# results = pool.starmap(log_quantile_F_X, X_train)

# phi = 0.5
# gamma = 1
# tau = 100
# # plt_xvals = np.linspace(2000, 2e5, num = 50)
# coefs = np.linspace(10, 200, num = 20)
# coefs = np.linspace(200, 1000, num = 9)
# plt_xvals = coefs * tau
# plt_FX = [F_X(x, phi, gamma, tau) for x in plt_xvals] # 1min30sec 20 points
# plt_FX_approx = [F_X_approx(x, phi, gamma) for x in plt_xvals]
# plt.plot(plt_xvals, np.array(plt_FX) - np.array(plt_FX_approx), 'g.-')
# plt.xlabel('xvals')
# plt.ylabel('difference in CDF')
# np.array(plt_FX) - np.array(plt_FX_approx)

# %%

# phi = 0.8
# gamma = 1
# tau = 8
# plt_xvals = np.linspace(1e4, 4e5, num = 20)
# # coefs = np.linspace(10, 200, num = 20)
# # coefs = np.linspace(200, 1000, num = 9)
# # plt_xvals = coefs * tau
# # plt_FX = [F_X(x, phi, gamma, tau) for x in plt_xvals] # 1min30sec 20 points
# plt_FX_smooth = [F_X_smooth(x, phi, gamma) for x in plt_xvals]

# plt_FX_approx = [F_X_approx(x, phi, gamma) for x in plt_xvals]
# # plt.plot(plt_xvals, np.array(plt_FX) - np.array(plt_FX_approx), 'g.-')
# # plt.plot(plt_xvals, np.array(plt_FX_approx), 'g.-')
# plt.xlabel('xvals')
# # plt.ylabel('difference in CDF')
# # np.array(plt_FX) - np.array(plt_FX_approx)
# # %%
# from multiprocessing import Pool
# pY = 0.9997
# gamma = 1
# tau = 8
# phis = np.linspace(0.4, 0.6, num = 40)
# pool = Pool(processes=6)
# task_matrix = np.column_stack((np.tile(pY, 40), phis, np.tile(gamma,40), np.tile(tau, 40)))
# result = pool.starmap(quantile_F_X, task_matrix)
# pool.close()
# pool.join()
# plt.plot(phis, result, 'g.-')

# Graphing the quantile functions
# import matplotlib.pyplot as plt
# from multiprocessing import Pool
# pool = Pool(processes=6)
# numpoints = 50
# plt_xvals = np.linspace(0, 2000, num = numpoints)
# # plt_FX = [F_X(xval, 0.5, 1, 10) for xval in plt_xvals]

# phi = 0.5
# gamma = 1
# tau = 10
# task_matrix = np.column_stack((plt_xvals, np.tile(phi, numpoints), np.tile(gamma, numpoints), np.tile(tau, numpoints)))
# plt_FX = pool.starmap(F_X, task_matrix)
# plt_FX_smooth = [F_X_smooth_mpmath(xval, 0.5, 1) for xval in plt_xvals]
# plt.plot(plt_xvals, plt_FX, 'g.-', label='cubature fix')
# plt.plot(plt_xvals, plt_FX_smooth, 'r.-', label='smooth mpmath')
# plt.legend()
# plt.xlabel('x value')
# plt.ylabel('CDF F(x)')
# plt.savefig('CDF_good_bad.svg', dpi = 1200)