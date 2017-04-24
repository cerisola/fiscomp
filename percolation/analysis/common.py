import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate


Z_normal = { None: 1, '90': 1.644854, '95': 1.959964, '99': 2.575829, '99.9': 3.290527, '99.99': 3.890592 }


# % Generic % #
def mean(v):
    return np.mean(v)


def var(v):
    return np.var(v, ddof=1)


def std(v):
    return np.std(v, ddof=1)


def sem(v, ci=None):
    Z = Z_normal[ci]
    return Z*stats.sem(v)


def cdf_mean(F, x):
    return (1 - integrate.simps(y=F, x=x))


def cdf_var(F, x):
    return (2*integrate.simps(y=x*(1-F), x=x) - cdf_mean(F, x)**2)


def cdf_std(F, x):
    return np.sqrt(cdf_var(F, x))


def cdf_median(Finf, Fsup, x):
    median_inf = x[np.where(Fsup <= 0.5)[0][-1]]
    median_sup = x[np.where(Finf >= 0.5)[0][0]]
    median_med = median_inf + (median_sup - median_inf)/2
    return median_med, median_inf, median_sup


# % Binomial Distribution Aux % #
def binomial_var(p, n):
    return n*p*(1-p)


def binomial_std(p, n):
    return np.sqrt(n*p*(1 - p))


def binomial_sem(p, n, ci=None):
    Z = Z_normal[ci]
    return Z*np.sqrt(p*(1 - p)/n)


def binomial_ci_wald(p, n, ci=None):
    Z = Z_normal[ci]
    normal_stderr = Z*np.sqrt(p*(1 - p)/n)
    p_min = p - normal_stderr
    p_max = p + normal_stderr
    return p_min, p_max


def binomial_ci_wilson(p, n, ci=None):
    Z = Z_normal[ci]
    p_min = (2*n*p + Z**2 - (Z*np.sqrt(Z**2 - 1/n + 4*n*p*(1-p) + (4*p - 2)) + 1))/(2*(n + Z**2))
    p_max = (2*n*p + Z**2 + (Z*np.sqrt(Z**2 - 1/n + 4*n*p*(1-p) - (4*p - 2)) + 1))/(2*(n + Z**2))
    p_min = np.maximum(0, p_min)
    p_max = np.minimum(1, p_max)
    return p_min, p_max


# % Utility function to apply above funtions to lists of different sizes of arrays % #
def listmap(func, v, args=None):
    return np.array([func(v[idx], **args) if args else func(v[idx]) for idx in range(len(v))])


def listop(func, v, w, args=None):
    return np.array([func(v[idx], w[idx], **args) if args else func(v[idx], w[idx]) for idx in range(len(v))])


def varlistop(func, v, args=None):
    ret = []
    for idx in range(len(v[0])):
        vargs = []
        for idx_arg in range(len(v)):
            vargs.append(v[idx_arg][idx])
        if args:
            ret.append(func(*vargs, **args))
        else:
            ret.append(func(*vargs))
    return np.array(ret)

