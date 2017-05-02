import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import statsmodels.api as sm


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


def median(v):
    return np.median(v)


def err_chebyshev(v, ci):
    N = v.size
    P = 1 - float(ci)/100
    sigma2 = var(v)
    epsilon2 = sigma2/(N*P)
    return np.sqrt(epsilon2)


def err_median(v, ci):
    #Z = { '90': 1.22, '95': 1.36, '99': 1.69 }
    N = v.size
    P = 1 - float(ci)/100
    epsilon2 = np.abs(np.log(1/P/2)/(4*N))
    return np.sqrt(epsilon2)


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


# % Utility fit functions % #
def chi2reduced(fit_result, x, y, add_constant=True, log_scale=False):
    dof = len(x) - len(fit_result.params)
    if log_scale:
        X = np.log(x)
        Y = np.log(y)
    else:
        X = x
        Y = y
    if add_constant:
        X = sm.add_constant(X)
    residuals = (Y - fit_result.predict(X))**2
    mse = np.sum(residuals)/dof
    return mse


def fiteval(fit_result, x, add_constant=True, log_scale=False, exp_result=False):
    if log_scale:
        X = np.log(x)
    else:
        X = x
    if add_constant:
        X = sm.add_constant(X)
    y = fit_result.predict(X)
    if exp_result:
        y = np.exp(y)
    return y
