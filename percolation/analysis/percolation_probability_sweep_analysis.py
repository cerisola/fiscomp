import importlib
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import load_data
import common
importlib.reload(load_data)
importlib.reload(common)


def fit_std(mean, std):
    idx_sort = np.argsort(std)
    idx_min = 0
    idx_max = 20
    fit_model = sm.OLS(mean[idx_sort][idx_min:idx_max], sm.add_constant(std[idx_sort][idx_min:idx_max]))
    fit_result = fit_model.fit()
    intercept = fit_result.params[0]
    slope = fit_result.params[1]
    return slope, intercept, fit_result


def plot_percolation_probability(p_occupation, p_percolation, p_percolation_inf, p_percolation_sup, L, ci=''):
    plt.figure()
    plt.title('Percolation probability for 2D square lattice with L = {}'.format(L))
    plt.plot(p_occupation, p_percolation, '-', color='royalblue', linewidth=1.0, label='$p_{\mathrm{percolation}}$')
    plt.fill_between(p_occupation, p_percolation_sup, p_percolation_inf, facecolor='royalblue', alpha=0.5, label='{}% CI'.format(ci))
    plt.grid()
    plt.xlabel('$p_{\mathrm{occupation}}$')
    plt.ylabel('$p_{\mathrm{percolation}}$')
    plt.legend()
    plt.show()


def plot_critical_probability_statistics(mean, std, median, median_inf, median_sup, L, ci_style='none'):
    plt.figure()
    if ci_style == 'errorbar':
        plt.errorbar(L, mean, yerr=std, fmt='-o', label='mean')
        plt.errorbar(L, median, yerr=median_err, fmt='-o', label='median')
    elif ci_style == 'area':
        plt.plot(L, mean, '-o', markersize=4.0, label='mean')
        plt.fill_between(L, mean + std, mean - std, alpha=0.5)
        plt.plot(L, median, '-o', markersize=4.0, label='median')
        plt.fill_between(L, median_sup, median_inf, alpha=0.5)
    else:
        plt.plot(L, mean, '-o', markersize=4.0, label='mean')
        plt.plot(L, median, '-o', markersize=4.0, label='median')
    plt.grid()
    plt.xlabel('L')
    plt.ylabel('$p_{\mathrm{avg}}$')
    plt.legend()
    plt.show()

    plt.figure()
    plt.loglog(L, std, '-o', markersize=4.0, label='mean-std')
    plt.loglog(L, median_sup - median_inf, '-o', markersize=4.0, label='median-ci')
    plt.grid()
    plt.xlabel('L')
    plt.ylabel('$p_{\mathrm{avg}}$ err')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(std, mean, '-o', markersize=4.0, label='mean-std')
    plt.grid()
    plt.xlabel('$\sigma$')
    plt.ylabel('$p_{\mathrm{avg}}$')
    plt.legend()
    plt.show()


def plot_std_fit(mean, std, slope, intercept):
    idx_sort = np.argsort(std)
    plt.figure()
    plt.title('$\sigma$ fit; intercept = {}'.format(intercept))
    plt.plot(std[idx_sort], mean[idx_sort], 'o', markersize=4.0, label='observations')
    plt.plot(std[idx_sort], slope*std[idx_sort] + intercept, '--', markersize=4.0, label='fit')
    plt.grid()
    plt.xlabel('$\sigma$')
    plt.ylabel('$p_{\mathrm{avg}}$')
    plt.legend()
    plt.show()


save_figures = False
ci = '99.99'

if not save_figures:
    plt.ion()

files_root_prefix = 'print/data/probability_sweep/v9/'
files = load_data.get_probability_sweep_file_list(files_root_prefix)
p_occupation, p_percolation, _, nsamples, L = load_data.load_probability_sweep_file_list(files)

mean = common.listop(common.cdf_mean, p_percolation, p_occupation)
var = common.listop(common.cdf_var, p_percolation, p_occupation)
std = common.listop(common.cdf_std, p_percolation, p_occupation)
p_percolation_bounds = common.listop(common.binomial_ci_wald, p_percolation, nsamples, args={'ci': ci})
p_percolation_inf = np.array([b[0] for b in p_percolation_bounds])
p_percolation_sup = np.array([b[1] for b in p_percolation_bounds])

median_results = common.varlistop(common.cdf_median, [p_percolation_inf, p_percolation_sup, p_occupation])
median = np.array([r[0] for r in median_results])
median_inf = np.array([r[1] for r in median_results])
median_sup = np.array([r[2] for r in median_results])

std_fit_slope, std_fit_intercept, _ = fit_std(mean, std)

plot_critical_probability_statistics(mean, std, median, median_inf, median_sup, L)
plot_std_fit(mean, std, std_fit_slope, std_fit_intercept)

idx = 15
plot_percolation_probability(p_occupation[idx], p_percolation[idx],
        p_percolation_inf[idx], p_percolation_sup[idx], L[idx], ci=ci)

if not save_figures:
    plt.ioff()
