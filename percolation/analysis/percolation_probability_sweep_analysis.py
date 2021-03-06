import importlib
import numpy as np
from scipy.integrate import simps
from scipy.stats import linregress
import matplotlib.pyplot as plt
import load_data
import common
importlib.reload(load_data)
importlib.reload(common)


def percolation_probability_ci(p_percolation, nsamples, ci='99.9'):
    p_percolation_min, p_percolation_max = common.binomial_ci_wilson(p_percolation, nsamples, ci=ci)
    return p_percolation_min, p_percolation_max


def percolation_probability_mean(p_occupation, p_percolation):
    p_mean = common.cdf_mean(p_percolation, p_occupation)
    p_var = common.cdf_var(p_percolation, p_occupation)
    return p_mean, p_var


def percolation_probability_median(p_occupation, p_percolation, nsamples, ci='99.9'):
    p_percolation_min, p_percolation_max = percolation_probability_ci(p_percolation, nsamples, ci=ci)
    p_percolation_median_min = p_occupation[(p_percolation_max >= 0.5).argmax()]
    p_percolation_median_max = p_occupation[(p_percolation_min >= 0.5).argmax()]
    p_percolation_median_err = (p_percolation_median_max - p_percolation_median_min)/2
    p_percolation_median_med = p_percolation_median_min + p_percolation_median_err
    return p_percolation_median_med, p_percolation_median_err


def percolation_probability_statistics(p_occupation, p_percolation, L, nsamples, ci='99.9'):
    data_mean = [percolation_probability_mean(p_occupation[i], p_percolation[i]) for i in range(L.size)]
    data_median = [percolation_probability_median(p_occupation[i], p_percolation[i], nsamples[i]) for i in range(L.size)]
    mean = np.array([d[0] for d in data_mean])
    var = np.array([d[1] for d in data_mean])
    median = np.array([d[0] for d in data_median])
    median_err = np.array([d[1] for d in data_median])
    return mean, var, median, median_err


def fit_std(p_occupation, p_percolation, nsamples):
    mean = np.array([common.cdf_mean(p_percolation[idx], p_occupation[idx]) for idx in range(len(p_percolation))])
    var = np.array([common.cdf_var(p_percolation[idx], p_occupation[idx]) for idx in range(len(p_percolation))])
    std = np.sqrt(var)
    idx_sort = np.argsort(std)
    idx_min = 0
    idx_max = 20
    slope, intercept, _, _, std_err = linregress(std[idx_sort][idx_min:idx_max], mean[idx_sort][idx_min:idx_max])
    return slope, intercept, std_err


def plot_percolation_probability(p_occupation, p_percolation, L, nsamples, ci='99.9'):
    p_percolation_min, p_percolation_max = percolation_probability_ci(p_percolation, nsamples, ci)
    plt.figure()
    plt.title('Percolation probability for 2D square lattice with L = {}'.format(L))
    plt.plot(p_occupation, p_percolation, '-', color='royalblue', linewidth=1.0, label='$p_{\mathrm{percolation}}$')
    plt.fill_between(p_occupation, p_percolation_max, p_percolation_min, facecolor='royalblue', alpha=0.5, label='{}% CI'.format(ci))
    plt.grid()
    plt.xlabel('$p_{\mathrm{occupation}}$')
    plt.ylabel('$p_{\mathrm{percolation}}$')
    plt.legend()
    plt.show()


def plot_critical_probability_statistics(p_occupation, p_percolation, L, nsamples, ci='99.9', ci_style='none'):
    mean, var, median, median_err = percolation_probability_statistics(p_occupation, p_percolation, L, nsamples, ci=ci)
    std = np.sqrt(var)
    plt.figure()
    if ci_style == 'errorbar':
        plt.errorbar(L, mean, yerr=std, fmt='-o', label='mean')
        plt.errorbar(L, median, yerr=median_err, fmt='-o', label='median')
    elif ci_style == 'area':
        plt.plot(L, mean, '-o', markersize=4.0, label='mean')
        plt.fill_between(L, mean + std, mean - std, alpha=0.5)
        plt.plot(L, median, '-o', markersize=4.0, label='median')
        plt.fill_between(L, median + median_err, median - median_err, alpha=0.5)
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
    plt.loglog(L, median_err, '-o', markersize=4.0, label='median-ci')
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


def plot_std_fit(p_occupation, p_percolation, nsamples):
    mean = np.array([common.cdf_mean(p_percolation[idx], p_occupation[idx]) for idx in range(len(p_percolation))])
    var = np.array([common.cdf_var(p_percolation[idx], p_occupation[idx]) for idx in range(len(p_percolation))])
    std = np.sqrt(var)
    slope, intercept, _ = fit_std(p_occupation, p_percolation, nsamples)

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

if not save_figures:
    plt.ion()

files_root_prefix = 'print/data/probability_sweep/v5/'
files = load_data.get_probability_sweep_file_list(files_root_prefix)
p_occupation, p_percolation, _, nsamples, L = load_data.load_probability_sweep_file_list(files)
plot_critical_probability_statistics(p_occupation, p_percolation, L, nsamples)
plot_std_fit(p_occupation, p_percolation, nsamples)

plot_percolation_probability(p_occupation[15], p_percolation[15], L[15], nsamples[15])

if not save_figures:
    plt.ioff()
