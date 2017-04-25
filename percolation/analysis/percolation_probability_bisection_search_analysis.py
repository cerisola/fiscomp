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


def plot_percolation_probability_statistics(mean, std, sem, L, ci='', ci_style='none'):
    plt.figure()
    if ci_style == 'errorbar':
        plt.errorbar(L, mean, yerr=sem, fmt='-o', label='mean ({}% CI)'.format(ci))
    elif ci_style == 'area':
        plt.plot(L, mean, '-o', markersize=4.0, label='mean')
        plt.fill_between(L, mean + sem, mean - sem, alpha=0.5, label='{}% CI'.format(ci))
    else:
        plt.plot(L, mean, '-o', markersize=4.0, label='mean')
    plt.grid()
    plt.xlabel('L')
    plt.ylabel('$p_{\mathrm{avg}}$ via bisection')
    plt.legend()
    plt.show()

    plt.figure()
    plt.loglog(L, std, '-o', markersize=4.0, label='mean-std')
    plt.loglog(L, sem, '-o', markersize=4.0, label='mean-sem')
    plt.grid()
    plt.xlabel('L')
    plt.ylabel('$p_{\mathrm{avg}}$ err')
    plt.legend()
    plt.show()


def plot_bisection_search_results(p_percolation, L, ntrials):
    plt.figure()
    plt.title('Bisection search of $p_c$ for L = {}, avg over {} trials'.format(L, ntrials))
    plt.plot(p_percolation, 'o', markersize=2.0)
    plt.grid()
    plt.xlabel('iter')
    plt.ylabel('p')
    plt.show()


def plot_bisection_search_histogram(p_percolation, L, ntrials):
    plt.figure()
    plt.title('Bisection search results histogram for L = {}'.format(L))
    plt.hist(p_percolation, bins=90)
    plt.show()


def plot_bisection_search_histogram_list(p_percolation, L, ntrials, cumulative=True):
    nvalues = [(np.unique(p_percolation[idx])).size for idx in range(len(L))]
    std = common.listmap(common.std, p_percolation)
    plt.figure()
    plt.title('Bisection search results histogram')
    for idx in range(L.size):
        plt.hist(p_percolation[idx], bins=nvalues[idx]//50, normed=True, cumulative=False,
                 histtype='step', linewidth=1.2, alpha=0.8, label='L = {}'.format(L[idx]))
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Bisection search results histogram')
    for idx in range(L.size):
        hist, bins = np.histogram(p_percolation[idx], bins=(int)(nvalues[idx]/std[idx]/2000), density=True)
        #widths = width = np.diff(bins)
        #centers = (bins[:-1] + bins[1:]) / 2
        #plt.step(centers, hist, '-', label='L = {}'.format(L[idx]))
        bins = [0] + list(bins) + [1]
        plt.hist(p_percolation[idx], bins=bins, normed=True, cumulative=False,
                histtype='step', linewidth=1.2, alpha=0.8, label='L = {}'.format(L[idx]))
    plt.xlim((0.4, 0.75))
    plt.grid()
    plt.legend()
    plt.show()

    if cumulative:
        plt.figure()
        plt.title('Bisection search results cumulative function histogram')
        for idx in range(L.size):
            plt.hist(p_percolation[idx], bins=nvalues[idx], normed=True, cumulative=True,
                     histtype='step', linewidth=1.1, label='L = {}'.format(L[idx]),
                     zorder=3)
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('Bisection search results histogram')
        for idx in range(L.size):
            hist, bins = np.histogram(p_percolation[idx], bins=nvalues[idx], density=True)
            #widths = width = np.diff(bins)
            #centers = (bins[:-1] + bins[1:]) / 2
            #plt.step(centers, hist.cumsum()/hist.sum(), '-', label='L = {}'.format(L[idx]))
            bins = [0] + list(bins) + [1]
            plt.hist(p_percolation[idx], bins=bins, normed=True, cumulative=True,
                histtype='step', linewidth=1.1, label='L = {}'.format(L[idx]),
                zorder=3)
        plt.xlim((0.4, 0.8))
        plt.grid()
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

files_root_prefix = 'print/data/critical_bisection_search/v4/'
files = load_data.get_bisection_critical_search_file_list(files_root_prefix)
p_percolation, L, ntrials = load_data.load_bisection_critical_search_file_list(files)

mean = common.listmap(common.mean, p_percolation)
var = common.listmap(common.var, p_percolation)
std = np.sqrt(var)
sem = common.listmap(common.sem, p_percolation, args={'ci': ci})

std_fit_slope, std_fit_intercept, _ = fit_std(mean, std)

plot_percolation_probability_statistics(mean, std, sem, L, ci=ci, ci_style='area')
plot_bisection_search_results(p_percolation[4], L[4], ntrials[4])
plot_std_fit(mean, std, std_fit_slope, std_fit_intercept)

idx_hist = [9, 22, 30, -1]
plot_bisection_search_histogram_list([p_percolation[idx] for idx in idx_hist], L[idx_hist], ntrials[idx_hist])

if not save_figures:
    plt.ioff()
