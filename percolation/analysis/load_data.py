import re
import glob
from operator import itemgetter
import numpy as np

## %% Load Files %% ##

# % Cluster statistics % #
def get_cluster_statistics_file_list(path, L=None, p_occupation=None):
    files_full_path = path + '/clusters_*'
    if L:
        files_full_path += '{}x{}_*'.format(L, L)
    if p_occupation:
        if type(p_occupation) == type(''):
            files_full_path += '{}*'.format(p_occupation)
        else:
            files_full_path += '{:e}*'.format(p_occupation)
    files_full_path += '.csv'
    return glob.glob(files_full_path)


def load_cluster_statistics_file(file_name):
    L = 0
    p_occupation = 0
    nrealizations = 0
    size = []
    count = []
    percolated = []
    with open(file_name) as f:
        for line in f:
            if line.startswith(';rows'):
                m = re.search(';rows:(?P<L>\d+)', line)
                L = int(m.group(1))
            elif line.startswith(';probability'):
                m = re.search(r';probability:\b(?P<probability>.+)\b', line)
                p_occupation = float(m.group(1))
            elif line.startswith(';realizations'):
                m = re.search(r';realizations:(?P<L>\d+)', line)
                nrealizations = int(m.group(1))
            elif not line.startswith(';'):
                columns = line.split(',')
                size.append(int(columns[0]))
                count.append(int(columns[1]))
                percolated.append(int(columns[2]))
    size = np.asarray(size)
    count = np.asarray(count)
    percolated = np.asarray(percolated)
    idx = np.argsort(size)
    return size[idx], count[idx], percolated[idx], L, p_occupation, nrealizations


def load_cluster_statistics_file_list(files):
    data = [load_cluster_statistics_file(f) for f in files]
    data = sorted(data, key=itemgetter(3, 4))
    size = [d[0] for d in data]
    count = [d[1] for d in data]
    percolated = [d[2] for d in data]
    L = np.array([d[3] for d in data])
    p_occupation = np.array([d[4] for d in data])
    nrealizations = np.array([d[5] for d in data])
    return size, count, percolated, L, p_occupation, nrealizations


# % Probability Sweep % #
def get_probability_sweep_file_list(path, L=None, p_occupation=None):
    files_full_path = path + '/percolation_probability_*'
    if L:
        files_full_path += '{}x{}_*'.format(L, L)
    if p_occupation:
        if type(p_occupation) == type(''):
            files_full_path += '{}*'.format(p_occupation)
        else:
            files_full_path += '{:e}*'.format(p_occupation)
    files_full_path += '.csv'
    return glob.glob(files_full_path)


def load_probability_sweep_file(file_name):
    L = 0
    p_occupation = []
    n_percolation = []
    nsamples = []
    nsamples_common = 0
    p_percolation = []
    with open(file_name) as f:
        for line in f:
            if line.startswith(';rows'):
                m = re.search(';rows:(?P<L>\d+)', line)
                L = int(m.group(1))
            elif line.startswith(';realizations'):
                m = re.search(';realizations:(?P<realizations>\d+)', line)
                nsamples_common = int(m.group(1))
            elif not line.startswith(';'):
                columns = line.split(',')
                p_occupation.append(float(columns[0]))
                n_percolation.append(int(columns[1]))
                if len(columns) == 3:
                    nsamples.append(nsamples_common)
                    p_percolation.append(float(columns[2]))
                else:
                    nsamples.append(int(columns[2]))
                    p_percolation.append(float(columns[3]))
    p_occupation = np.asarray(p_occupation)
    n_percolation = np.asarray(n_percolation)
    nsamples = np.asarray(nsamples)
    p_percolation = np.asarray(p_percolation)
    return p_occupation, p_percolation, n_percolation, nsamples, L


def load_probability_sweep_file_list(files):
    data = [load_probability_sweep_file(f) for f in files]
    data = sorted(data, key=itemgetter(4))
    p_occupation = [d[0] for d in data]
    p_percolation = [d[1] for d in data]
    n_percolation = [d[2] for d in data]
    nsamples = [d[3] for d in data]
    L = np.array([d[4] for d in data])
    return p_occupation, p_percolation, n_percolation, nsamples, L


# % Bisection Search % #
def get_bisection_critical_search_file_list(path, L=None):
    files_full_path = path + '/critical_bisection_search_*'
    if L:
        files_full_path += '{}x{}*'.format(L, L)
    files_full_path += '.csv'
    return glob.glob(files_full_path)


def load_bisection_critical_search_file(file_name):
    L = 0
    ntrials = 0
    critical_probability = []
    with open(file_name) as f:
        for line in f:
            if line.startswith(';rows'):
                m = re.search(';rows:(?P<L>\d+)', line)
                L = int(m.group(1))
            elif line.startswith(';ntrials'):
                m = re.search(';ntrials:(?P<L>\d+)', line)
                ntrials = int(m.group(1))
            elif not line.startswith(';'):
                critical_probability.append(float(line))
    critical_probability = np.asarray(critical_probability)
    return critical_probability, L, ntrials


def load_bisection_critical_search_file_list(files):
    data = [load_bisection_critical_search_file(f) for f in files]
    data = sorted(data, key=itemgetter(1, 2))
    critical_probability = [d[0] for d in data]
    L = np.array([d[1] for d in data])
    ntrials = np.array([d[2] for d in data])
    return critical_probability, L, ntrials
