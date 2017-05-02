import numpy as np

# % Single value % #
def cluster_densities_finite(count, percolated, realizations, L):
    return (count - percolated)/realizations/(L*L)


def percolating_cluster_mass(size, percolated, realizations):
    return np.sum(size*percolated)/realizations


def percolating_cluster_density(size, percolated, realizations, L):
    return percolating_cluster_mass(size, percolated, realizations)/(L*L)


def percolating_cluster_strength(size, percolated, realizations, L):
    return percolating_cluster_mass(size, percolated, realizations)/(L*L)


# % Multiple values (list) % #
def cluster_densities_finite_list(count, percolated, realizations, L):
    data = [cluster_densities_finite(count[i], percolated[i], realizations[i], L) for i in range(len(count))]
    return data


def percolating_cluster_mass_list(size, percolated, realizations):
    data = [percolating_cluster_mass(size[i], percolated[i], realizations[i]) for i in range(len(size))]
    return np.array(data)


def percolating_cluster_density_list(size, percolated, realizations, L):
    return percolating_cluster_mass_list(size, percolated, realizations)/(L*L)


def percolating_cluster_strength_list(size, percolated, realizations, L):
    return percolating_cluster_mass_list(size, percolated, realizations)/(L*L)
