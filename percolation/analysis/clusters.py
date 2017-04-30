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
    data = [cluster_densities(count[i], percolated[i], realizations[i], L[i]) for i in range(L.size)]
    return data


def percolating_cluster_mass_list(size, percolated):
    data = [percolating_cluster_mass(size[i], percolated[i]) for i in range(len(size))]
    return np.array(data)


def percolating_cluster_density_list(size, percolated, L):
    return percolating_cluster_mass_list(size, percolated)/(L*L)


def percolating_cluster_strength_list(size, percolated, L):
    return percolating_cluster_mass_list(size, percolated)/(L*L)


# % Averaged values over many realizations % #
def percolating_cluster_mass_average(size, percolated, p_percolation):
    return percolating_cluster_mass_list(size, percolated) * p_percolation


def percolating_cluster_density_average(size, percolated, p_percolation, L):
    return percolating_cluster_mass_average(size, percolated, p_percolation)/(L*L)


def percolating_cluster_strength_average(size, percolated, p_percolation, L):
    return percolating_cluster_mass_average(size, percolated, p_percolation)/(L*L)

