import numpy as np


def power_allocation(P, step_depths):
    """Implementation of the geometrical water-filling algorithm.
       Given an array of ascending step depths and a power budget P,
       the power per step is computed"""
    K = len(step_depths)
    k = np.linspace(0, K-1, K).astype(int)
    P_2 = []
    for i in k:
        power_below_step_k = 0
        for j in range(i):
            power_below_step_k += (step_depths[i] - step_depths[j])
        power_above_step_k = P - power_below_step_k
        if power_above_step_k < 0:
            power_above_step_k = 0
        P_2.append(power_above_step_k)

    min_power = np.Inf
    k_star = 0
    for i, power in enumerate(P_2):
        if power != 0 and power < min_power:
            min_power = power
            k_star = i
    power_k_star = 1/(k_star+1)*min_power

    power = np.zeros(K)
    for i in range(k_star+1):
        power[i] = power_k_star + step_depths[k_star] - step_depths[i]
    return power


def sort_steps(step_depths):
    """Sorts the given step depths in ascending order, while saving their
       original order."""
    indices = step_depths.argsort()
    sorted_steps = step_depths[indices]
    return sorted_steps, indices


def rearrange_powers(power, indices):
    """Rearranges the given powers in the right order."""
    remapping_indices = np.zeros(len(indices))
    for i, ind in enumerate(indices):
        remapping_indices[ind] = i
    return power[remapping_indices.astype(int)]


def divide_spectrum(f, spectrum, num_bands):
    """Divides the given spectrum into a number of distinct frequency bands."""
    if len(f) != len(spectrum):
        raise ValueError("Spectrum and frequency axis should be equal in length.")
    band_width = int(len(spectrum)/num_bands)
    f_values = np.zeros(num_bands)
    spectrum_values = np.zeros(num_bands)
    for i in range(num_bands):
        f_values[i] = f[int(band_width*(0.5+i))]
        spectrum_values[i] = np.mean(spectrum[i*band_width:(i+1)*band_width])
    return f_values, spectrum_values




