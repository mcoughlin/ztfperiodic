#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 07:44:47 2019

@author: yuhanyao
"""
import numpy as np
from scipy.special import gammaln
from gatspy.periodic import LombScargleFast


def FAP_single(Z, N, normalization='standard'):
    """False Alarm Probability for a single observation"""
    NH = N - 1  # DOF for null hypothesis
    NK = N - 3  # DOF for periodic hypothesis
    if normalization == 'psd':
        return np.exp(-Z)
    elif normalization == 'standard':
        # Note: astropy's standard normalization is Z = 2/NH * z_1 in Baluev's terms
        return (1 - Z) ** (NK / 2)
    elif normalization == 'model':
        # Note: astropy's model normalization is Z = 2/NK * z_2 in Baluev's terms
        return (1 + Z) ** -(NK / 2)
    elif normalization == 'log':
        # Note: astropy's log normalization is Z = 2/NK * z_3 in Baluev's terms
        return np.exp(-0.5 * NK * Z)
    else:
        raise NotImplementedError("normalization={0}".format(normalization))


def P_single(Z, N, normalization='standard'):
    """Cumulative Probability for a single observation"""
    return 1 - FAP_single(Z, N, normalization=normalization)


def FAP_estimated(Z, N, fmax, t, normalization='standard'):
    """False Alarm Probability based on estimated number of indep frequencies"""
    T = max(t) - min(t)
    N_eff = fmax * T
    return 1 - P_single(Z, N, normalization=normalization) ** N_eff


def weighted_sum(val, dy):
    return (val / dy ** 2).sum()


def weighted_mean(val, dy):
    return weighted_sum(val, dy) / weighted_sum(np.ones_like(val), dy)


def weighted_var(val, dy):
    return weighted_mean(val ** 2, dy) - weighted_mean(val, dy) ** 2


def gamma(N):
    # Note: this is closely approximated by (1 - 1 / N) for large N
    return np.sqrt(2 / N) * np.exp(gammaln(N / 2) - gammaln((N - 1) / 2))


def tau_davies(Z, N, fmax, t, y, dy, normalization='standard'):
    """tau factor for estimating Davies bound (see Baluev 2008, Table 1)"""
    # Variable names follow the discussion in Baluev 2008
    NH = N - 1  # DOF for null hypothesis
    NK = N - 3  # DOF for periodic hypothesis
    Dt = weighted_var(t, dy)
    Teff = np.sqrt(4 * np.pi * Dt)
    W = fmax * Teff
    if normalization == 'psd':
        return W * np.exp(-Z) * np.sqrt(Z)
    elif normalization == 'standard':
        # Note: astropy's standard normalization is Z = 2/NH * z_1 in Baluev's terms
        return gamma(NH) * W * (1 - Z) ** (0.5 * (NK - 1)) * np.sqrt(NH * Z / 2)
    elif normalization == 'model':
        # Note: astropy's model normalization is Z = 2/NK * z_2 in Baluev's terms
        return gamma(NK) * W * (1 + Z) ** (- 0.5 * NK) * np.sqrt(NK * Z / 2)
    elif normalization == 'log':
        # Note: astropy's log normalization is Z = 2/NK * z_3 in Baluev's terms
        return gamma(NK) * W * np.exp(-0.5 * Z * (NK - 0.5)) * np.sqrt(NK * np.sinh(0.5 * Z))
    else:
        raise NotImplementedError("normalization={0}".format(normalization))

    
def FAP_davies(Z, N, fmax, t, y, dy, normalization='standard'):
    """Davies bound (Eqn 5 of Baluev 2008)"""
    FAP_s = FAP_single(Z, N, normalization=normalization)
    tau = tau_davies(Z, N, fmax, t, y, dy, normalization=normalization)
    return FAP_s + tau


def FAP_aliasfree(Z, N, fmax, t, y, dy, normalization='standard'):
    """Alias-free approximation to FAP (Eqn 6 of Baluev 2008)"""
    P_s = P_single(Z, N, normalization=normalization)
    tau = tau_davies(Z, N, fmax, t, y, dy, normalization=normalization)
    return 1 - P_s * np.exp(-tau)


def period_search_ls(t, mag, magerr, data_out, remove_harmonics = True):
    ls = LombScargleFast(silence_warnings=True)
    T = np.max(t) - np.min(t)
    ls.optimizer.period_range = (0.1, T)
    ls.fit(t, mag, magerr)
    # period = ls.best_period
    # power = ls.periodogram(period)
    
    # https://github.com/astroML/gatspy/blob/master/examples/FastLombScargle.ipynb
    oversampling = 2
    N = len(t)
    df = 1. / (oversampling * T) # frequency grid spacing
    fmin = 1 / T
    fmax = 10 # minimum period is 0.05 d
    Nf = (fmax - fmin) // df
    freqs = fmin + df * np.arange(Nf)
    periods = 1 / freqs
    powers = ls._score_frequency_grid(fmin, df, Nf)
    ind_best = np.argsort(powers)[-1]
    period = periods[ind_best]
    power = powers[ind_best]
    
    # calcualte false alarm probability (FAP)
    Z = power
    normalization='standard'
    # fap_Neff is underestimate
    fap_Neff = FAP_estimated(Z, N, fmax, t, normalization=normalization)
    """
    # fap_Baluev is overestimate
    fap_Baluev = FAP_aliasfree(Z, N, fmax, t, mag, magerr, normalization=normalization)
    """
    psigma = (np.percentile(powers, 84) - np.percentile(powers, 16))/2
    significance = power / psigma
    
    if remove_harmonics == True:
        # In some cases, the period search is not successful:
        harmonics = np.array([1/5, 1/4, 1/3, 1/2, 1., 2.])
        if abs(period - T)<0.005:
            period = -99
        else:
            for harmonic in harmonics:
                if abs(period - harmonic)<0.005:
                    if fap_Neff > 0.001:
                        period = -99
    
    data_out["period"] = period
    data_out["significance"] = significance
    data_out["freqs"] = freqs
    data_out["powers"] = powers
    data_out["power"] = power
    data_out["Nztfobs"] = N
    data_out["fap_Neff"] = fap_Neff
    # data_out["fap_Baluev"] = fap_Baluev
    return data_out