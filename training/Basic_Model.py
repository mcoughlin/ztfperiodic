
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt

import ellc
import fast_histogram
from gatspy.periodic import LombScargleFast
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

def CE(period, data, xbins=10, ybins=5):
    """
    Returns the conditional entropy of *data* rephased with *period*.

    **Parameters**

    period : number
        The period to rephase *data* by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    xbins : int, optional
        Number of phase bins (default 10).
    ybins : int, optional
        Number of magnitude bins (default 5).
    """
    if period <= 0:
        return np.PINF

    #r = rephase(data, period)
    r = np.ma.array(data, copy=True)
    r[:, 0] = np.mod(r[:, 0], period) / period

    #bins, xedges, yedges = np.histogram2d(r[:,0], r[:,1], bins=[xbins, ybins], range=[[0,1], [0,1]])
    bins = fast_histogram.histogram2d(r[:,0], r[:,1], range=[[0, 1], [0, 1]], bins=[xbins, ybins])
    size = r.shape[0]

    if size > 0:
        # bins[i,j] / size
        divided_bins = bins / size
        # indices where that is positive
        # to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1) #changed 0 by 1

        # array is repeated row-wise, so that it can be sliced by arg_positive
        #column_sums = np.repeat(np.reshape(column_sums, (1,-1)), xbins, axis=0)
        column_sums = np.repeat(np.atleast_2d(column_sums).T, ybins, axis=1)

        # select only the elements in both arrays which correspond to a
        # positive bin
        select_divided_bins = divided_bins[arg_positive]
        select_column_sums  = column_sums[arg_positive]

        # initialize the result array
        A = np.empty((xbins, ybins), dtype=float)
        # store at every index [i,j] in A which corresponds to a positive bin:
        # bins[i,j]/size * log(bins[i,:] / size / (bins[i,j]/size))
        A[ arg_positive] = select_divided_bins \
                         * np.log(select_column_sums / select_divided_bins)
        # store 0 at every index in A which corresponds to a non-positive bin
        A[~arg_positive] = 0

        # return the summation
        return np.sum(A)
    else:
        return np.PINF

def basic_model(t,pars,grid='default'):

    try:
        m = ellc.lc(t_obs=t,
                radius_1=pars[0],
                radius_2=pars[1],
                sbratio=pars[2],
                incl=pars[3],
                t_zero=58664.0,
                q=pars[4],
                period=pars[6]/86400.0,
                shape_1='roche',
                shape_2='roche',
                t_exp=30/86400.0,
                grid_1=grid,
                grid_2=grid, heat_2=pars[5], exact_grav=False)
        m *= 1.0

    except:
        print("Failed with parameters:", pars)
        return t * 10**99

    return m

def synthetic_lightcurve(times,magnitudes,filterid):

    flux=10**(-0.4*magnitudes)

    period = 14.4 # minutes
    inclination = 8.57442638e+01

    if filterid=='g':
        sbratio = 2.22357777e-01
        sbratio = 0.8
        model_pars = [1.14720332e-01,1.25365731e-01,sbratio,inclination,1.0,1.90169100e+00,period*60.0]
    else:
        sbratio = 2.44108430e-01
        model_pars = [1.23843754e-01,1.19953610e-01,sbratio,inclination,1.0,2.27979385e+00,period*60.0]
    y=basic_model(times[:],model_pars)


    y=y/(np.max(y))
    y2=y*flux
    y3=-2.5*np.log10(y2)

    return y3

def time(n=100,mean_dt=3,sig_t=2):
    """
    Returns time array of length n with dt given by Gaussian distribution
    
    INPUT:
            n = length of time array
                Default: 100
                
            mean_dt = average dt between each time value
                Default: 3
                
            sig_t = standard deviation of the Gaussian
                Default: 2
            
    OUTPUT:
            t = time array
        
    """
    t = np.zeros(n)

    for i in range(len(t)):
        if i != 0:
            t[i] += t[i-1] + np.abs(np.random.normal(mean_dt,sig_t,1))

    return t

def period_search_ls(t, mag, magerr, remove_harmonics = True):
    ls = LombScargleFast(silence_warnings=True)
    T = np.max(t) - np.min(t)
    ls.optimizer.period_range = (0.01, T)
    ls.fit(t, mag, magerr)
    period = ls.best_period
    power = ls.periodogram(period)

    # https://github.com/astroML/gatspy/blob/master/examples/FastLombScargle.ipynb
    oversampling = 3.0
    N = len(t)
    df = 1. / (oversampling * T) # frequency grid spacing
    fmin = 2 / T
    fmax = 480 # minimum period is 0.05 d
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

    data_out = {}
    data_out["period"] = period
    data_out["significance"] = significance
    data_out["freqs"] = freqs
    data_out["powers"] = powers
    data_out["power"] = power
    data_out["Nztfobs"] = N
    data_out["fap_Neff"] = fap_Neff
    # data_out["fap_Baluev"] = fap_Baluev
    return data_out

def FAP_estimated(Z, N, fmax, t, normalization='standard'):
    """False Alarm Probability based on estimated number of indep frequencies"""
    T = max(t) - min(t)
    N_eff = fmax * T
    return 1 - P_single(Z, N, normalization=normalization) ** N_eff

def P_single(Z, N, normalization='standard'):
    """Cumulative Probability for a single observation"""
    return 1 - FAP_single(Z, N, normalization=normalization)

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


times = time(n=500, mean_dt=3, sig_t = 2)    
magnitudes = 16.0 * np.ones(times.shape)
filterid = 'g'

mag = synthetic_lightcurve(times, magnitudes, filterid)
magerr = 0.1*np.ones(times.shape)

plt.figure()
plt.plot(times, mag, 'kx')
plt.gca().invert_yaxis()
plt.grid()
plt.xlabel('Times [days]')
plt.ylabel('Magnitude [arb]')
plt.savefig('unfolded.pdf')
plt.close()

data_out = period_search_ls(times, mag, magerr, remove_harmonics = True)

plt.figure()
plt.plot(data_out["freqs"], data_out["powers"], 'k-')
plt.plot([1.0/data_out["period"],1.0/data_out["period"]],[np.min(data_out["powers"]),np.max(data_out["powers"])],'b--')
plt.grid()
plt.xlabel('Frequency [1/day]')
plt.ylabel('Power [arb]')
plt.savefig('lomb_scargle.pdf')
plt.close()

period = data_out["period"]
newphase = (times-times[0])/(period)%2
plt.figure()
plt.plot(newphase, mag, 'kx')
plt.gca().invert_yaxis()
plt.grid()
plt.xlabel('Phase')
plt.ylabel('Magnitude [arb]')
plt.savefig('folded_lomb_scargle.pdf')
plt.close()

phase_bins=20
mag_bins=50
periods = 1.0/data_out["freqs"]

data = np.vstack((times,mag))
copy = np.ma.copy(data).T
copy[:,1] = (copy[:,1] - np.min(copy[:,1])) / (np.max(copy[:,1]) - np.min(copy[:,1]))

period_true = 14.4 * 60.0/86400.0

baseline = max(copy[:,0])-min(copy[:,0])
fmin, fmax = 20.0, 480
samples_per_peak = 3
df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
nf = 1000
freqs = fmin + df * np.arange(nf)
freqs = 1.0/period_true + df * np.arange(-nf,nf+1)
periods = 1.0/freqs

#period_true = 14.4 * 60.0/86400.0
#print(period_true, data_out["period"])
entropy = CE(period_true, data=copy, xbins=phase_bins, ybins=mag_bins)
print(period_true, entropy)

entropies = []
print('Running CE...')
for ii, period in enumerate(periods):
    if np.mod(ii,10000) == 0:
        print('Iteration %d/%d' % (ii, len(periods)))
    entropy = CE(period, data=copy, xbins=phase_bins, ybins=mag_bins)
    entropies.append(entropy)

entropies = np.array(entropies)
period = periods[np.argmin(entropies)]

plt.figure()
plt.loglog(freqs, entropies, 'k-')
plt.plot([1.0/period,1.0/period],[np.min(entropies),np.max(entropies)],'b--')
plt.plot([1.0/period_true,1.0/period_true],[np.min(entropies),np.max(entropies)],'r--')
plt.grid()
plt.xlabel('Frequency [1/day]')
plt.ylabel('Entropy [arb]')
plt.savefig('CE.pdf')
plt.close()

period = periods[np.argmin(entropies)]
newphase = (times-times[0])/(period)%2
plt.figure()
plt.plot(newphase, mag, 'kx')
plt.gca().invert_yaxis()
plt.grid()
plt.xlabel('Phase')
plt.ylabel('Magnitude [arb]')
plt.savefig('folded_CE.pdf')
plt.close()

