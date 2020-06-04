
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import AOV_cython

data_out = np.loadtxt('photometry.dat')

times, mags, errs = data_out[:,0], data_out[:,1], data_out[:,2]
npts = len(times)

magvariance_top = np.sum(mags/(errs*errs))
magvariance_bot = (mags.size - 1)*np.sum(1.0/(errs*errs)) / mags.size
magvariance = magvariance_top/magvariance_bot

frequency_true = 1.0/(2*0.014098)
nharmonics = 1

baseline = times[-1] - times[0]
fmin, fmax = 2/baseline, 48

samples_per_peak = 3
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))

freqs = fmin + df * np.arange(nf)
stats = np.zeros(len(freqs))

for ii, frequency in enumerate(freqs):
    theta_aov = AOV_cython.compute(frequency, times, mags,
                                   np.mean(mags), npts, 10)
    stats[ii] = theta_aov
    if np.mod(ii, 1000) == 0:
        print('%d/%d' % (ii, len(freqs)))

stats[np.isnan(stats)] = 1e-3
stats[stats<0] = 1e-3

plt.figure()
plt.loglog(1./freqs, stats)
plt.plot([1.0/frequency_true, 1.0/frequency_true], [np.min(stats), np.max(stats)], 'k--')
plt.savefig('test.png')
plt.close()


