import os, sys, glob, optparse, shutil, warnings
import numpy as np
np.random.seed(0)

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt

filename = "../../data/spectra/40minute/LRIS/lris20190903_ZTFJ1901.0.spec"
data_out = np.loadtxt(filename)

wavelengths = data_out[:,0]
flux = data_out[:,1]
sky = data_out[:,2]
fluxerr = data_out[:,3]

texp = 120
airmass = 1.5

log10S0 = [[5000, 15.978, np.nan, 3200, 15.721, 15.597],
        [5500, 16.381, 14.922, 3500, 15.934, 15.865],
        [6000, 16.465, 16.453, 4000, 16.226, 16.179],
        [6500, 16.487, 16.473, 4500, 16.270, 16.257],
        [7000, 16.488, 16.462, 5000, 16.239, 16.237],
        [7500, 16.456, 16.433, 5500, 16.167, 16.162],
        [8000, 16.391, 16.341, 5800, 16.109, 14.811],
        [8500, 16.310, 16.242, 6000, 16.063, np.nan],
        [9000, 16.229, 16.151, 6450, 15.940, np.nan],
        [9500, 15.963, 15.893, np.nan, np.nan, np.nan],
        [10000, 15.802, 15.396, np.nan, np.nan, np.nan]]

log10S0 = np.array(log10S0)
log10S_blue = np.interp(wavelengths,log10S0[:,3],log10S0[:,4])
log10S_red = np.interp(wavelengths,log10S0[:,0],log10S0[:,1])

cutoff = 5800
idx_blue = np.where(wavelengths < cutoff)[0]
idx_red = np.where(wavelengths > cutoff)[0] 

log10S0 = np.zeros((len(wavelengths),))
log10S0[idx_blue] = log10S_blue[idx_blue]
log10S0[idx_red] = log10S_red[idx_red]

K = [[3200, 0.866],
     [3500, 0.511],
     [4000, 0.311],
     [4500, 0.207],
     [5000, 0.153],
     [5500, 0.128],
     [6000, 0.113],
     [6450, 0.088],
     [6500, 0.085],
     [7000, 0.063],
     [7500, 0.053],
     [8000, 0.044],
     [8210, 0.043],
     [8260, 0.042],
     [8370, 0.041],
     [8708, 0.026],
     [10256, 0.020]]
K = np.array(K)
K_new = np.interp(wavelengths,K[:,0],K[:,1])

log10Flam = np.log10(flux) 
log10S = log10S0 + log10Flam + np.log10(texp) - 0.4*K_new*airmass

g = 2.5
sig = 2.5
SNR = g*(10**(log10S)) / np.sqrt(g*(10**(log10S)) + sig*sig) 

plt.figure(figsize=(8,6))
plt.plot(wavelengths, 10**log10S, 'k--')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('ADU / $\AA$')
plt.show()
plt.savefig('../../plots/LBT/counts.pdf',dpi=200)
plt.close('all')

plt.figure(figsize=(8,6))
plt.plot(wavelengths, SNR, 'k--')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('SNR')
plt.show()
plt.savefig('../../plots/LBT/SNR.pdf',dpi=200)
plt.close('all')

