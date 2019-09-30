#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:56:26 2019

@author: yuhanyao
"""
import numpy as np
from scipy import fftpack
from scipy.signal import correlate
from scipy.interpolate import interp1d


def correlate_spec(spectral_data, band = [6475.0, 6650.0]):
    """
    Cross correlate the spectra
    
    Method: Tonry & Davis (1979)
    I followed steps given by Blondin et al. (2007)
    """
    mykeys = np.array(list(spectral_data.keys()))
    correlation_funcs = {}
    if len(mykeys)<=1:
        return correlation_funcs
    else:
        xmin, xmax = band[0], band[1]
        spectral_chunks = {}
        for key in spectral_data:
            idx = np.where((spectral_data[key]["lambda"] >= xmin) &
                           (spectral_data[key]["lambda"] <= xmax))[0]
            mywave = spectral_data[key]["lambda"][idx]
            myflux = spectral_data[key]["flux"][idx]
            # quick-and-dirty normalization
            myflux -= np.median(myflux)
            myflux /= np.percentile(abs(myflux), 90)
            
            spectral_chunks[key] = {}
            spectral_chunks[key]["lambda"] = mywave
            spectral_chunks[key]["flux"] = myflux
        count = 0
        for ii in mykeys:
            for jj in mykeys[ii+1:]:
                wv1 = spectral_chunks[ii]["lambda"]
                fl1 = spectral_chunks[ii]["flux"]
                wv2 = spectral_chunks[jj]["lambda"]
                fl2 = spectral_chunks[jj]["flux"]
                """
                plt.plot(wv1, fl1)
                plt.plot(wv2, fl2)
                """
                # N bins, I want an even number
                N = len(wv1) 
                N = N//2 * 2
                lambda0 = max(wv1[0], wv2[0])+1
                lambda1 = min(wv1[-1], wv2[-1])-1
                
                dlnwv = np.log(lambda1/lambda0)/N
                n = np.arange(N+1)
                x = lambda0 * np.exp(n * dlnwv)
                """
                A = N / np.log(lambda1/lambda0)
                B = -N * np.log(lambda0) / np.log(lambda1/lambda0)
                n = A * np.log(x) + B
                This should give you ~ the same values of n as above
                """
                # interpolate the spectrum
                func1 = interp1d(wv1, fl1)
                func2 = interp1d(wv2, fl2)
                y1 = func1(x)
                y2 = func2(x)
                """
                plt.plot(x, y1)
                plt.plot(x, y2)
                """
                # take the FFT of both signals
                Y1f = fftpack.fft(y1)
                Y2f = fftpack.fft(y2)
                Y1fr = -Y1f.conjugate()
                Y2fr = -Y2f.conjugate()
                sigma1 = np.sqrt(1/N * np.sum(y1**2))
                sigma2 = np.sqrt(1/N * np.sum(y2**2))
                # Fourier Transform of the correlation function: C(k)
                # transfer back to c(n)
                Cn = np.abs(fftpack.ifft(Y1fr*Y2f)) / (sigma1 * sigma2 * N)
                # Cn_ = np.abs(fftpack.ifft(Y1f*Y2fr)) / (sigma1 * sigma2 * N)
                nsymmetry = n
                nsymmetry[N//2+1:] -= (N+1)
                v = nsymmetry * dlnwv * 3e+5 # in km / s
                ix = np.argsort(nsymmetry)
                vnew = v[ix]
                Cvnew = Cn[ix]
                """
                plt.plot(vnew, Cvnew)
                """
                correlation_funcs[count] = {}
                correlation_funcs[count]["velocity"] = vnew
                correlation_funcs[count]["correlation"] = Cvnew
                count += 1
                
        return correlation_funcs
                    
                