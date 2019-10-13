#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:56:26 2019

@author: yuhanyao
"""
import glob
import numpy as np
from copy import deepcopy
from scipy import fftpack
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.time import Time
import astropy.constants as const
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def correlate_spec(spectral_data, band = [6475.0, 6650.0], 
                   period = None, plot_figure = False):
    """
    Cross correlate the spectra
    
    Method: Tonry & Davis (1979)
    I followed steps given by Blondin et al. (2007)
    For uncertainty measurement, I adopt the assumption in Kurtz & Mink (1998)
    """
    mykeys = np.array(list(spectral_data.keys()))
    correlation_funcs = {}
    if len(mykeys)<=1:
        return correlation_funcs
    else:
        xmin, xmax = band[0], band[1]
        spectral_chunks = {}
        for key in mykeys:
            spectral_chunks[key] = {}
            idx = np.where((spectral_data[key]["lambda"] >= xmin) &
                           (spectral_data[key]["lambda"] <= xmax))[0]
            mywave = spectral_data[key]["lambda"][idx]
            myflux = spectral_data[key]["flux"][idx]
            # quick-and-dirty normalization
            myflux -= np.median(myflux)
            if len(myflux) == 0: 
                spectral_chunks[key]["lambda"] = []
                spectral_chunks[key]["flux"] = []
            else:
                myflux /= np.percentile(abs(myflux), 90)
                spectral_chunks[key]["lambda"] = mywave
                spectral_chunks[key]["flux"] = myflux
        count = 0
        for ii in mykeys:
            for jj in mykeys[ii+1:]:
                wv1 = spectral_chunks[ii]["lambda"]
                fl1 = spectral_chunks[ii]["flux"]
                wv2 = spectral_chunks[jj]["lambda"]
                fl2 = spectral_chunks[jj]["flux"]

                if (len(wv1) == 0) or (len(wv2) == 0):
                    correlation_funcs[count] = {}
                    correlation_funcs[count]["velocity"] = []
                    correlation_funcs[count]["correlation"] = []
                    count += 1
                    continue

                if plot_figure == True:
                    plt.figure(figsize=(10, 10))
                    ax1 = plt.subplot(311)
                    ax1.plot(wv1, fl1, '.-', label='input spec 1', color='c')
                    ax1.plot(wv2, fl2, '.-', label='input spec 2', color='r')
                    ax1.set_xlabel("Wavelength [Angstrom]")
                    ax1.legend()
                # N bins, I want an even number
                N = len(wv1) 
                N = N//2 * 20
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
                func1 = interp1d(wv1, fl1, fill_value="extrapolate")
                func2 = interp1d(wv2, fl2, fill_value="extrapolate")
                y1 = func1(x)
                y2 = func2(x)
                if plot_figure==True:
                    ax2 = plt.subplot(312)
                    ax2.plot(n, y1, '.-', label='t(n), N=%d'%N, color='c')
                    ax2.plot(n, y2, '.-', label='s(n)', color='r')
                    ax2.set_xlabel("n")
                # take the FFT of both signals
                Y1f = fftpack.fft(y1)
                Y2f = fftpack.fft(y2)
                Y1fr = -Y1f.conjugate()
                # Y2fr = -Y2f.conjugate()
                sigma1 = np.sqrt(1/N * np.sum(y1**2))
                sigma2 = np.sqrt(1/N * np.sum(y2**2))
                # Fourier Transform of the correlation function: C(k)
                # transfer back to c(n)
                Cn = np.abs(fftpack.ifft(Y1fr*Y2f)) / (sigma1 * sigma2 * N)
                # Cn_ = np.abs(fftpack.ifft(Y1f*Y2fr)) / (sigma1 * sigma2 * N)
                nsymmetry = deepcopy(n)
                nsymmetry[N//2+1:] -= (N+1)
                v = nsymmetry * dlnwv * 3e+5 # in km / s
                ix = np.argsort(nsymmetry)
                vnew = v[ix]
                Cvnew = Cn[ix]
                nnew = nsymmetry[ix]
                # only retain -2000 < v < 2000 km/s
                
                if plot_figure == True:
                    ax3 = plt.subplot(325)
                    ax3.plot(nnew, Cvnew, '-', color='k', label='c(n)=s(n)'+r'$\ast$'+'t(n)')
                    ax3.set_xlabel('n')
                    # ax3.set_xlim(-200,200)
                    ax4 = plt.subplot(326)
                    ax4.plot(vnew, Cvnew, '-', color='k')
                    ax4.set_xlabel("v [km/s]")
                    plt.tight_layout()
                
                C_peak = max(Cvnew)
                ind_peak = np.where(Cvnew == C_peak)[0][0]
                v_peak = vnew[ind_peak]
                n_peak = nnew[ind_peak]
                alpha_max = sigma2 / sigma1 * C_peak
                
                if n_peak<0:
                    y1s = np.hstack([y1[-n_peak:], y1[:-n_peak]])*alpha_max# y1 shifted
                else:
                    y1s = np.hstack([y1[n_peak:], y1[:n_peak]])*alpha_max# y1 shifted
                sigma1s = np.sqrt(1/N * np.sum(y1s**2))
                Y1sf = fftpack.fft(y1s)
                Cn_ = np.abs(fftpack.ifft(Y1fr*Y1sf)) / (sigma1 * sigma1s * N)
                nsymmetry_ = deepcopy(n)
                nsymmetry_[N//2+1:] -= (N+1)
                v_ = nsymmetry_ * dlnwv * 3e+5 # in km / s
                ix_ = np.argsort(nsymmetry_)
                vnew_ = v_[ix_]
                Cvnew_ = Cn_[ix_]
                nnew_ = nsymmetry_[ix_]
                
                ######### calculated uncertainty in delta
                avnew = Cvnew - Cvnew_
                sigma_a = np.std(avnew)
                r_value = C_peak / (np.sqrt(2) * sigma_a)
                # FWHM of the correlation peak
                ixright = nnew > n_peak
                nright = nnew[ixright]
                Cright = Cvnew[ixright]
                nleft = nnew[~ixright]
                Cleft = Cvnew[~ixright]
                id_right = np.where(Cright<0.5*C_peak)[0][0]
                if abs(Cright[id_right]-0.5*C_peak) > abs(Cright[id_right-1]-0.5*C_peak):
                    id_right = id_right-1
                id_left = np.where(Cleft<0.5*C_peak)[0][-1]
                if abs(Cleft[id_left]-0.5*C_peak) > abs(Cleft[id_left+1]-0.5*C_peak):
                    id_right = id_right+1
                width = nright[id_right] - nleft[id_left]
                sigma_n = 3*width/(8*(1+r_value))
                # convert sigma_n to sigma_v
                sigma_v = sigma_n * dlnwv * 3e+5
                
                if plot_figure==True:
                    ax2.plot(n, y1s, '-', color='k', label=r'$\alpha$'+'t(n-'+r'$\delta$'+')')
                    ax2.legend()
                    ax3.plot(nnew_, Cvnew_, '--', color='m', label=r'$\alpha$'+'t(n-'+r'$\delta$'+')'+r'$\ast$'+'t(n)')
                    ax3.plot(nnew_, avnew, color='b', label='a(n)')
                    ax3.plot([nleft[id_left], nright[id_right]], [0.5, 0.5], 'k:')
                    ax3.legend(frameon=False)
                    ax3.set_title('r=%.2f'%r_value)
                    ax4.text(0, 0.5, "v = %.0f +- %.0f"%(v_peak, sigma_v))
                    ax4.set_xlim(-1000, 1000)
                    ax4.xaxis.set_minor_locator(AutoMinorLocator())
                    ax4.yaxis.set_minor_locator(AutoMinorLocator())
                    ax4.tick_params(direction='in', axis='both', which = 'both', top=True, right=True)
                    ax4.tick_params(which='major', length=4)
                    ax4.tick_params(which='minor', length=2)
                    # add mass function x-axis
                    if period!=None:
                        new_tick_locations = np.array([-900, -600, -300, 0, 300, 600, 900])
                        ax4_ = ax4.twiny()
                        ax4_.set_xlim(ax4.get_xlim())
                        ax4_.set_xticks(new_tick_locations)
                        ax4_.set_xticklabels(tick_function(new_tick_locations, period))
                        ax4_.set_xlabel("f($M$) ("+r'$M_\odot$'+')')
                correlation_funcs[count] = {}
                correlation_funcs[count]["velocity"] = vnew
                correlation_funcs[count]["correlation"] = Cvnew
                correlation_funcs[count]["v_peak"] = v_peak
                correlation_funcs[count]["v_peak_unc"] = sigma_v
                correlation_funcs[count]["C_peak"] = C_peak
                correlation_funcs[count]["r_value"] = r_value
                count += 1
                
        return correlation_funcs
    
    
def tick_function(X, period):
    K = X/2. # [km/s] assuming that the velocity variation is max and min in rv curve
    P = 2*period # [day] if ellipsodial modulation, amplitude are roughly the same, 
                    # then the photometric period is probably half of the orbital period
    fmass = (K * 100000)**3 * (P*86400) / (2*np.pi*const.G.cgs.value) / const.M_sun.cgs.value
    fmass_tick = []
    for z in fmass:
        if z!=0:
            fmass_tick.append("%.1f"%z)
        else:
            fmass_tick.append("%d"%z)
    return fmass_tick
                    
    
def adjust_subplots_band(ax, ax_):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', axis='both', which = 'both', top=True, right=True)
    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2)
    
    ax_.xaxis.set_minor_locator(AutoMinorLocator())
    ax_.yaxis.set_minor_locator(AutoMinorLocator())
    ax_.tick_params(direction='in', axis='both', which = 'both', top=True, right=True)
    ax_.tick_params(which='major', length=4)
    ax_.tick_params(which='minor', length=2)
    
    
def test_compute_spectra():
    # READ spectrum
    spectral_data = {}
    fitsNames = glob.glob('../data/spectra/86.12431_46.41912/*.fits.gz')
    for fitsName in fitsNames:
        hdul = fits.open(fitsName)
        sp = hdul[0]
        lam = sp.data[2,:]
        flux = sp.data[0,:]
        obstime = Time(sp.header['DATE-OBS'], format = 'isot', scale = 'utc')
        obsjd = obstime.jd
        key = len(list(spectral_data.keys()))
        spectral_data[key] = {}
        spectral_data[key]["lambda"] = lam
        spectral_data[key]["flux"] = flux
        spectral_data[key]["obsjd"] = obsjd
    
    wmax = 6563 *(1. + 1500/3e+5)
    wmin = 6563 *(1. - 1500/3e+5)
    band = [wmin, wmax]
    period = 0.49 # in day
    correlate_spec(spectral_data, band = band, period = period, plot_figure = True)
    
    
    
    
    