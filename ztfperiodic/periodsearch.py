
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ztfperiodic.utils

def find_periods(algorithm, lightcurves, freqs, batch_size=1,
                 doGPU=False, doCPU=False, doSaveMemory=False,
                 doRemoveTerrestrial=False,
                 doRemoveWindow=False,
                 doUsePDot=False,
                 freqs_to_remove=None,
                 phase_bins=20, mag_bins=10):

    if doRemoveTerrestrial and (freqs_to_remove is not None) and not (algorithm=="LS"):
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')
    if doGPU:
    
        if algorithm == "CE":
            from cuvarbase.ce import ConditionalEntropyAsyncProcess
    
            proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=phase_bins, mag_bins=mag_bins, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
    
            if doSaveMemory:
                periods_best, significances = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=True)
            else:

                results = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=False)
                cnt = 0
                for lightcurve, out in zip(lightcurves,results):
                    periods = 1./out[0]
                    entropies = out[1]

                    significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
                    period = periods[np.argmin(entropies)]
    
                    periods_best.append(period)
                    significances.append(significance)
   
        elif algorithm == "BLS":
            from cuvarbase.bls import eebls_gpu_fast
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
                copy = np.ma.copy(data).T
                powers = eebls_gpu_fast(copy[:,0],copy[:,1], copy[:,2],
                                        freq_batch_size=batch_size,
                                        freqs = freqs)
    
                significance = np.abs(np.mean(powers)-np.max(powers))/np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq
    
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "LS":
            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
    
            nfft_sigma, spp = 10, 10
    
            ls_proc = LombScargleAsyncProcess(use_double=True,
                                                  sigma=nfft_sigma)
    
            if doSaveMemory:
                periods_best, significances = ls_proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, use_fft=True, samples_per_peak=spp, returnBestFreq=True, freqs = freqs, doRemoveTerrestrial=doRemoveTerrestrial, freqs_to_remove=freqs_to_remove)
            else:
                results = ls_proc.batched_run_const_nfreq(lightcurves,
                                                          batch_size=batch_size,
                                                          use_fft=True,
                                                          samples_per_peak=spp,
                                                          returnBestFreq=False,
                                                          freqs = freqs)
    
                for data, out in zip(lightcurves,results):
                    freqs, powers = out
                    if doRemoveTerrestrial and (freqs_to_remove is not None):
                        for pair in freqs_to_remove:
                            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                            freqs = freqs[idx]
                            powers = powers[idx]

                    copy = np.ma.copy(data).T
                    fap = fap_baluev(copy[:,0], copy[:,2], powers, np.max(freqs))
                    idx = np.argmin(fap)
    
                    period = 1./freqs[idx]
                    significance = 1./fap[idx]

                    periods_best.append(period)
                    significances.append(significance)

            ls_proc.finish()
    
        elif algorithm == "PDM":
            from cuvarbase.pdm import PDMAsyncProcess
    
            kind, nbins = 'binned_linterp', 10
    
            pdm_proc = PDMAsyncProcess()
            for lightcurve in lightcurves:
                results = pdm_proc.run([lightcurve], kind=kind, nbins=nbins)
                pdm_proc.finish()
                powers = results[0]
    
                significance = np.abs(np.mean(powers)-np.max(powers))/np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq
    
                periods_best.append(period)
                significances.append(significance)


        elif algorithm == "GCE":
            from gcex.gce import ConditionalEntropy
        
            nphase = 50
            ce = ConditionalEntropy(phase_bins=nphase)

            if doUsePDot:
                num_pdots = 10
                max_pdot = 1e-10
                min_pdot = 1e-12
                pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
                pdots_to_test = np.append(0,pdots_to_test)
                #pdots_to_test = np.array([-2.365e-11])
            else:
                pdots_to_test = np.array([0.0])

            lightcurves_stack = [] 
            for lightcurve in lightcurves:
                idx = np.argsort(lightcurve[0])
                lightcurve = (lightcurve[0][idx],
                              lightcurve[1][idx],
                              lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1])).T
                lightcurves_stack.append(lightcurve_stack)

            periods_best = np.zeros((len(lightcurves),1))
            significances = np.zeros((len(lightcurves),1))
            pdots = np.zeros((len(lightcurves),1))

            pdots_split = np.array_split(pdots_to_test,len(pdots_to_test))
            for ii, pdot in enumerate(pdots_split):
                print("Running pdot %d / %d" % (ii+1, len(pdots_split)))

                print("Number of lightcurves: %d" % len(lightcurves_stack))
                print("Batch size: %d" % batch_size)
                print("Number of frequency bins: %d" % len(freqs))
                print("Number of phase bins: %d" % nphase)

                results = ce.batched_run_const_nfreq(lightcurves_stack, batch_size, freqs, pdot, show_progress=False)
                periods = 1./freqs
           
                for jj, (lightcurve, entropies2) in enumerate(zip(lightcurves,results)):
                    for kk, entropies in enumerate(entropies2):
                        significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
                        period = periods[np.argmin(entropies)]

                        if significance > significances[jj]:
                            periods_best[jj] = period
                            significances[jj] = significance
                            pdots[jj] = pdot[kk]*1.0 
            pdots, periods_best, significances = pdots.flatten(), periods_best.flatten(), significances.flatten()

        elif algorithm == "FFT":
            from cuvarbase.lombscargle import fap_baluev
            from reikna import cluda
            from reikna.fft.fft import FFT

            T = 30.0/86400.0
            fs = 1.0/T

            api = cluda.get_api('cuda')
            dev = api.get_platforms()[0].get_devices()[0]
            thr = api.Thread(dev)

            x = np.arange(0.0, 12.0/24.0, T).astype(np.complex128)
            fft  = FFT(x, axes=(0,))
            fftc = fft.compile(thr, fast_math=True)

            lightcurves_stack = []

            period_min, period_max = 60.0/86400.0, 12.0*3600.0/86400.0
            freq_min, freq_max = 1/period_max, 1/period_min

            for ii, lightcurve in enumerate(lightcurves):
                bins_tmp = np.arange(np.min(lightcurve[0]), np.max(lightcurve[1]), 1/24.0)
                bins = np.vstack((bins_tmp[:-12],bins_tmp[12:]))
                n, bins_tmp = ztfperiodic.utils.overlapping_histogram(lightcurve[0], bins)
                idx = np.argmax(n)
                bins_max = bins[:,idx]

                x = np.arange(bins_max[0], bins_max[0] + 12.0/24.0, T)
                y = np.interp(x, lightcurve[0], lightcurve[1]).astype(np.complex128)
                yerr = np.interp(x, lightcurve[0], lightcurve[2])
                if len(y) == 0:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue                    
                y = y - np.median(y)
                y = y * np.hanning(len(y))

                dev   = thr.to_device(y)
                fftc(dev, dev)

                Y = dev.get()
                powers = np.abs(Y)
                N = len(y)
                freqs = np.linspace(0.0, 1.0/(2.0*T), N/2)

                powers = powers[:int(N/2)]
                idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

                freqs, powers = freqs[idx], powers[idx]
                powers = powers * freqs**2

                significance = np.abs(np.median(powers)-np.max(powers))/np.std(powers)   
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq

                periods_best.append(period)
                significances.append(significance)

    elif doCPU:
    
        periods = 1/freqs
        period_jobs=1
    
        if algorithm == "LS":
            from astropy.stats import LombScargle
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
                copy = np.ma.copy(data).T
                nrows, ncols = copy.shape
    
                if nrows == 1:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue
    
                ls = LombScargle(copy[:,0], copy[:,1], copy[:,2])
                power = ls.power(freqs)
                fap = ls.false_alarm_probability(power,maximum_frequency=np.max(freqs))
    
                idx = np.argmin(fap)
                significance = 1./fap[idx]
                period = 1./freqs[idx]
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "CE":
            from ztfperiodic.period import CE
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
    
                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))
                entropies = []
                for period in periods:
                    entropy = CE(period, data=copy, xbins=phase_bins, ybins=mag_bins)
                    entropies.append(entropy)
                significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
                period = periods[np.argmin(entropies)]
    
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "AOV":
            from ztfperiodic.pyaov.pyaov import aovw
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
    
                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))
    
                aov, fr, _ = aovw(copy[:,0], copy[:,1], copy[:,2],
                                  fstop=np.max(1.0/periods),
                                  fstep=1/periods[0])
    
                significance = np.abs(np.mean(aov)-np.max(aov))/np.std(aov)
                period = periods[np.argmax(aov)]
    
                periods_best.append(period)
                significances.append(significance)
    
    return np.array(periods_best), np.array(significances), np.array(pdots)
