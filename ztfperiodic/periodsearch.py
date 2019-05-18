
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_periods(algorithm, lightcurves, freqs, batch_size=1,
                 doGPU=False, doCPU=False, doSaveMemory=False,
                 doRemoveTerrestrial=False,
                 doRemoveWindow=False,
                 freqs_to_remove=None,
                 phase_bins=20, mag_bins=10):

    if doRemoveTerrestrial and (freqs_to_remove is not None) and not (algorithm=="LS"):
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
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
    
    return np.array(periods_best), np.array(significances)
