#!/usr/bin/python

import os, sys
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ztfperiodic
from ztfperiodic.period import CE

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--doGPU",  action="store_true", default=False)
    parser.add_option("--doCPU",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    parser.add_option("-m","--matchFile",default="/home/michael.coughlin/ZTF/Matchfiles/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.h5")
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-a","--algorithm",default="CE")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

algorithm = opts.algorithm
matchFile = opts.matchFile
outputDir = opts.outputDir
batch_size = opts.batch_size

if opts.doCPU and algorithm=="BLS":
    print("BLS only available for --doGPU")
    exit(0)

if not os.path.isfile(matchFile):
    print("%s missing..."%matchFile)
    exit(0)

period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

catalogDir = os.path.join(outputDir,'catalog',algorithm)
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

matchFileEnd = matchFile.split("/")[-1].replace("h5","dat")
catalogFile = os.path.join(catalogDir,matchFileEnd)
matchFileEndSplit = matchFileEnd.split("_")
fil = matchFileEndSplit[2][1]

lightcurves = []
coordinates = []
baseline=0

print('Organizing lightcurves...')
f = h5py.File(matchFile, 'r+')
for key in f.keys():
    keySplit = key.split("_")
    nid, ra, dec = int(keySplit[0]), float(keySplit[1]), float(keySplit[2])
    coordinates.append((ra,dec))

    data = list(f[key])
    data = np.array(data).T
    if len(data[:,0]) < 50: continue
    lightcurve=(data[:,0],data[:,1],data[:,2])
    lightcurves.append(lightcurve)

    newbaseline = max(data[:,0])-min(data[:,0])
    if newbaseline>baseline:
        baseline=newbaseline

if baseline<10:
    basefolder = os.path.join(outputDir,'%sHC'%algorithm)
    fmin, fmax = 18, 1440
else:
    basefolder = os.path.join(outputDir,'%s'%algorithm)
    fmin, fmax = 2/baseline, 480
f.close()

samples_per_peak = 10
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

if opts.doGPU and (algorithm == "PDM"):
    from cuvarbase.utils import weights
    lightcurves_pdm = []
    for lightcurve in lightcurves:
        t, y, dy = lightcurve
        lightcurves_pdm.append((t, y, weights(np.ones(dy.shape)), freqs))
    lightcurves = lightcurves_pdm 

significances, periods_best = [], []

print('Period finding lightcurves...')
if opts.doGPU:
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    from cuvarbase.bls import eebls_gpu_fast
    from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
    from cuvarbase.pdm import PDMAsyncProcess

    if algorithm == "CE":
        proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=phase_bins, mag_bins=mag_bins, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
        results = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True)
        for out in results:
            periods = 1./out[0]
            entropies = out[1]
            significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
            period = periods[np.argmin(entropies)]

            periods_best.append(period)
            significances.append(significance)  

    elif algorithm == "BLS":
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
        nfft_sigma, spp = 5, 3

        ls_proc = LombScargleAsyncProcess(use_double=True,
                                              sigma=nfft_sigma)

        results = ls_proc.batched_run_const_nfreq(lightcurves, 
                                                  batch_size=batch_size,
                                                  use_fft=True,
                                                  samples_per_peak=spp)  
        ls_proc.finish()

        for data, out in zip(lightcurves,results):
            freqs, powers = out
            copy = np.ma.copy(data).T
            fap = fap_baluev(copy[:,0], copy[:,1], powers, np.max(freqs))
            idx = np.argmin(fap)

            period = 1./freqs[idx]
            significance = 1./fap[idx]

            periods_best.append(period)
            significances.append(significance)

    elif algorithm == "PDM":
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

elif opts.doCPU:

    periods = 1/freqs
    period_jobs=1

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

print('Plotting lightcurves...')
fid = open(catalogFile,'w')
for lightcurve, coordinate, period, significance in zip(lightcurves,coordinates,periods_best,significances):
    fid.write('%.10f %.10f %.10f %.10f\n'%(coordinate[0],coordinate[1],period, significance))
    if significance>6:
        if opts.doGPU and (algorithm == "PDM"):
            copy = np.ma.copy((lightcurve[0],lightcurve[1],lightcurve[2])).T
        else:
            copy = np.ma.copy(lightcurve).T
        phases = np.mod(copy[:,0],2*period)/(2*period)
        magnitude, err = copy[:,1], copy[:,2]
        RA, Dec = coordinate

        fig = plt.figure(figsize=(10,10))
        plt.gca().invert_yaxis()
        ax=fig.add_subplot(1,1,1)
        ax.errorbar(phases, magnitude,err,ls='none',c='k')
        period2=period
        ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))

        figfile = "%.10f_%.10f_%.10f_%.10f_%s.png"%(significance, RA, Dec, 
                                                  period, fil)
        idx = np.where((period>=period_ranges[:-1]) & (period<=period_ranges[1:]))[0][0]
        if folders[idx.astype(int)] == None:
            continue
        folder = os.path.join(basefolder,folders[idx.astype(int)])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pngfile = os.path.join(folder,figfile)
        fig.savefig(pngfile)
        plt.close()

fid.close()
