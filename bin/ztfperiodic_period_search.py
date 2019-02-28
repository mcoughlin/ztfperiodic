#!/usr/bin/env python

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
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import get_kowalski_bulk
from ztfperiodic.utils import get_kowalski_list
from ztfperiodic.utils import get_matchfile

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("--doGPU",  action="store_true", default=False)
    parser.add_option("--doCPU",  action="store_true", default=False)
    parser.add_option("--doSaveMemory",  action="store_true", default=False)
    parser.add_option("--doRemoveTerrestrial",  action="store_true", default=False)
    parser.add_option("--doLightcurveStats",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    #parser.add_option("-m","--matchFile",default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.pytable") 
    parser.add_option("-m","--matchFile",default="/media/Data/mcoughlin/Matchfiles/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.h5")
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-a","--algorithm",default="CE")

    parser.add_option("-f","--field",default=251,type=int)
    parser.add_option("-c","--ccd",default=16,type=int)
    parser.add_option("-q","--quadrant",default=4,type=int)

    parser.add_option("-l","--lightcurve_source",default="matchfiles")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()

# Parse command line
opts = parse_commandline()

if not opts.lightcurve_source in ["matchfiles","h5files","Kowalski"]:
    print("--lightcurve_source must be either matchfiles, h5files, or Kowalski")
    exit(0)

if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

algorithm = opts.algorithm
matchFile = opts.matchFile
outputDir = opts.outputDir
batch_size = opts.batch_size
field = opts.field
ccd = opts.ccd
quadrant = opts.quadrant

if opts.doCPU and algorithm=="BLS":
    print("BLS only available for --doGPU")
    exit(0)

period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

catalogDir = os.path.join(outputDir,'catalog',algorithm)
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

lightcurves = []
coordinates = []
baseline=0

print('Organizing lightcurves...')
if opts.lightcurve_source == "Kowalski":

    catalogFile = os.path.join(catalogDir,"%d_%d_%d.dat"%(field, ccd, quadrant))

    kow = Kowalski(username=opts.user, password=opts.pwd)

    if opts.source_type == "quadrant":
        catalogFile = os.path.join(catalogDir,"%d_%d_%d.dat"%(field, ccd, quadrant))
        lightcurves, coordinates, baseline = get_kowalski_bulk(field, ccd, quadrant, kow)
    elif opts.source_type == "catalog":
        catalog_file = opts.catalog_file
        lines = [line.rstrip('\n') for line in open(catalog_file)]
        ras, decs = [], []
        for line in lines:
            lineSplit = line.split(" ")
            ras.append(float(lineSplit[1]))
            decs.append(float(lineSplit[2]))
        ras, decs = np.array(ras), np.array(decs)

        catalog_file_split = catalog_file.replace(".dat","").split("/")[-1]
        catalogFile = os.path.join(catalogDir,"%s.dat"%(catalog_file_split))
        lightcurves, coordinates, baseline = get_kowalski_list(ras, decs, kow)
    else:
        print("Source type unknown...")
        exit(0)

elif opts.lightcurve_source == "matchfiles":
    if not os.path.isfile(matchFile):
        print("%s missing..."%matchFile)
        exit(0)

    matchFileEnd = matchFile.split("/")[-1].replace("pytable","dat")
    catalogFile = os.path.join(catalogDir,matchFileEnd)

    lightcurves, coordinates, baseline = get_matchfile(matchFile)

    if len(lightcurves) == 0:
        print("No data available...")
        exit(0)

elif opts.lightcurve_source == "h5files":
    if not os.path.isfile(matchFile):
        print("%s missing..."%matchFile)
        exit(0)

    matchFileEnd = matchFile.split("/")[-1].replace("h5","dat")
    catalogFile = os.path.join(catalogDir,matchFileEnd)
    matchFileEndSplit = matchFileEnd.split("_")
    fil = matchFileEndSplit[2][1]

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
    f.close()

if len(lightcurves) == 0:
    touch(catalogFile)
    print('No lightcurves available for field %d, ccd %d, quadrant %d... exiting.'%(field,ccd,quadrant))
    exit(0)

if baseline<10:
    basefolder = os.path.join(outputDir,'%sHC'%algorithm)
    fmin, fmax = 18, 1440
else:
    basefolder = os.path.join(outputDir,'%s'%algorithm)
    fmin, fmax = 2/baseline, 480

samples_per_peak = 10
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

if opts.doRemoveTerrestrial:
    idx = np.where((freqs < 0.99) | (freqs > 1.01))[0]
    freqs = freqs[idx]

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

        lightcurves = lightcurves[:10]

        if opts.doSaveMemory:
            periods_best, significances = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=True)
        else:
            results = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=False)
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

        if opts.doSaveMemory:
            periods_best, significances = ls_proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, use_fft=True, samples_per_peak=spp, returnBestFreq=True)
        else:
            results = ls_proc.batched_run_const_nfreq(lightcurves,
                                                      batch_size=batch_size,
                                                      use_fft=True,
                                                      samples_per_peak=spp,
                                                      returnBestFreq=False)
            for data, out in zip(lightcurves,results):
                freqs, powers = out
                copy = np.ma.copy(data).T
                fap = fap_baluev(copy[:,0], copy[:,1], powers, np.max(freqs))
                idx = np.argmin(fap)

                period = 1./freqs[idx]
                significance = 1./fap[idx]

                periods_best.append(period)
                significances.append(significance)

        ls_proc.finish()

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

    if opts.algorithm == "LS":
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

    elif opts.algorithm == "CE":
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

if opts.doLightcurveStats:
    print('Running lightcurve stats...')
    stats = []
    for ii,data in enumerate(lightcurves):
        period = periods_best[ii]
        if np.mod(ii,10) == 0:
            print("%d/%d"%(ii,len(lightcurves)))
        copy = np.ma.copy(data).T
        t, mag, magerr = copy[:,0], copy[:,1], copy[:,2]

        stat = calc_stats(t, mag, magerr, period)
        stats.append(stat)

print('Cataloging / Plotting lightcurves...')
cnt = 0
fid = open(catalogFile,'w')
for lightcurve, coordinate, period, significance in zip(lightcurves,coordinates,periods_best,significances):
    if opts.doLightcurveStats:
        fid.write('%.10f %.10f %.10f %.10f '%(coordinate[0], coordinate[1], period, significance))
        fid.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n"%(stats[cnt][0], stats[cnt][1], stats[cnt][2], stats[cnt][3], stats[cnt][4], stats[cnt][5], stats[cnt][6], stats[cnt][7], stats[cnt][8], stats[cnt][9], stats[cnt][10], stats[cnt][11], stats[cnt][12], stats[cnt][13], stats[cnt][14], stats[cnt][15], stats[cnt][16], stats[cnt][17], stats[cnt][18], stats[cnt][19], stats[cnt][20], stats[cnt][21], stats[cnt][22], stats[cnt][23], stats[cnt][24], stats[cnt][25], stats[cnt][26], stats[cnt][27], stats[cnt][28], stats[cnt][29], stats[cnt][30], stats[cnt][31], stats[cnt][32], stats[cnt][33], stats[cnt][34], stats[cnt][35]))
    else:
        fid.write('%.10f %.10f %.10f %.10f\n'%(coordinate[0], coordinate[1], period, significance))

    if opts.doPlots and (significance>6):
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

    cnt = cnt + 1
fid.close()
