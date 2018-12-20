#!/usr/bin/python

import os, sys
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--doGPU",  action="store_true", default=False)
    parser.add_option("--doCPU",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    parser.add_option("-m","--matchFile",default="/home/michael.coughlin/ZTF/Matchfiles/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.h5")

    opts, args = parser.parse_args()

    return opts

def rephase(data, period=1.0, shift=0.0, col=0, copy=True):
    """
    Returns *data* (or a copy) phased with *period*, and shifted by a
    phase-shift *shift*.

    **Parameters**

    data : array-like, shape = [n_samples, n_cols]
        Array containing the time or phase values to be rephased in column
        *col*.
    period : number, optional
        Period to phase *data* by (default 1.0).
    shift : number, optional
        Phase shift to apply to phases (default 0.0).
    col : int, optional
        Column in *data* containing the time or phase values to be rephased
        (default 0).
    copy : bool, optional
        If True, a new array is returned, otherwise *data* is rephased
        in-place (default True).

    **Returns**

    rephased : array-like, shape = [n_samples, n_cols]
        Array containing the rephased *data*.
    """
    rephased = np.ma.array(data, copy=copy)
    rephased[:, col] = get_phase(rephased[:, col], period, shift)

    return rephased



def get_phase(time, period=1.0, shift=0.0):
    """
    Returns *time* transformed to phase-space with *period*, after applying a
    phase-shift *shift*.

    **Parameters**

    time : array-like, shape = [n_samples]
        The times to transform.
    period : number, optional
        The period to phase by (default 1.0).
    shift : number, optional
        The phase-shift to apply to the phases (default 0.0).

    **Returns**

    phase : array-like, shape = [n_samples]
        *time* transformed into phase-space with *period*, after applying a
        phase-shift *shift*.
    """
    return (time / period - shift) % 1

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

    r = rephase(data, period)
    bins, xedges, yedges = np.histogram2d(r[:,0], r[:,1], [xbins, ybins], [[0,1], [0,1]])
    size = r.shape[0]

# The following code was once more readable, but much slower.
# Here is what it used to be:
# -----------------------------------------------------------------------
#    return np.sum((lambda p: p * np.log(np.sum(bins[i,:]) / size / p) \
#                             if p > 0 else 0)(bins[i][j] / size)
#                  for i in np.arange(0, xbins)
#                  for j in np.arange(0, ybins)) if size > 0 else np.PINF
# -----------------------------------------------------------------------
# TODO: replace this comment with something that's not old code
    if size > 0:
        # bins[i,j] / size
        divided_bins = bins / size
        # indices where that is positive
        # to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1) #changed 0 by 1
        # array is repeated row-wise, so that it can be sliced by arg_positive
        column_sums = np.repeat(np.reshape(column_sums, (xbins,1)), ybins, axis=1)
        #column_sums = np.repeat(np.reshape(column_sums, (1,-1)), xbins, axis=0)


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

# Parse command line
opts = parse_commandline()

if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

matchFile = opts.matchFile
outputDir = opts.outputDir

if not os.path.isfile(matchFile):
    print("%s missing..."%matchFile)
    exit(0)

period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

catalogDir = os.path.join(outputDir,'catalog')
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

matchFileEnd = matchFile.split("/")[-1].replace("h5","dat")
catalogFile = os.path.join(catalogDir,matchFileEnd)

lightcurves = []
coordinates = []
baseline=0

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
    basefolder = os.path.join(outputDir,'CEHC')
    fmin, fmax = 18, 1440
else:
    basefolder = os.path.join(outputDir,'CE')
    fmin, fmax = 2/baseline, 480
f.close()

samples_per_peak = 10
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

significances, periods_best = [], []

if opts.doGPU:
    from cuvarbase.ce import ConditionalEntropyAsyncProcess

    proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=phase_bins, mag_bins=mag_bins, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
    results = proc.batched_run_const_nfreq(lightcurves, batch_size=10, freqs = freqs, only_keep_best_freq=True,show_progress=True)
    for out in results:
        period = 1./out[0]
        significance=out[2]
        periods_best.append(period)
        significances.append(significance)  

elif opts.doCPU:

    periods = 1/freqs
    period_jobs=1

    for ii,data in enumerate(lightcurves):
        if np.mod(ii,10) == 0:
            print("%d/%d"%(ii,len(lightcurves)))

        copy = np.ma.copy(data)
        copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
           / (np.max(copy[:,1]) - np.min(copy[:,1]))
        entropies = []
        for period in periods:
            entropy = CE(period, data=copy, xbins=phase_bins, ybins=mag_bins)
            entropies.append(entropy)

        period = periods[np.argmin(entropies)]
        significance = np.min(entropies)

        periods_best.append(period)
        significances.append(significance)

fid = open(catalogFile,'w')

for lightcurve, coordinate, period, significance in zip(lightcurves,coordinates,periods_best,significances):
    fid.write('%.10f %.10f %.10f %.10f\n'%(coordinate[0],coordinate[1],period, significance))
    if significance>6:
        phases = np.mod(lightcurve[:,0],2*period)/(2*period)
        magnitude, err = lightcurve[:,1], lightcurve[:,2]
        RA, Dec = coordinate

        fig = plt.figure(figsize=(10,10))
        plt.gca().invert_yaxis()
        ax=fig.add_subplot(1,1,1)
        ax.errorbar(phases, magnitude,err,ls='none',c='k')
        period2=period
        ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))

        figfile = "%.10f_%.10f_%.10f_%s.png"%(significance, RA, Dec, 
                                                  period, fil)
        idx = np.where((period>=period_ranges[0:]) & (period<=period_ranges[:-1]))[0]
        folder = os.path.join(basefolder,folders[idx.astype(int)])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pngfile = os.path.join(folder,figfile)
        fig.savefig(pngfile)
        plt.close()

fid.close()
