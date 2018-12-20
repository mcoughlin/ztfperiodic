
import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

from cuvarbase.ce import ConditionalEntropyAsyncProcess

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    parser.add_option("-m","--matchFile",default="/home/michael.coughlin/ZTF/Matchfiles/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.h5")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

matchFile = opts.matchFile
outputDir = opts.outputDir

if not os.path.isfile(matchFile):
    print("%s missing..."%matchFile)
    exit(0)

period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

lightcurves = []
baseline=0

f = h5py.File(matchFile, 'r+')
for key in f.keys():
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

proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=20, mag_bins=10, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
samples_per_peak = 10
df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)
results = proc.batched_run_const_nfreq(lightcurves, batch_size=10, freqs = freqs, only_keep_best_freq=True,show_progress=True)
finalresults=(([x[0] for x in coordinates]),([x[1] for x in coordinates]),([x[0] for x in results]),([x[2] for x in results]))
np.concatenate(finalresults)
finalresults=np.transpose(finalresults)    

k=0
while k<len(lightcurves):
    out  = results[k]
    period = 1./out[0]
    significance=out[2]
    if significance>6:
        phases=[]
        for i in lightcurves[k][0]:
            y=float(i)
            phases.append(np.remainder(y,2*period)/2*period)

            magnitude=lightcurves[k][1]
            err=lightcurves[k][2]
            RA=coordinates[k][0]
            Dec=coordinates[k][1]

            fig = plt.figure(figsize=(10,10))
            plt.gca().invert_yaxis()
            ax=fig.add_subplot(1,1,1)
            ax.errorbar(phases, magnitude,err,ls='none',c='k')
            period2=period
            ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))

            figfile = "%.10f_%.10f_%.10f_%s.png"%(significance, RA, Dec, 
                                                  period, fil)
            pngfile = os.path.join(basefolder,figfile)
            fig.savefig(pngfile)
            plt.close()
