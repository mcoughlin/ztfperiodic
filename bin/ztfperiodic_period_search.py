#!/usr/bin/env python

import os, sys
import time
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

from astropy import units as u
from astropy.coordinates import SkyCoord

import ztfperiodic
from ztfperiodic.period import CE
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import get_kowalski_bulk
from ztfperiodic.utils import get_kowalski_list
from ztfperiodic.utils import get_matchfile
from ztfperiodic.periodsearch import find_periods

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
    parser.add_option("--doRemoveBrightStars",  action="store_true", default=False)
    parser.add_option("--doRemoveHC",  action="store_true", default=False)
    parser.add_option("--doLongPeriod",  action="store_true", default=False)
    parser.add_option("--doCombineFilt",  action="store_true", default=False)

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=4,type=int)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    #parser.add_option("-m","--matchFile",default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.pytable") 
    parser.add_option("-m","--matchFile",default="/media/Data/mcoughlin/Matchfiles/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.h5")
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-k","--kowalski_batch_size",default=1000,type=int)
    parser.add_option("-a","--algorithm",default="CE")

    parser.add_option("-f","--field",default=251,type=int)
    parser.add_option("-c","--ccd",default=16,type=int)
    parser.add_option("-q","--quadrant",default=4,type=int)

    parser.add_option("-l","--lightcurve_source",default="matchfiles")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../catalogs/swift.dat")
    parser.add_option("--Ncatalog",default=1,type=int)
    parser.add_option("--Ncatindex",default=0,type=int)

    parser.add_option("--stardist",default=100.0,type=float)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("-p","--program_ids",default="2,3")
    parser.add_option("--min_epochs",default=50,type=int)

    opts, args = parser.parse_args()

    return opts

def brightstardist(filename,ra,dec):
     catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
     with h5py.File(filename, 'r') as f:
         ras, decs = f['ra'][:], f['dec'][:]
     c = SkyCoord(ra=ras*u.degree, dec=decs*u.degree,frame='icrs')
     idx,sep,_ = catalog.match_to_catalog_sky(c)
     return sep.arcsec

def slicestardist(lightcurves, coordinates):

    ras, decs = [], []
    for coordinate in coordinates:
        ras.append(coordinate[0])
        decs.append(coordinate[1])
    ras, decs = np.array(decs), np.array(decs)

    filename = "%s/bsc5.hdf5" % starCatalogDir
    sep = brightstardist(filename,ras,decs)
    idx1 = np.where(sep >= opts.stardist)[0]
    filename = "%s/Gaia.hdf5" % starCatalogDir
    sep = brightstardist(filename,ras,decs)
    idx2 = np.where(sep >= opts.stardist)[0]
    idx = np.union1d(idx1,idx2).astype(int)

    return [lightcurves[i] for i in idx], [coordinates[i] for i in idx]

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
program_ids = list(map(int,opts.program_ids.split(",")))
min_epochs = opts.min_epochs
catalog_file = opts.catalog_file
doCombineFilt = opts.doCombineFilt
doRemoveHC = opts.doRemoveHC

scriptpath = os.path.realpath(__file__)
starCatalogDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"catalogs")

if opts.doCPU and algorithm=="BLS":
    print("BLS only available for --doGPU")
    exit(0)

if (opts.source_type == "catalog") and ("blue" in catalog_file):
    period_ranges = [0,0.0020833333333333333,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
    folders = [None,"3min","4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]
else:
    period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
    folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

epoch_ranges = [0,100,500,np.inf]
epoch_folders = ["0-100","100-500","500-all"]

catalogDir = os.path.join(outputDir,'catalog',algorithm)
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

lightcurves = []
coordinates = []
baseline=0
fil = 'all'

print('Organizing lightcurves...')
if opts.lightcurve_source == "Kowalski":

    catalogFile = os.path.join(catalogDir,"%d_%d_%d.dat"%(field, ccd, quadrant))

    kow = Kowalski(username=opts.user, password=opts.pwd)

    if opts.source_type == "quadrant":
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.dat"%(field, ccd, quadrant,opts.Ncatindex))
        lightcurves, coordinates, filters, baseline = get_kowalski_bulk(field, ccd, quadrant, kow, program_ids=program_ids, min_epochs=min_epochs, num_batches=opts.Ncatalog, nb=opts.Ncatindex)
        if opts.doRemoveBrightStars:
            lightcurves, coordinates = slicestardist(lightcurves, coordinates)

    elif opts.source_type == "catalog":

        amaj, amin, phi = None, None, None
        if doCombineFilt:
            default_err = 3.0
        else:
            default_err = 5.0

        if ".dat" in catalog_file:
            lines = [line.rstrip('\n') for line in open(catalog_file)]
            ras, decs, errs = [], [], []
            if ("fermi" in catalog_file):
                amaj, amin, phi = [], [], []
            for line in lines:
                lineSplit = list(filter(None,line.split(" ")))
                if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file):
                    ras.append(float(lineSplit[0]))
                    decs.append(float(lineSplit[1]))
                    errs.append(default_err)
                elif ("vlss" in catalog_file):
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                elif ("fermi" in catalog_file):
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                    amaj.append(float(lineSplit[5]))
                    amin.append(float(lineSplit[6]))
                    phi.append(float(lineSplit[7]))
                elif ("swift" in catalog_file) or ("xmm" in catalog_file):
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = float(lineSplit[3])
                    errs.append(err)
                else:
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    errs.append(default_err)
            ras, decs, errs = np.array(ras), np.array(decs), np.array(errs)
            if ("fermi" in catalog_file):
                amaj, amin, phi = np.array(amaj), np.array(amin), np.array(phi)
        elif ".hdf5" in catalog_file:
            with h5py.File(catalog_file, 'r') as f:
                ras, decs = f['ra'][:], f['dec'][:]
            errs = default_err*np.ones(ras.shape)

        if opts.doRemoveBrightStars:
            filename = "%s/bsc5.hdf5" % starCatalogDir
            sep = brightstardist(filename,ras,decs)
            idx1 = np.where(sep >= opts.stardist)[0]
            filename = "%s/Gaia.hdf5" % starCatalogDir
            sep = brightstardist(filename,ras,decs)
            idx2 = np.where(sep >= opts.stardist)[0]
            idx = np.union1d(idx1,idx2)
            ras, decs, errs = ras[idx], decs[idx], errs[idx]
            if ("fermi" in catalog_file):
                amaj, amin, phi = amaj[idx], amin[idx], phi[idx]

        ras_split = np.array_split(ras,opts.Ncatalog)
        decs_split = np.array_split(decs,opts.Ncatalog)
        errs_split = np.array_split(errs,opts.Ncatalog)

        ras = ras_split[opts.Ncatindex]
        decs = decs_split[opts.Ncatindex]
        errs = errs_split[opts.Ncatindex]

        if ("fermi" in catalog_file):
            amaj_split = np.array_split(amaj,opts.Ncatalog)
            amin_split = np.array_split(amin,opts.Ncatalog)
            phi_split = np.array_split(phi,opts.Ncatalog)

            amaj = amaj_split[opts.Ncatindex]
            amin = amin_split[opts.Ncatindex]
            phi = phi_split[opts.Ncatindex]

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").split("/")[-1]
        catalogFile = os.path.join(catalogDir,"%s_%d.dat"%(catalog_file_split,
                                                           opts.Ncatindex))
        lightcurves, coordinates, filters, baseline = get_kowalski_list(ras, decs,
                                                 kow,
                                                 program_ids=program_ids,
                                                 min_epochs=min_epochs,
                                                 errs=errs,
                                                 amaj=amaj, amin=amin, phi=phi,
                                                 doCombineFilt=doCombineFilt,
                                                 doRemoveHC=doRemoveHC)
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
    if opts.doRemoveBrightStars:
        lightcurves, coordinates = slicestardist(lightcurves, coordinates)

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
        if len(data[:,0]) < min_epochs: continue
        lightcurve=(data[:,0],data[:,1],data[:,2])
        lightcurves.append(lightcurve)

        newbaseline = max(data[:,0])-min(data[:,0])
        if newbaseline>baseline:
            baseline=newbaseline
    f.close()

    if opts.doRemoveBrightStars:
        lightcurves, coordinates = slicestardist(lightcurves, coordinates)

if len(lightcurves) == 0:
    touch(catalogFile)
    print('No lightcurves available... exiting.')
    exit(0)

if baseline<10:
    basefolder = os.path.join(outputDir,'%sHC'%algorithm)
    if opts.doLongPeriod:
        fmin, fmax = 18, 48
    else:
        fmin, fmax = 18, 1440
else:
    basefolder = os.path.join(outputDir,'%s'%algorithm)
    if opts.doLongPeriod:
        fmin, fmax = 2/baseline, 48
    else:
        fmin, fmax = 2/baseline, 480

samples_per_peak = 10
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

if opts.doRemoveTerrestrial:
    idx = np.where((freqs < 0.95) | (freqs > 1.05))[0]
    freqs = freqs[idx]
    idx = np.where((freqs < 0.48) | (freqs > 0.52))[0]
    freqs = freqs[idx]

if opts.doGPU and (algorithm == "PDM"):
    from cuvarbase.utils import weights
    lightcurves_pdm = []
    for lightcurve in lightcurves:
        t, y, dy = lightcurve
        lightcurves_pdm.append((t, y, weights(np.ones(dy.shape)), freqs))
    lightcurves = lightcurves_pdm 

print('Analyzing %d lightcurves...' % len(lightcurves))
start_time = time.time()
periods_best, significances = find_periods(algorithm, lightcurves, freqs, doGPU=opts.doGPU, doCPU=opts.doCPU)
end_time = time.time()
print('Lightcurve analysis took %.2f seconds' % (end_time - start_time))

if opts.doLightcurveStats:
    print('Running lightcurve stats...')

    if opts.doParallel:
        from joblib import Parallel, delayed
        stats = Parallel(n_jobs=opts.Ncore)(delayed(calc_stats)(LC[0],LC[1],LC[2],p) for LC,p in zip(lightcurves,periods_best))
    else:
        stats = []
        for ii,data in enumerate(lightcurves):
            period = periods_best[ii]
            if np.mod(ii,10) == 0:
                print("%d/%d"%(ii,len(lightcurves)))
            copy = np.ma.copy(data).T
            t, mag, magerr = copy[:,0], copy[:,1], copy[:,2]

            stat = calc_stats(t, mag, magerr, period)
            stats.append(stat)

if algorithm == "LS":
    sigthresh = 100
else:
    sigthresh = 7

print('Cataloging / Plotting lightcurves...')
cnt = 0
fid = open(catalogFile,'w')
for lightcurve, filt, coordinate, period, significance in zip(lightcurves,filters,coordinates,periods_best,significances):
    if opts.doLightcurveStats:
        fid.write('%.10f %.10f %.10f %.10f '%(coordinate[0], coordinate[1], period, significance))
        fid.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n"%(stats[cnt][0], stats[cnt][1], stats[cnt][2], stats[cnt][3], stats[cnt][4], stats[cnt][5], stats[cnt][6], stats[cnt][7], stats[cnt][8], stats[cnt][9], stats[cnt][10], stats[cnt][11], stats[cnt][12], stats[cnt][13], stats[cnt][14], stats[cnt][15], stats[cnt][16], stats[cnt][17], stats[cnt][18], stats[cnt][19], stats[cnt][20], stats[cnt][21], stats[cnt][22], stats[cnt][23], stats[cnt][24], stats[cnt][25], stats[cnt][26], stats[cnt][27], stats[cnt][28], stats[cnt][29], stats[cnt][30], stats[cnt][31], stats[cnt][32], stats[cnt][33], stats[cnt][34], stats[cnt][35]))
    else:
        fid.write('%.10f %.10f %.10f %.10f\n'%(coordinate[0], coordinate[1], period, significance))

    if opts.doPlots and (significance>sigthresh):
        if opts.doGPU and (algorithm == "PDM"):
            copy = np.ma.copy((lightcurve[0],lightcurve[1],lightcurve[2])).T
        else:
            copy = np.ma.copy(lightcurve).T
        phases = np.mod(copy[:,0],2*period)/(2*period)
        magnitude, err = copy[:,1], copy[:,2]
        RA, Dec = coordinate

        fig = plt.figure(figsize=(10,10))
        ax=fig.add_subplot(1,1,1)
        ax.errorbar(phases, magnitude,err,ls='none',c='k')
        period2=period
        ymed = np.nanmedian(magnitude)
        y10, y90 = np.nanpercentile(magnitude,10), np.nanpercentile(magnitude,90)
        ystd = np.nanmedian(err)
        ymin = y10 - 3*ystd
        ymax = y90 + 3*ystd
        plt.ylim([ymin,ymax])
        plt.gca().invert_yaxis()
        ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))

        filt_str = [str(x) for x in filt]
        figfile = "%.10f_%.10f_%.10f_%.10f_%s.png"%(significance, RA, Dec, 
                                                  period, "".join(filt_str))
        idx = np.where((period>=period_ranges[:-1]) & (period<=period_ranges[1:]))[0][0]
        if folders[idx.astype(int)] == None:
            continue

        nepoch = np.array(len(copy[:,0]))
        idx2 = np.where((nepoch>=epoch_ranges[:-1]) & (nepoch<=epoch_ranges[1:]))[0][0]
        if epoch_folders[idx2.astype(int)] == None:
            continue

        folder = os.path.join(basefolder,folders[idx.astype(int)],epoch_folders[idx2.astype(int)])
        if not os.path.isdir(folder):
            os.makedirs(folder)
        pngfile = os.path.join(folder,figfile)
        fig.savefig(pngfile)
        plt.close()

    cnt = cnt + 1
fid.close()
