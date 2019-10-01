#!/usr/bin/env python

import os, sys
import tempfile
import time
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py
from scipy.signal import lombscargle

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
import astropy.io.fits

import ztfperiodic
from ztfperiodic.period import CE
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import get_kowalski_bulk
from ztfperiodic.utils import get_kowalski_list
from ztfperiodic.utils import get_simulated_list
from ztfperiodic.utils import get_matchfile
from ztfperiodic.utils import convert_to_hex
from ztfperiodic.periodsearch import find_periods
from ztfperiodic.specfunc import correlate_spec

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
    parser.add_option("--doExtinction",  action="store_true", default=False)
    parser.add_option("--doSpectra",  action="store_true", default=False)

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=4,type=int)

    parser.add_option("--doSimulateLightcurves",  action="store_true", default=False)
    parser.add_option("--doUsePDot",  action="store_true", default=False)
    parser.add_option("--doVariability",  action="store_true", default=False)
    parser.add_option("--doQuadrantFile",  action="store_true", default=False)
    parser.add_option("--quadrant_file",default="../input/quadrant_file.dat")
    parser.add_option("--quadrant_index",default=0,type=int)

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


def slicestardist(lightcurves, coordinates, filters, ids, absmags, bp_rps, names):

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

    return [lightcurves[i] for i in idx], [coordinates[i] for i in idx], [filters[i] for i in idx], [ids[i] for i in idx], [absmags[i] for i in idx], [bp_rps[i] for i in idx], [names[i] for i in idx]


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
quadrant_file = opts.quadrant_file
doCombineFilt = opts.doCombineFilt
doRemoveHC = opts.doRemoveHC
doSimulateLightcurves = opts.doSimulateLightcurves
doUsePDot = opts.doUsePDot
doExtinction = opts.doExtinction
Ncatindex = opts.Ncatindex

if opts.doQuadrantFile:
    quad_out = np.loadtxt(quadrant_file)
    idx = np.where(quad_out[:,0] == opts.quadrant_index)[0]
    row = quad_out[idx,:][0]
    field, ccd, quadrant = row[1], row[2], row[3]
    Ncatindex = row[4]

scriptpath = os.path.realpath(__file__)
starCatalogDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"catalogs")

WDcat = os.path.join(starCatalogDir,'GaiaHRSet.hdf5')
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)

if opts.doCPU and algorithm=="BLS":
    print("BLS only available for --doGPU")
    exit(0)

if (opts.source_type == "catalog") and (("blue" in catalog_file) or ("wdb" in catalog_file)):
    period_ranges = [0,0.0020833333333333333,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
    folders = [None,"3min","4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]
else:
    period_ranges = [0,0.002777778,0.0034722,0.0041666,0.004861111,0.006944444,0.020833333,0.041666667,0.083333333,0.166666667,0.5,3.0,10.0,50.0,np.inf]
    folders = [None,"4min","5min","6min","7_10min","10_30min","30_60min","1_2hours","2_4hours","4_12hours","12_72hours","3_10days","10_50days","50_baseline"]

epoch_ranges = [0,100,500,np.inf]
epoch_folders = ["0-100","100-500","500-all"]

catalogDir = os.path.join(outputDir,'catalog',algorithm)
if (opts.source_type == "catalog") and ("fermi" in catalog_file):
    catalogDir = os.path.join(catalogDir,'%d' % Ncatindex)

if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

lightcurves = []
coordinates = []
baseline=0
fil = 'all'

print('Organizing lightcurves...')
if opts.lightcurve_source == "Kowalski":

    catalogFile = os.path.join(catalogDir,"%d_%d_%d.dat"%(field, ccd, quadrant))

    kow = []
    nquery = 10
    cnt = 0
    while cnt < nquery:
        try:
            kow = Kowalski(username=opts.user, password=opts.pwd)
            break
        except:
            time.sleep(5)
        cnt = cnt + 1
    if cnt == nquery:
        raise Exception('Kowalski connection failed...')

    if opts.source_type == "quadrant":
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.dat"%(field, ccd, quadrant,Ncatindex))
        lightcurves, coordinates, filters, ids,\
        absmags, bp_rps, names, baseline =\
            get_kowalski_bulk(field, ccd, quadrant, kow, 
                              program_ids=program_ids, min_epochs=min_epochs,
                              num_batches=opts.Ncatalog, nb=Ncatindex)
        if opts.doRemoveBrightStars:
            lightcurves, coordinates, filters, ids, absmags, bp_rps, names =\
                slicestardist(lightcurves, coordinates, filters,
                              ids, absmags, bp_rps, names)

    elif opts.source_type == "catalog":

        amaj, amin, phi = None, None, None
        if doCombineFilt:
            default_err = 3.0
        else:
            default_err = 5.0

        if ".dat" in catalog_file:
            lines = [line.rstrip('\n') for line in open(catalog_file)]
            names, ras, decs, errs = [], [], [], []
            if ("fermi" in catalog_file):
                amaj, amin, phi = [], [], []
            for line in lines:
                lineSplit = list(filter(None,line.split(" ")))
                if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file):
                    ra_hex, dec_hex = convert_to_hex(float(lineSplit[0])*24/360.0,delimiter=''), convert_to_hex(float(lineSplit[1]),delimiter='')
                    if dec_hex[0] == "-":
                        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
                    else:
                        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
                    names.append(objname)
                    ras.append(float(lineSplit[0]))
                    decs.append(float(lineSplit[1]))
                    errs.append(default_err)
                elif ("vlss" in catalog_file):
                    names.append(lineSplit[0])
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                elif ("fermi" in catalog_file):
                    names.append(lineSplit[0])
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                    amaj.append(float(lineSplit[3]))
                    amin.append(float(lineSplit[4]))
                    phi.append(float(lineSplit[5]))
                elif ("swift" in catalog_file) or ("xmm" in catalog_file):
                    names.append(lineSplit[0])
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = float(lineSplit[3])
                    errs.append(err)
                else:
                    names.append(lineSplit[0])
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    errs.append(default_err)
            names = np.array(names)
            ras, decs, errs = np.array(ras), np.array(decs), np.array(errs)
            if ("fermi" in catalog_file):
                amaj, amin, phi = np.array(amaj), np.array(amin), np.array(phi)
        elif ".hdf5" in catalog_file:
            with h5py.File(catalog_file, 'r') as f:
                ras, decs = f['ra'][:], f['dec'][:]
            errs = default_err*np.ones(ras.shape)

            names = []
            for ra, dec in zip(ras, decs):
                ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
                if dec_hex[0] == "-":
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
                else:
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
                names.append(objname)
            names = np.array(names)

        if opts.doRemoveBrightStars:
            filename = "%s/bsc5.hdf5" % starCatalogDir
            sep = brightstardist(filename,ras,decs)
            idx1 = np.where(sep >= opts.stardist)[0]
            filename = "%s/Gaia.hdf5" % starCatalogDir
            sep = brightstardist(filename,ras,decs)
            idx2 = np.where(sep >= opts.stardist)[0]
            idx = np.union1d(idx1,idx2)
            names = names[idx]
            ras, decs, errs = ras[idx], decs[idx], errs[idx]
            if ("fermi" in catalog_file):
                amaj, amin, phi = amaj[idx], amin[idx], phi[idx]

        names_split = np.array_split(names,opts.Ncatalog)
        ras_split = np.array_split(ras,opts.Ncatalog)
        decs_split = np.array_split(decs,opts.Ncatalog)
        errs_split = np.array_split(errs,opts.Ncatalog)

        names = names_split[Ncatindex]
        ras = ras_split[Ncatindex]
        decs = decs_split[Ncatindex]
        errs = errs_split[Ncatindex]

        if ("fermi" in catalog_file):
            amaj_split = np.array_split(amaj,opts.Ncatalog)
            amin_split = np.array_split(amin,opts.Ncatalog)
            phi_split = np.array_split(phi,opts.Ncatalog)

            amaj = amaj_split[Ncatindex]
            amin = amin_split[Ncatindex]
            phi = phi_split[Ncatindex]

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").split("/")[-1]
        catalogFile = os.path.join(catalogDir,"%s_%d.dat"%(catalog_file_split,
                                                           Ncatindex))

        if doSimulateLightcurves:
            lightcurves, coordinates, filters, ids,\
            absmags, bp_rps, names, baseline =\
                get_simulated_list(ras, decs,
                                  min_epochs=min_epochs,
                                  names=names,
                                  doCombineFilt=doCombineFilt,
                                  doRemoveHC=doRemoveHC,
                                  doUsePDot=doUsePDot)
        else:
            lightcurves, coordinates, filters, ids,\
            absmags, bp_rps, names, baseline =\
                get_kowalski_list(ras, decs,
                                  kow,
                                  program_ids=program_ids,
                                  min_epochs=min_epochs,
                                  errs=errs,
                                  names=names,
                                  amaj=amaj, amin=amin, phi=phi,
                                  doCombineFilt=doCombineFilt,
                                  doRemoveHC=doRemoveHC,
                                  doExtinction=doExtinction)
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
        lightcurves, coordinates, filters, ids =\
            slicestardist(lightcurves, coordinates, filters, ids)

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

if (opts.source_type == "catalog") and ("fermi" in catalog_file):
    basefolder = os.path.join(basefolder,'%d' % Ncatindex)

samples_per_peak = 3
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))

freqs = fmin + df * np.arange(nf)

if opts.doRemoveTerrestrial:
    freqs_to_remove = [[3e-2,4e-2], [47.99,48.01], [46.99,47.01], [45.99,46.01], [3.95,4.05], [2.95,3.05], [1.95,2.05], [0.95,1.05], [0.48, 0.52]]
else:
    freqs_to_remove = None

#for lightcurve in lightcurves:
#    pgram = lombscargle(lightcurve[0],
#                        np.ones(lightcurve[0].shape),
#                        2*np.pi*freqs,
#                        normalize=True)
#    idx = np.where(pgram > np.max(pgram) * 0.2)[0]

if opts.doGPU and (algorithm == "PDM"):
    from cuvarbase.utils import weights
    lightcurves_pdm = []
    for lightcurve in lightcurves:
        t, y, dy = lightcurve
        lightcurves_pdm.append((t, y, weights(np.ones(dy.shape)), freqs))
    lightcurves = lightcurves_pdm 

P = (414.79153768 + 9*(0.75/1000))/86400.0
freq = 1/P
#freqs = np.append(freqs, freq)
#freqs = freq*np.ones(freqs.shape)

print('Analyzing %d lightcurves...' % len(lightcurves))
start_time = time.time()
periods_best, significances, pdots = find_periods(algorithm, lightcurves, 
                                                  freqs, 
                                                  doGPU=opts.doGPU,
                                                  doCPU=opts.doCPU,
                                                  doSaveMemory=opts.doSaveMemory,
                                                  doRemoveTerrestrial=opts.doRemoveTerrestrial,
                                                  freqs_to_remove=freqs_to_remove,
                                                  doUsePDot=opts.doUsePDot)
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

if opts.doVariability:
    sigthresh = 0.15
else:
    if algorithm == "LS":
        sigthresh = 1e6
    elif algorithm == "FFT":
        sigthresh = 0
    elif algorithm == "GCE":
        sigthresh = 7
    else:
        sigthresh = 7

if opts.doSpectra:
    lamostpage = "http://dr4.lamost.org/spectrum/png/"
    lamostfits = "http://dr4.lamost.org/spectrum/fits/"

    LAMOSTcat = os.path.join(starCatalogDir,'lamost.hdf5')
    with h5py.File(LAMOSTcat, 'r') as f:
        lamost_ra, lamost_dec = f['ra'][:], f['dec'][:]
    LAMOSTidxcat = os.path.join(starCatalogDir,'lamost_indices.hdf5')
    with h5py.File(LAMOSTidxcat, 'r') as f:
        lamost_obsid = f['obsid'][:]
        lamost_inverse = f['inverse'][:]

    lamost = SkyCoord(ra=lamost_ra*u.degree, dec=lamost_dec*u.degree, frame='icrs')    

print('Cataloging / Plotting lightcurves...')
cnt = 0
fid = open(catalogFile,'w')
for lightcurve, filt, objid, name, coordinate, absmag, bp_rp, period, significance, pdot in zip(lightcurves,filters,ids,names,coordinates,absmags,bp_rps,periods_best,significances,pdots):
    filt_str = "_".join([str(x) for x in filt])

    if opts.doLightcurveStats:
        fid.write('%s %d %.10f %.10f %.10f %.10f %.10e %s '%(name, objid, coordinate[0], coordinate[1], period, significance, pdot, filt_str))
        fid.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n"%(stats[cnt][0], stats[cnt][1], stats[cnt][2], stats[cnt][3], stats[cnt][4], stats[cnt][5], stats[cnt][6], stats[cnt][7], stats[cnt][8], stats[cnt][9], stats[cnt][10], stats[cnt][11], stats[cnt][12], stats[cnt][13], stats[cnt][14], stats[cnt][15], stats[cnt][16], stats[cnt][17], stats[cnt][18], stats[cnt][19], stats[cnt][20], stats[cnt][21], stats[cnt][22], stats[cnt][23], stats[cnt][24], stats[cnt][25], stats[cnt][26], stats[cnt][27], stats[cnt][28], stats[cnt][29], stats[cnt][30], stats[cnt][31], stats[cnt][32], stats[cnt][33], stats[cnt][34], stats[cnt][35]))
    else:
        fid.write('%s %d %.10f %.10f %.10f %.10f %.10e %s\n'%(name, objid, coordinate[0], coordinate[1], period, significance, pdot, filt_str))
        fid.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n"%(stats[cnt][0], stats[cnt][1], stats[cnt][2], stats[cnt][3], stats[cnt][4], stats[cnt][5], stats[cnt][6], stats[cnt][7], stats[cnt][8], stats[cnt][9], stats[cnt][10], stats[cnt][11], stats[cnt][12], stats[cnt][13], stats[cnt][14], stats[cnt][15], stats[cnt][16], stats[cnt][17], stats[cnt][18], stats[cnt][19], stats[cnt][20], stats[cnt][21], stats[cnt][22], stats[cnt][23], stats[cnt][24], stats[cnt][25], stats[cnt][26], stats[cnt][27], stats[cnt][28], stats[cnt][29], stats[cnt][30], stats[cnt][31], stats[cnt][32], stats[cnt][33], stats[cnt][34], stats[cnt][35]))
     
    if opts.doVariability:
        significance = stats[cnt][5]        

    if opts.doPlots and (significance>sigthresh):
        RA, Dec = coordinate
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

        if opts.doGPU and (algorithm == "PDM"):
            copy = np.ma.copy((lightcurve[0],lightcurve[1],lightcurve[2])).T
        else:
            copy = np.ma.copy(lightcurve).T

        if opts.doVariability:
            phases = copy[:,0]
        else:
            if pdot == 0:
                phases = np.mod(copy[:,0],2*period)/(2*period)
            else:
                time_vals = copy[:,0] - np.min(copy[:,0])
                phases=np.mod((time_vals-(1.0/2.0)*(pdot/period)*(time_vals)**2),2*period)/(2*period)
        magnitude, err = copy[:,1], copy[:,2]

        spectral_data = {}
        if opts.doSpectra:
            coord = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, frame='icrs')
            xid = SDSS.query_region(coord, spectro=True)
            if not xid is None:
                spec = SDSS.get_spectra(matches=xid)[0]
                for ii, sp in enumerate(spec):
                    try:
                        sp.data["loglam"]
                    except:
                        continue
                    lam = 10**sp.data["loglam"]
                    flux = sp.data["flux"]
                    key = len(list(spectral_data.keys()))
                    spectral_data[key] = {}
                    spectral_data[key]["lambda"] = lam
                    spectral_data[key]["flux"] = flux            

            sep = coord.separation(lamost).deg
            idx = np.argmin(sep)
            if sep[idx] < 3.0/3600.0:
                idy = np.where(idx == lamost_inverse)[0]
                obsids = lamost_obsid[idy]
                for obsid in obsids:
                    requestpage = "%s/%d" % (lamostfits, obsid)
 
                    with tempfile.NamedTemporaryFile(mode='w') as f:
                        wget_command = "wget %s -O %s" % (requestpage, f.name)
                        os.system(wget_command)
                        hdul = astropy.io.fits.open(f.name)
        
                    for ii, sp in enumerate(hdul):
                        lam = sp.data[2,:]
                        flux = sp.data[0,:]
                        key = len(list(spectral_data.keys()))
                        spectral_data[key] = {}
                        spectral_data[key]["lambda"] = lam
                        spectral_data[key]["flux"] = flux

        if len(spectral_data.keys()) > 0:
            #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10))
            fig = plt.figure(figsize=(25,10))
            gs = fig.add_gridspec(nrows=3, ncols=6)
            ax1 = fig.add_subplot(gs[:, 0:2])
            ax2 = fig.add_subplot(gs[:, 2:4])
            #ax3 = fig.add_subplot(gs[0, 2])
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        ax1.errorbar(phases, magnitude,err,ls='none',c='k')
        period2=period
        ymed = np.nanmedian(magnitude)
        y10, y90 = np.nanpercentile(magnitude,10), np.nanpercentile(magnitude,90)
        ystd = np.nanmedian(err)
        ymin = y10 - 7*ystd
        ymax = y90 + 7*ystd
        ax1.set_ylim([ymin,ymax])
        ax1.invert_yaxis()
        asymmetric_error = np.atleast_2d([absmag[1], absmag[2]]).T
        hist2 = ax2.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
        if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
            ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                         c='r',zorder=1,fmt='o')
        ax2.set_xlim([-1,4.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        fig.colorbar(hist2[3],ax=ax2)
        if len(spectral_data.keys()) > 0:
            bands = [[4750.0, 4950.0], [6475.0, 6650.0], [8450, 8700]]
            for jj, band in enumerate(bands):
                ax = fig.add_subplot(gs[jj, 4])
                ax_ = fig.add_subplot(gs[jj, 5])
                xmin, xmax = band[0], band[1]
                ymin, ymax = np.inf, -np.inf
                for key in spectral_data:
                    idx = np.where( (spectral_data[key]["lambda"] >= xmin) &
                                    (spectral_data[key]["lambda"] <= xmax))[0]
                    wave = spectral_data[key]["lambda"][idx]
                    myflux = spectral_data[key]["flux"][idx]
                    # quick-and-dirty normalization
                    myflux -= np.median(myflux)
                    if len(myflux) == 0: continue
                    myflux /= np.max(np.abs(myflux))
                    y1 = np.nanpercentile(myflux,1)
                    y99 = np.nanpercentile(myflux,99)
                    ydiff = y99 - y1
                    ymintmp = y1 - ydiff
                    ymaxtmp = y99 + ydiff
                    if ymin > ymintmp:
                        ymin = ymintmp
                    if ymaxtmp > ymax:
                        ymax = ymaxtmp
                    ax.plot(wave, myflux, '--')
                correlation_funcs = correlate_spec(spectral_data, band = band)
                # cross correlation
                if correlation_funcs == {}:
                    pass
                else:
                    for key in correlation_funcs:
                        ax_.plot(correlation_funcs[key]["velocity"], correlation_funcs[key]["correlation"])                     
                ax.set_ylim([ymin,ymax])
                ax.set_xlim([xmin,xmax])
                ax_.set_ylim([0,1])
                ax_.set_xlim([-1000,1000])
                if jj == len(bands)-1:
                    ax.set_xlabel('Wavelength [A]')
                    ax_.set_xlabel('Velocity [km/s]')
                #ax.set_ylabel('Flux')
                #if jj == 1:
                #    ax_.set_ylabel('Correlation amplitude')
        if pdot == 0:
            plt.suptitle(str(period2)+"_"+str(RA)+"_"+str(Dec))
        else:
            plt.suptitle(str(period2)+"_"+str(RA)+"_"+str(Dec)+"_"+str(pdot))
        fig.savefig(pngfile, bbox_inches='tight')
        plt.close()

    cnt = cnt + 1
fid.close()
