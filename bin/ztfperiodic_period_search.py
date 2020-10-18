#!/usr/bin/env python

import os, sys
import tempfile
import time
import glob
import optparse
import pickle
from functools import partial
import subprocess
import warnings
warnings.filterwarnings("ignore")

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
import astropy.constants as const
import astropy.io.fits

import ztfperiodic
from ztfperiodic.period import CE
from ztfperiodic.lcstats import calc_basic_stats, calc_fourier_stats
from ztfperiodic.utils import get_kowalski_bulk
from ztfperiodic.utils import get_kowalski_list
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.utils import get_simulated_list
from ztfperiodic.utils import get_matchfile
from ztfperiodic.utils import find_matchfile
from ztfperiodic.utils import convert_to_hex
from ztfperiodic.periodsearch import find_periods
from ztfperiodic.specfunc import correlate_spec, adjust_subplots_band, tick_function

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
    parser.add_option("--doRemoveBrightStars",  action="store_true", default=False)
    parser.add_option("--doSingleTimeSegment",  action="store_true", default=False)

    parser.add_option("--doRemoveHC",  action="store_true", default=False)
    parser.add_option("--doHCOnly",  action="store_true", default=False)
    parser.add_option("--doLongPeriod",  action="store_true", default=False)
    parser.add_option("--doCombineFilt",  action="store_true", default=False)
    parser.add_option("--doExtinction",  action="store_true", default=False)
    parser.add_option("--doSpectra",  action="store_true", default=False)

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=8,type=int)

    parser.add_option("--doSimulateLightcurves",  action="store_true", default=False)
    parser.add_option("--doNotPeriodFind",  action="store_true", default=False)
    parser.add_option("--doUsePDot",  action="store_true", default=False)
    parser.add_option("--doVariability",  action="store_true", default=False)
    parser.add_option("--doQuadrantFile",  action="store_true", default=False)
    parser.add_option("--quadrant_file",default="../input/quadrant_file.dat")
    parser.add_option("--quadrant_index",default=0,type=int)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    #parser.add_option("-m","--matchFile",default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc63/fr000251-000300/ztf_000259_zr_c16_q4_match.pytable") 
    parser.add_option("-m","--matchFile",default="/home/mcoughlin/ZTF/matchfiles/rc00/fr000201-000250/ztf_000245_zg_c01_q1_match.pytable")
    parser.add_option("--matchfileDir",default="/home/mcoughlin/ZTF/matchfiles/")

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

    parser.add_option("--stardist",default=13.0,type=float)
    parser.add_option("--sigthresh",default=None,type=float)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("-p","--program_ids",default="2,3")
    parser.add_option("--min_epochs",default=50,type=int)
    parser.add_option("--default_err",default=None,type=float)

    parser.add_option("--doRsyncFiles",  action="store_true", default=False)
    parser.add_option("--rsync_directory",default="mcoughlin@schoty.caltech.edu:/gdata/Data/ztfperiodic_results")

    parser.add_option("--doSigmaClipping",  action="store_true", default=False)
    parser.add_option("--sigmathresh",default=3.0,type=float)
    parser.add_option("--doOutbursting",  action="store_true", default=False)
    parser.add_option("--doCrossMatch",  action="store_true", default=False)
    parser.add_option("--crossmatch_radius",default=3.0,type=float)

    parser.add_option("--doPercentile",  action="store_true", default=False)
    parser.add_option("--percmin",default=10.0,type=float)
    parser.add_option("--percmax",default=90.0,type=float)

    parser.add_option("--doObjIDFilenames",  action="store_true", default=False)
    parser.add_option("--doCheckLightcurves",  action="store_true", default=False)

    parser.add_option("--samples_per_peak",default=10,type=int)

    opts, args = parser.parse_args()

    return opts


def brightstardist(filename,ra,dec):
     catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
     with h5py.File(filename, 'r') as f:
         ras, decs = f['ra'][:], f['dec'][:]
     c = SkyCoord(ra=ras*u.degree, dec=decs*u.degree,frame='icrs')
     idx,sep,_ = catalog.match_to_catalog_sky(c)

     seps = []
     for i,ii,s in zip(np.arange(len(sep)),idx,sep):
         if s.arcsec > 30.0:
             seps.append(s.arcsec)
         else:
             radiff = np.abs(3600*(ra[i]-ras[ii]))
             decdiff = np.abs(3600*(dec[i]-decs[ii]))
             if (radiff <= 2.0) and (decdiff <= 30.0):
                 seps.append(10.0)
             else:
                 seps.append(s.arcsec)
     seps = np.array(seps)

     return seps


def slicestardist(lightcurves, coordinates, filters, ids, absmags, bp_rps, names):

    ras, decs = [], []
    for coordinate in coordinates:
        ras.append(coordinate[0])
        decs.append(coordinate[1])
    ras, decs = np.array(ras), np.array(decs)

    filename = "%s/bsc5.hdf5" % inputDir
    sep = brightstardist(filename,ras,decs)
    idx1 = np.where(sep >= opts.stardist)[0]
    sep = brightstardist(filename,ras,decs)
    filename = "%s/Gaia.hdf5" % inputDir
    sep = brightstardist(filename,ras,decs)
    idx2 = np.where(sep >= opts.stardist)[0]
    idx = np.intersect1d(idx1,idx2).astype(int)

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

algorithms = opts.algorithm.split(',')
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
doHCOnly = opts.doHCOnly
doSimulateLightcurves = opts.doSimulateLightcurves
doUsePDot = opts.doUsePDot
doExtinction = opts.doExtinction
Ncatalog = opts.Ncatalog
Ncatindex = opts.Ncatindex
doSigmaClipping = opts.doSigmaClipping
sigmathresh = opts.sigmathresh
doOutbursting = opts.doOutbursting
doCheckLightcurves = opts.doCheckLightcurves
doCrossMatch = opts.doCrossMatch
crossmatch_radius = opts.crossmatch_radius
doPercentile=opts.doPercentile
percmin = opts.percmin
percmax = opts.percmax

if opts.doQuadrantFile:
    if opts.lightcurve_source == "Kowalski":
        quad_out = np.loadtxt(quadrant_file)
        idx = np.where(quad_out[:,0] == opts.quadrant_index)[0]
        row = quad_out[idx,:][0]
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex, Ncatalog = row[4], row[5]
    elif opts.lightcurve_source == "matchfiles":
        lines = [line.rstrip('\n') for line in open(quadrant_file)]
        for line in lines:
            lineSplit = list(filter(None,line.split(" ")))
            if int(lineSplit[0]) == opts.quadrant_index:
                matchFile = lineSplit[1]
                print("Using matchfile %s" % matchFile)
                Ncatindex = int(lineSplit[2])

scriptpath = os.path.realpath(__file__)
starCatalogDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"catalogs")
inputDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"input")

WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)

if opts.doCPU and "BLS" in algorithms:
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

catalogDir = os.path.join(outputDir,'catalog',"_".join(algorithms))
if (opts.source_type == "catalog") and ("fermi" in catalog_file):
    catalogDir = os.path.join(catalogDir,'%d' % Ncatindex)
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

if opts.doSpectra:
    spectraDir = os.path.join(outputDir,'spectra')
    if (opts.source_type == "spectra") and ("fermi" in spectra_file):
        spectraDir = os.path.join(spectraDir,'%d' % Ncatindex)
    if not os.path.isdir(spectraDir):
        os.makedirs(spectraDir)

lightcurves = []
coordinates = []
baseline=0
fil = 'all'

try:
    print('Running on host %s' % (subprocess.check_output(['hostname','-f']).decode().replace("\n","")))
except:
    pass

print('Organizing lightcurves...')
if opts.lightcurve_source == "Kowalski":

    catalogFile = os.path.join(catalogDir,"%d_%d_%d.h5"%(field, ccd, quadrant))
    if opts.doSpectra:
        spectraFile = os.path.join(spectraDir,"%d_%d_%d.pkl"%(field, ccd, quadrant))

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
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if opts.doSpectra:
            spectraFile = os.path.join(spectraDir,"%d_%d_%d_%d.pkl"%(field, ccd, quadrant,Ncatindex))

        lightcurves, coordinates, filters, ids,\
        absmags, bp_rps, names, baseline =\
            get_kowalski_bulk(field, ccd, quadrant, kow, 
                              program_ids=program_ids, min_epochs=min_epochs,
                              num_batches=Ncatalog, nb=Ncatindex,
                              doRemoveHC=doRemoveHC, doHCOnly=doHCOnly,
                              doSigmaClipping=doSigmaClipping,
                              sigmathresh=sigmathresh,
                              doPercentile=doPercentile,
                              percmin = percmin, percmax = percmax)
        if opts.doRemoveBrightStars:
            lightcurves, coordinates, filters, ids, absmags, bp_rps, names =\
                slicestardist(lightcurves, coordinates, filters,
                              ids, absmags, bp_rps, names)

    elif opts.source_type == "catalog":

        amaj, amin, phi = None, None, None
        if not opts.default_err is None:
            default_err = opts.default_err
        else:
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
                if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file) or ("lamost_mira" in catalog_file) or ("rotators" in catalog_file) or ("ap" in catalog_file):
                    ra_hex, dec_hex = convert_to_hex(float(lineSplit[0])*24/360.0,delimiter=''), convert_to_hex(float(lineSplit[1]),delimiter='')
                    if dec_hex[0] == "-":
                        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
                    else:
                        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
                    names.append(objname)
                    ras.append(float(lineSplit[0]))
                    decs.append(float(lineSplit[1]))
                    errs.append(default_err)
                elif "gaia_large_rv.dat" in catalog_file:
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
                elif ("apogee" in catalog_file):
                    names.append(lineSplit[0])
                    ras.append(float(lineSplit[3]))
                    decs.append(float(lineSplit[4]))
                    errs.append(default_err)
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

            names_split = np.array_split(names,Ncatalog)
            ras_split = np.array_split(ras,Ncatalog)
            decs_split = np.array_split(decs,Ncatalog)
            errs_split = np.array_split(errs,Ncatalog)
    
            names = names_split[Ncatindex]
            ras = ras_split[Ncatindex]
            decs = decs_split[Ncatindex]
            errs = errs_split[Ncatindex]
    
            if ("fermi" in catalog_file):
                amaj_split = np.array_split(amaj,Ncatalog)
                amin_split = np.array_split(amin,Ncatalog)
                phi_split = np.array_split(phi,Ncatalog)
    
                amaj = amaj_split[Ncatindex]
                amin = amin_split[Ncatindex]
                phi = phi_split[Ncatindex]

        elif ".hdf5" in catalog_file:
            if "underMS" in catalog_file:
                with h5py.File(catalog_file.replace(".hdf5","_ra.hdf5"), 'r') as f:
                    ras = f['ra'][:]
                with h5py.File(catalog_file.replace(".hdf5","_dec.hdf5"), 'r') as f:
                    decs = f['dec'][:]
            else:
                with h5py.File(catalog_file, 'r') as f:
                    ras, decs = f['ra'][:], f['dec'][:]
            errs = default_err*np.ones(ras.shape)

            ras_split = np.array_split(ras,Ncatalog)
            decs_split = np.array_split(decs,Ncatalog)
            errs_split = np.array_split(errs,Ncatalog)
 
            ras = ras_split[Ncatindex]
            decs = decs_split[Ncatindex]
            errs = errs_split[Ncatindex]
 
            if ("fermi" in catalog_file):
                amaj_split = np.array_split(amaj,Ncatalog)
                amin_split = np.array_split(amin,Ncatalog)
                phi_split = np.array_split(phi,Ncatalog)
 
                amaj = amaj_split[Ncatindex]
                amin = amin_split[Ncatindex]
                phi = phi_split[Ncatindex]

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
            filename = "%s/bsc5.hdf5" % inputDir
            sep = brightstardist(filename,ras,decs)
            idx1 = np.where(sep >= opts.stardist)[0]
            filename = "%s/Gaia.hdf5" % inputDir
            sep = brightstardist(filename,ras,decs)
            idx2 = np.where(sep >= opts.stardist)[0]
            idx = np.union1d(idx1,idx2)
            names = names[idx]
            ras, decs, errs = ras[idx], decs[idx], errs[idx]
            if ("fermi" in catalog_file):
                amaj, amin, phi = amaj[idx], amin[idx], phi[idx]

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").replace(".h5","").split("/")[-1]
        catalogFile = os.path.join(catalogDir,"%s_%d.h5"%(catalog_file_split,
                                                           Ncatindex))
        if opts.doSpectra:
            spectraFile = os.path.join(spectraDir,"%s_%d.pkl"%(catalog_file_split,
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
                                  doExtinction=doExtinction,
                                  doSigmaClipping=doSigmaClipping,
                                  sigmathresh=sigmathresh,
                                  doOutbursting=doOutbursting,
                                  doCrossMatch=doCrossMatch,
                                  crossmatch_radius=crossmatch_radius)

    elif opts.source_type == "objid":

        if (".dat" in catalog_file) or (".txt" in catalog_file):
            objids = np.loadtxt(catalog_file)
            objids = objids[:,0] 
        elif ".npy" in catalog_file:
            objids = np.load(catalog_file)
        else:
            print("Sorry I don't know this file extension...")
            exit(0)

        objids_split = np.array_split(objids,Ncatalog)
        objids = objids_split[Ncatindex]

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").replace(".h5","").replace(".npy","").split("/")[-1]
        catalogFile = os.path.join(catalogDir,"%s_%d.h5"%(catalog_file_split,
                                                           Ncatindex))

        if opts.doSpectra:
            spectraFile = os.path.join(spectraDir,"%s_%d.pkl"%(catalog_file_split,
                                                               Ncatindex))


        lightcurves, coordinates, filters, ids,\
        absmags, bp_rps, names, baseline =\
            get_kowalski_objids(objids, kow,
                                program_ids=program_ids,
                                min_epochs=min_epochs,
                                doRemoveHC=doRemoveHC,
                                doExtinction=doExtinction,
                                doSigmaClipping=doSigmaClipping,
                                sigmathresh=sigmathresh,
                                doOutbursting=doOutbursting,
                                doPercentile=doPercentile,
                                percmin = percmin, percmax = percmax,
                                doParallel = opts.doParallel,
                                Ncore = opts.Ncore)
    else:
        print("Source type unknown...")
        exit(0)

elif opts.lightcurve_source == "matchfiles":
    if ":" in matchFile:
        matchFile_end = matchFile.split(":")[-1].split("/")[-1]
        matchFile_out = "/scratch/mcoughlin/%s" % matchFile_end
        if not os.path.isfile(matchFile_out):
            print('Fetching %s...' % matchFile)
            wget_command = "scp -i /home/mcoughlin/.ssh/id_rsa_passwordless %s %s" % (matchFile, matchFile_out)
            os.system(wget_command)
        matchFile = matchFile_out

    if not os.path.isfile(matchFile):
        print("%s missing..."%matchFile)
        exit(0)

    matchFile_split = matchFile.replace(".pytable","").replace(".hdf5","").replace(".h5","").split("/")[-1]
    catalogFile = os.path.join(catalogDir,"%s_%d.h5"%(matchFile_split,
                                                           Ncatindex))
    if opts.doSpectra:
        spectraFile = os.path.join(spectraDir,matchFileEnd)

    #matchFile = find_matchfile(opts.matchfileDir)
    lightcurves, coordinates, filters, ids,\
    absmags, bp_rps, names, baseline = get_matchfile(matchFile,
                                                     min_epochs=min_epochs,
                                                     doRemoveHC=doRemoveHC,
                                                     doHCOnly=doHCOnly,
                                                     Ncatalog=Ncatalog,
                                                     Ncatindex=Ncatindex)

    if opts.doRemoveBrightStars:
        lightcurves, coordinates, filters, ids, absmags, bp_rps, names =\
            slicestardist(lightcurves, coordinates, filters,
                          ids, absmags, bp_rps, names)

    if len(lightcurves) == 0:
        print("No data available...")
        exit(0)

elif opts.lightcurve_source == "h5files":
    if not os.path.isfile(matchFile):
        print("%s missing..."%matchFile)
        exit(0)

    matchFileEnd = matchFile.split("/")[-1].replace("h5","h5")
    catalogFile = os.path.join(catalogDir,matchFileEnd)
    if opts.doSpectra:
        spectraFile = os.path.join(spectraDir,matchFileEnd)
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
    if opts.doSpectra:
        touch(spectraFile)
    print('No lightcurves available... exiting.')
    exit(0)

if opts.doCheckLightcurves:
    print('Just checking that there are lightcurves to analyze... exiting.')
    exit(0)

print('Running lightcurve basic stats...')
start_time = time.time()

if opts.doParallel:
    from joblib import Parallel, delayed
    stats = Parallel(n_jobs=opts.Ncore)(delayed(calc_basic_stats)(LC[0],LC[1],LC[2]) for LC in lightcurves)
else:
    stats = []
    for ii,data in enumerate(lightcurves):
        if np.mod(ii,100) == 0:
            print("%d/%d"%(ii,len(lightcurves)))
        copy = np.ma.copy(data).T
        t, mag, magerr = copy[:,0], copy[:,1], copy[:,2]

        stat = calc_basic_stats(t, mag, magerr)
        stats.append(stat)
end_time = time.time()
print('Lightcurve basic statistics took %.2f seconds' % (end_time - start_time))

if baseline<10:
    if opts.doLongPeriod:
        fmin, fmax = 18, 48
    else:
        fmin, fmax = 18, 1440
else:
    if opts.doLongPeriod:
        fmin, fmax = 2/baseline, 48
    else:
        fmin, fmax = 2/baseline, 480

print('Using baseline: %.5f, fmin: %.5f, fmax %.5f' %(baseline, fmin, fmax))

samples_per_peak = opts.samples_per_peak
phase_bins, mag_bins = 20, 10

df = 1./(samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))

freqs = fmin + df * np.arange(nf)

if opts.doRemoveTerrestrial:
    #freqs_to_remove = [[3e-2,4e-2], [47.99,48.01], [46.99,47.01], [45.99,46.01], [3.95,4.05], [2.95,3.05], [1.95,2.05], [0.95,1.05], [0.48, 0.52]]
    freqs_to_remove = [[3e-2,4e-2], [3.95,4.05], [2.95,3.05], [1.95,2.05], [0.95,1.05], [0.48, 0.52]]
else:
    freqs_to_remove = None

periodic_stats_algorithms = {}
for algorithm in algorithms:    
    if opts.doGPU and (algorithm == "PDM"):
        from cuvarbase.utils import weights
        lightcurves_pdm = []
        for lightcurve in lightcurves:
            t, y, dy = lightcurve
            lightcurves_pdm.append((t, y, weights(np.ones(dy.shape)), freqs))
        lightcurves = lightcurves_pdm 
    
    if opts.doNotPeriodFind:
        periods_best = np.ones((len(lightcurves),1))
        significances = np.ones((len(lightcurves),1))
        pdots = np.ones((len(lightcurves),1))
    else:
        print('Analyzing %d lightcurves...' % len(lightcurves))
        start_time = time.time()
        periods_best, significances, pdots = find_periods(algorithm, lightcurves, 
                                                          freqs, 
                                                          doGPU=opts.doGPU,
                                                          doCPU=opts.doCPU,
                                                          doSaveMemory=opts.doSaveMemory,
                                                          doRemoveTerrestrial=opts.doRemoveTerrestrial,
                                                          freqs_to_remove=freqs_to_remove,
                                                          doUsePDot=opts.doUsePDot,
                                                          doSingleTimeSegment=opts.doSingleTimeSegment,
                                                          doParallel=opts.doParallel,
                                                          Ncore=opts.Ncore)
        end_time = time.time()
        print('Lightcurve analysis took %.2f seconds' % (end_time - start_time))
    
    print('Running lightcurve stats...')
    start_time = time.time()

    if opts.doParallel:
        from joblib import Parallel, delayed
        periodic_stats = Parallel(n_jobs=opts.Ncore)(delayed(calc_fourier_stats)(LC[0],LC[1],LC[2],p) for LC,p in zip(lightcurves,periods_best))
    else:
        periodic_stats = []
        for ii,data in enumerate(lightcurves):
            period = periods_best[ii]
            if np.mod(ii,100) == 0:
                print("%d/%d"%(ii,len(lightcurves)))
            copy = np.ma.copy(data).T
            t, mag, magerr = copy[:,0], copy[:,1], copy[:,2]

            periodic_stat = calc_fourier_stats(t, mag, magerr, period)
            periodic_stats.append(periodic_stat)
    end_time = time.time()
    print('Lightcurve statistics took %.2f seconds' % (end_time - start_time))
    
    if not opts.sigthresh is None:
        sigthresh = opts.sigthresh
    else:
        if opts.doVariability:
            sigthresh = 0.15
        else:
            if algorithm == "LS":
                sigthresh = 1e6
            elif algorithm == "FFT":
                sigthresh = 0
            elif algorithm == "GCE":
                sigthresh = 7
            elif algorithm == "GCE_LS_AOV":
                sigthresh = 10
            elif algorithm == "ECE":
                sigthresh = 6
            elif algorithm == "EAOV":
                sigthresh = 15
            elif algorithm == "ELS":
                sigthresh = 15
            else:
                sigthresh = 7
    
    if opts.doSpectra:
        lamostpage = "http://dr5.lamost.org/spectrum/png/"
        lamostfits = "http://dr5.lamost.org/spectrum/fits/"
    
        LAMOSTcat = os.path.join(starCatalogDir,'lamost.hdf5')
        with h5py.File(LAMOSTcat, 'r') as f:
            lamost_ra, lamost_dec = f['ra'][:], f['dec'][:]
        LAMOSTidxcat = os.path.join(starCatalogDir,'lamost_indices.hdf5')
        with h5py.File(LAMOSTidxcat, 'r') as f:
            lamost_obsid = f['obsid'][:]
            lamost_inverse = f['inverse'][:]
    
        lamost = SkyCoord(ra=lamost_ra*u.degree, dec=lamost_dec*u.degree, frame='icrs')    
    
    print('Cataloging / Plotting lightcurves...')
    if opts.doSpectra:
        data_out = {}
    
    str_stats = np.empty((0,2))
    data_stats = np.empty((0,25))
    data_periodic_stats = np.empty((0,18))
    
    if baseline<10:
        basefolder = os.path.join(outputDir,'%sHC'%algorithm)
    else:
        basefolder = os.path.join(outputDir,'%s'%algorithm)
    if (opts.source_type == "catalog") and ("fermi" in catalog_file):
        basefolder = os.path.join(basefolder,'%d' % Ncatindex)
    
    cnt = 0
    fid = open(catalogFile,'w')
    for lightcurve, filt, objid, name, coordinate, absmag, bp_rp, period, significance, pdot in zip(lightcurves,filters,ids,names,coordinates,absmags,bp_rps,periods_best,significances,pdots):
        filt_str = "_".join([str(x) for x in filt])
    
        str_stats = np.append(str_stats,
                              np.array([[np.string_(name),
                                         np.string_(filt_str)]]), axis=0)
                                        
        data_stats = np.append(data_stats,
                               np.array([[objid, coordinate[0],
                                          coordinate[1],
                                          stats[cnt][0], stats[cnt][1],
                                          stats[cnt][2], stats[cnt][3],
                                          stats[cnt][4], stats[cnt][5],
                                          stats[cnt][6], stats[cnt][7],
                                          stats[cnt][8], stats[cnt][9],
                                          stats[cnt][10], stats[cnt][11],
                                          stats[cnt][12], stats[cnt][13],
                                          stats[cnt][14], stats[cnt][15],
                                          stats[cnt][16], stats[cnt][17],
                                          stats[cnt][18], stats[cnt][19],
                                          stats[cnt][20], stats[cnt][21]]]),
                               axis=0)

        data_periodic_stats = np.append(data_periodic_stats,
                               np.array([[objid,
                                          period, significance,
                                          pdot,
                                          periodic_stats[cnt][0], periodic_stats[cnt][1],
                                          periodic_stats[cnt][2], periodic_stats[cnt][3],
                                          periodic_stats[cnt][4], periodic_stats[cnt][5],
                                          periodic_stats[cnt][6], periodic_stats[cnt][7],
                                          periodic_stats[cnt][8], periodic_stats[cnt][9],
                                          periodic_stats[cnt][10], periodic_stats[cnt][11],
                                          periodic_stats[cnt][12], periodic_stats[cnt][13]]]),
                               axis=0)
    
        if opts.doPlots and ((period/(1.0/fmax)) <= 1.05):
            print("%d %.5f %.5f %d: Period is within 5 per." % (objid, coordinate[0], coordinate[1], stats[cnt][0]))
    
        if opts.doVariability:
            significance = stats[cnt][9]        
    
        if opts.doSpectra:
            data_out[name] = {}
            data_out[name]["name"] = name
            data_out[name]["objid"] = objid
            data_out[name]["RA"] = coordinate[0]
            data_out[name]["Dec"] = coordinate[1]
            data_out[name]["period"] = period
            data_out[name]["significance"] = significance
            data_out[name]["pdot"] = pdot
            data_out[name]["filt"] = filt
            data_out[name]["stats"] = stats[cnt]
  
        if opts.doPlots and (significance>sigthresh):
            if opts.doHCOnly and np.isclose(period, 1.0/fmin, rtol=1e-2):
                print("Vetoing... period is 1/fmax")
                continue
    
            RA, Dec = coordinate
            if opts.doObjIDFilenames:
                figfile = "%d.png" % objid
            else:
                figfile = "%.10f_%.10f_%.10f_%.10f_%s.png"%(significance, RA, Dec,
                                                          period, "".join(filt_str))
    
                if opts.doNotPeriodFind:
                    thisfolder = 'noperiod'
                else:
                    idx = np.where((period>=period_ranges[:-1]) & (period<=period_ranges[1:]))[0][0]
                    thisfolder = folders[idx.astype(int)]
                    if thisfolder == None:
                        continue
    
            if opts.doGPU and (algorithm == "PDM"):
                copy = np.ma.copy((lightcurve[0],lightcurve[1],lightcurve[2])).T
            else:
                copy = np.ma.copy(lightcurve).T
    
            if opts.doObjIDFilenames:
                objid_str = str(objid)
                folder = os.path.join(basefolder, objid_str[2], objid_str[3])
            else:
                nepoch = np.array(len(copy[:,0]))
                idx2 = np.where((nepoch>=epoch_ranges[:-1]) & (nepoch<=epoch_ranges[1:]))[0][0]
                if epoch_folders[idx2.astype(int)] == None:
                    continue
    
                folder = os.path.join(basefolder,thisfolder,epoch_folders[idx2.astype(int)])
            if not os.path.isdir(folder):
                os.makedirs(folder)
            pngfile = os.path.join(folder,figfile)
    
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
                try:
                    xid = SDSS.query_region(coord, spectro=True)
                except:
                    xid = None
                if not xid is None:
                    try:
                        spec = SDSS.get_spectra(matches=xid)[0]
                    except:
                        spec = []
                        pass
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
            print(bp_rp)
            if not np.isnan(bp_rp[0]) or not np.isnan(absmag[0]):
                ax2.errorbar(bp_rp[0],absmag[0],yerr=asymmetric_error,
                             c='r',zorder=1,fmt='o')
            ax2.set_xlim([-1,4.0])
            ax2.set_ylim([-5,18])
            ax2.invert_yaxis()
            fig.colorbar(hist2[3],ax=ax2)
            nspec = len(spectral_data.keys())
            npairs = 0
            if nspec > 1:
                bands = [[4750.0, 4950.0], [6475.0, 6650.0], [8450, 8700]]
                npairs = int(nspec * (nspec-1)/2)
                v_values = np.zeros((len(bands), npairs))
                v_values_unc = np.zeros((len(bands), npairs))
                data_out[name]["spectra"] = {}
                for jj, band in enumerate(bands):
                    data_out[name]["spectra"][jj] = np.empty((0,3))
    
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
                        if len(correlation_funcs) == 1:
                            yheights = [0.5]
                        else:
                            yheights = np.linspace(0.25,0.75,len(correlation_funcs))
                        for kk, key in enumerate(correlation_funcs):
                            if not 'v_peak' in correlation_funcs[key]:
                                continue
                            vpeak = correlation_funcs[key]['v_peak']
                            vpeak_unc = correlation_funcs[key]['v_peak_unc']
                            Cpeak = correlation_funcs[key]['C_peak']
                            ax_.plot(correlation_funcs[key]["velocity"], correlation_funcs[key]["correlation"])
                            ax_.plot([vpeak, vpeak], [0, Cpeak], 'k--')
                            ax_.text(250, yheights[kk], "v=%.0f +- %.0f"%(vpeak, vpeak_unc))
                            v_values[jj][kk] = vpeak
                            v_values_unc[jj][kk] = vpeak_unc
                            data_out[name]["spectra"][jj] = np.vstack((data_out[name]["spectra"][jj], [vpeak, vpeak_unc, Cpeak]))
    
                    if np.isfinite(ymin) and np.isfinite(ymax):
                        ax.set_ylim([ymin,ymax])
                    ax.set_xlim([xmin,xmax])
                    ax_.set_ylim([0,1])
                    ax_.set_xlim([-1000,1000])
                    if jj == len(bands)-1:
                        ax.set_xlabel('Wavelength [A]')
                        ax_.set_xlabel('Velocity [km/s]')
                        adjust_subplots_band(ax, ax_)
                    else:
                        ax_.set_xticklabels([])
                    if jj==1:
                        new_tick_locations = np.array([-1000, -500, 0, 500, 1000])
                        axmass = ax_.twiny()
                        axmass.set_xlim(ax_.get_xlim())
                        axmass.set_xticks(new_tick_locations)
                        tick_labels = tick_function(new_tick_locations, period)
                        tick_labels = ["{0:.0f}".format(float(x)) for x in tick_labels]
                        axmass.set_xticklabels(tick_labels)
                        axmass.set_xlabel("f($M$) ("+r'$M_\odot$'+')')
            if npairs > 0:
                # calculate mass functon
                if npairs==1:
                    id_pair = 0
                else:
                    # select a pair with:
                    # (1) reasonable variance among all band measurements
                    stds = np.std(v_values, axis=0)
                    if np.sum(stds<50)>=1:
                        v_values = v_values[:, stds<50]
                        v_values_unc = v_values_unc[:, stds<50]
                    # (2) largest (absolute) velosity variation
                    vsums = np.sum(abs(v_values), axis=0)
                    id_pair = np.where(vsums == max(vsums))[0][0]
                v_adopt = np.median(v_values[:,id_pair])
                id_band = np.where(v_values[:,id_pair]==v_adopt)[0][0]
                v_adopt_unc = v_values_unc[id_band,id_pair]
                K = abs(v_adopt/2.) # [km/s] assuming that the velocity variation is max and min in rv curve
                K_unc = abs(v_adopt_unc/2.) # [km/s]
                P = 2*period # [day] if ellipsodial modulation, amplitude are roughly the same, 
                            # then the photometric period is probably half of the orbital period
                fmass = (K * 100000)**3 * (P*86400) / (2*np.pi*const.G.cgs.value) / const.M_sun.cgs.value
                fmass_unc = 3 * fmass / K * K_unc
                data_out[name]["fmass"] = fmass
                data_out[name]["fmass_unc"] = fmass_unc
            if pdot == 0:
                plt.suptitle(str(period2)+"_"+str(RA)+"_"+str(Dec))
            else:
                plt.suptitle(str(period2)+"_"+str(RA)+"_"+str(Dec)+"_"+str(pdot))
            fig.savefig(pngfile, bbox_inches='tight')
            plt.close()
    
        cnt = cnt + 1

    periodic_stats_algorithms[algorithm] = data_periodic_stats

with h5py.File(catalogFile, 'w') as hf:
    hf.create_dataset("names",  data=str_stats[:,0])
    hf.create_dataset("filters",  data=str_stats[:,1])
    hf.create_dataset("stats",  data=data_stats)

    for algorithm in algorithms:
        hf.create_dataset("stats_%s" % algorithm,
                          data=periodic_stats_algorithms[algorithm])

if opts.doSpectra:
    with open(spectraFile, 'wb') as handle:
        pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

if opts.doRsyncFiles:
    outputDirSplit = outputDir.split("/")[-1]
    rsync_command = "rsync -zarvh %s %s" % (outputDir,
                                            opts.rsync_directory)
    print(rsync_command)
    os.system(rsync_command)
