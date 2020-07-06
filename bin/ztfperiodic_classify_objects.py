#!/usr/bin/env python

import os, sys
import tempfile
import time
import glob
import optparse
import pickle
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
import astropy.constants as const
import astropy.io.fits

import ztfperiodic
from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski_features
from ztfperiodic.utils import get_kowalski_features_list
from ztfperiodic.utils import get_kowalski_features_objids
from ztfperiodic.classify import classify

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

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=4,type=int)

    parser.add_option("--doQuadrantFile",  action="store_true", default=False)
    parser.add_option("--quadrant_file",default="../input/quadrant_file.dat")
    parser.add_option("--quadrant_index",default=0,type=int)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output")
    parser.add_option("-m","--modelFiles",default="/home/michael.coughlin/ZTF/ZTFVariability/pipeline/saved_models/d11.ea.f.model") 

    parser.add_option("-k","--kowalski_batch_size",default=1000,type=int)
    parser.add_option("-a","--algorithm",default="xgboost")
    parser.add_option("--default_err",default=None,type=float)

    parser.add_option("-d","--dbname",default="ZTF_source_features_20191101")

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../catalogs/swift.dat")
    parser.add_option("--Ncatalog",default=1,type=int)
    parser.add_option("--Ncatindex",default=0,type=int)

    parser.add_option("-q","--query_type",default="ids")
    parser.add_option("-i","--ids_file",default="/home/michael.coughlin/ZTF/ZTFVariability/ids/ids.20fields.npy")

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

if not opts.lightcurve_source in ["Kowalski"]:
    print("--lightcurve_source must be either Kowalski")
    exit(0)

algorithm = opts.algorithm
outputDir = opts.outputDir
catalog_file = opts.catalog_file
quadrant_file = opts.quadrant_file
Ncatalog = opts.Ncatalog
Ncatindex = opts.Ncatindex
modelFiles = opts.modelFiles.split(",")
dbname = opts.dbname

modeltype = modelFiles[0].split("/")[-1].split(".")[-2]
for modelFile in modelFiles:
    modelname = modelFile.split("/")[-1].split(".")[-2]
    if not modelname == modeltype:
        print("model types differ... please run with same types")
        exit(0)

basecatalogDir = os.path.join(outputDir,'catalog',algorithm)
if (opts.source_type == "catalog") and ("fermi" in catalog_file):
    basecatalogDir = os.path.join(basecatalogDir,'%d' % Ncatindex)
if not os.path.isdir(basecatalogDir):
    os.makedirs(basecatalogDir)

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

lightcurves = []
coordinates = []
baseline=0
fil = 'all'

print('Organizing lightcurves...')
if opts.lightcurve_source == "Kowalski":

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
        if opts.query_type == "skiplimit":
            ids, features = get_kowalski_features(kow,
                                                  num_batches=Ncatalog,
                                                  nb=Ncatindex,
                                                  featuresetname=modeltype,
                                                  dbname=dbname) 
        elif opts.query_type == "ids":
            objids = np.load(opts.ids_file)
            nlightcurves = len(objids)
            objids_split = np.array_split(objids, Ncatalog)
            objids = objids_split[Ncatindex]

            ids, features = get_kowalski_features_objids(objids, kow, 
                                                         featuresetname=modeltype,
                                                         dbname=dbname)
    elif opts.source_type == "catalog":

        amaj, amin, phi = None, None, None
        if not opts.default_err is None:
            default_err = opts.default_err
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
                elif "gaia_large_rv.dat" in catalog_file:
                    ras.append(float(lineSplit[0]))
                    decs.append(float(lineSplit[1]))
                    errs.append(default_err)
                elif ("vlss" in catalog_file):
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                elif ("apogee" in catalog_file):
                    ras.append(float(lineSplit[3]))
                    decs.append(float(lineSplit[4]))
                    errs.append(default_err)
                elif ("fermi" in catalog_file):
                    ras.append(float(lineSplit[1]))
                    decs.append(float(lineSplit[2]))
                    err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                    errs.append(err)
                    amaj.append(float(lineSplit[3]))
                    amin.append(float(lineSplit[4]))
                    phi.append(float(lineSplit[5]))
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

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").replace(".h5","").split("/")[-1]
        catalogFile = os.path.join(basecatalogDir,"%s_%d.h5" % (catalog_file_split,
                                                                Ncatindex))

        ids, features = get_kowalski_features_list(ras, decs,
                                                   kow,
                                                   errs=errs,
                                                   amaj=amaj, amin=amin,
                                                   phi=phi,
                                                   featuresetname=modeltype,
                                                   dbname=dbname)

    elif opts.source_type == "objid":

        if (".dat" in catalog_file) or (".txt" in catalog_file):
            objids = np.loadtxt(catalog_file)
        else:
            print("Sorry I don't know this file extension...")
            exit(0)
        objids = objids[:,0]         
        objids_split = np.array_split(objids,Ncatalog)
        objids = objids_split[Ncatindex]

        catalog_file_split = catalog_file.replace(".dat","").replace(".hdf5","").replace(".h5","").split("/")[-1]
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
                                doOutbursting=doOutbursting)
    else:
        print("Source type unknown...")
        exit(0)

if len(features) == 0:
    for modelFile in modelFiles:
        modelName = modelFile.replace(".model","").split("/")[-1]
        catalogDir = os.path.join(basecatalogDir, modelName)
        if not os.path.isdir(catalogDir):
            os.makedirs(catalogDir)
        catalogFile = os.path.join(catalogDir,"%d.h5"%(Ncatindex))
        touch(catalogFile)
    print('No features available... exiting.')
    exit(0)

print('Analyzing %d lightcurves...' % len(features))
start_time = time.time()

for modelFile in modelFiles:
    pred = classify(algorithm, features, modelFile=modelFile)
    data_out = np.vstack([ids, pred]).T

    modelName = modelFile.replace(".model","").split("/")[-1]
    catalogDir = os.path.join(basecatalogDir, modelName)
    if not os.path.isdir(catalogDir):
        os.makedirs(catalogDir)
    catalogFile = os.path.join(catalogDir,"%d.h5"%(Ncatindex))

    with h5py.File(catalogFile, 'w') as hf:
        hf.create_dataset("preds",  data=data_out)

end_time = time.time()
print('Lightcurve analysis took %.2f seconds' % (end_time - start_time))
