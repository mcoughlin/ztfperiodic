#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py

import numpy as np

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import astropy
from astropy.table import Table, vstack
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

from ztfperiodic.utils import convert_to_hex

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/")
    parser.add_option("-m","--modelPath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/xgboost/")

    parser.add_option("--crossmatch_distance",default=1.0,type=float)

    opts, args = parser.parse_args()

    return opts

def load_catalog(catalog):

    filenames = sorted(glob.glob(os.path.join(catalog,"*.dat")))[::-1] + \
                sorted(glob.glob(os.path.join(catalog,"*.h5")))[::-1]
                     
    h5names = ["objid", "prob"]

    cnt = 0
    #filenames = filenames[:500]
    for ii, filename in enumerate(filenames):
        if np.mod(ii,100) == 0:
            print('Loading file %d/%d' % (ii, len(filenames)))

        filenameSplit = filename.split("/")
        catnum = filenameSplit[-1].replace(".dat","").replace(".h5","").split("_")[-1]

        try:
            with h5py.File(filename, 'r') as f:
                preds = f['preds'].value
        except:
            continue
        data_tmp = Table(rows=preds, names=h5names)
        if len(data_tmp) == 0: continue

        if cnt == 0:
            data = copy.copy(data_tmp)
        else:
            data = vstack([data,data_tmp])
        cnt = cnt + 1

    return data

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
modelPath = opts.modelPath

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

catalogPaths = glob.glob(os.path.join(modelPath, '*.*'))
for catalogPath in catalogPaths:
    modelName = catalogPath.split("/")[-1]
    cat1file = os.path.join(outputDir,'catalog_%s.fits' % modelName)

    if not os.path.isfile(cat1file):
        cat1 = load_catalog(catalogPath)
        cat1.write(cat1file, format='fits')
    else:
        cat1 = Table.read(cat1file, format='fits')

    idx = np.where(cat1["prob"] >= 0.9)[0]
    print("Model %s: %.5f%%" % (modelName, 100*len(idx)/len(cat1["prob"])))

    if opts.doPlots:
        pdffile = os.path.join(outputDir,'magnitude.pdf')
        fig = plt.figure(figsize=(10,10))
        ax=fig.add_subplot(1,1,1)
        hist2 = plt.hist2d(data_out[:,6], data_out[:,7], bins=100,
                           zorder=0,norm=LogNorm())
        plt.xlabel('IQR')
        plt.ylabel('Magnitude')
        fig.savefig(pdffile)
        plt.close()
