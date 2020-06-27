#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
from functools import reduce

import numpy as np
import pandas as pd

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

    parser.add_option("-f","--featuresetname",default="b")

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
featuresetname = opts.featuresetname

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

plotDir = os.path.join(outputDir, 'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

catalogPaths = glob.glob(os.path.join(modelPath, "*.*.%s" % featuresetname))
dictlist = []
for catalogPath in catalogPaths:
    modelName = catalogPath.split("/")[-1]
    cat1file = os.path.join(outputDir,'catalog_%s.fits' % modelName)

    if not os.path.isfile(cat1file):
        cat1 = load_catalog(catalogPath)
        cat1.write(cat1file, format='fits')
    else:
        cat1 = Table.read(cat1file, format='fits')
        continue

    idx = np.where(cat1["prob"] >= 0.9)[0]
    print("Model %s: %.5f%%" % (modelName, 100*len(idx)/len(cat1["prob"])))

    df = cat1.to_pandas()
    df.rename(columns={"prob": modelName}, inplace=True)
    df.set_index('objid', inplace=True)
    dictlist.append(df)

    if opts.doPlots:
        pdffile = os.path.join(plotDir,'%s.pdf' % modelName)
        fig = plt.figure(figsize=(10,8))
        ax=fig.add_subplot(1,1,1)
        plt.hist(cat1["prob"][cat1["prob"]>0.1])
        plt.title(modelName)
        plt.xlabel('Probability')
        plt.ylabel('Counts')
        fig.savefig(pdffile)
        plt.close()

cat1file = os.path.join(outputDir,'catalog.h5')
if not os.path.isfile(cat1file):
    # Merge the DataFrames
    df_merged = reduce(lambda  left,right: pd.merge(left,right, how='outer',
                                                    left_index=True,
                                                    right_index=True),
                       dictlist)
    df_merged.to_hdf(cat1file, key='df_merged', mode='w')
else:
    df_merged = pd.read_hdf(cat1file)

psums = df_merged.drop(columns=['d11.pnp.f', 'd11.vnv.f']).sum(axis=1)
cat1file = os.path.join(outputDir,'catalog_slice.h5')
df_merged.loc[psums>1.0].to_hdf(cat1file, key='df_merged', mode='w')

if opts.doPlots:
    pdffile = os.path.join(plotDir,'%s.pdf' % 'summed')
    fig = plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    plt.hist(psums.loc[psums>1.0])
    plt.xlabel('Sum (Probability)')
    plt.ylabel('Counts')
    fig.savefig(pdffile)
    plt.close()
