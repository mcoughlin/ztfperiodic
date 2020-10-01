#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
from functools import reduce

from joblib import Parallel, delayed
from tqdm import tqdm

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


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("--doLowVar",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/")
    parser.add_option("-m","--modelPath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/xgboost/")

    parser.add_option("--crossmatch_distance",default=1.0,type=float)

    parser.add_option("-f","--featuresetname",default="b")

    parser.add_option("--catalog_min",type=int,default=0)
    parser.add_option("--catalog_max",type=int,default=100000)

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=8,type=int)

    opts, args = parser.parse_args()

    return opts

def load_h5(filename):

    h5names = ["objid", "prob"]
    try:
        with h5py.File(filename, 'r') as f:
            preds = f['preds'][()]
    except:
        return []
    data_tmp = Table(rows=preds, names=h5names)
    if len(data_tmp) == 0:
        return []
    else:
        return data_tmp

def load_catalog(catalog, catalog_min=0, catalog_max=100000,
                 doParallel=False, Ncore=8):

    filenames = sorted(glob.glob(os.path.join(catalog,"*.dat")))[::-1] + \
                sorted(glob.glob(os.path.join(catalog,"*.h5")))[::-1]

    idx = []
    for ii, filename in enumerate(filenames):
        filenameSplit = filename.split("/")
        catnum = int(filenameSplit[-1].replace(".dat","").replace(".h5","").split("_")[-1])

        if (catnum < catalog_min) or (catnum > catalog_max):
            continue

        idx.append(ii)
    filenames = [filenames[ii] for ii in idx]

    if doParallel:
        data_all = ProgressParallel(n_jobs=Ncore,use_tqdm=True,total=len(filenames))(delayed(load_h5)(filename) for filename in filenames)
    else:
        data_all = []
        for ii, filename in enumerate(filenames):
            data_tmp = load_h5(filename)
            data_all.append(data_tmp)
    data = vstack(data_all)

    return data

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
modelPath = opts.modelPath
featuresetname = opts.featuresetname
catalog_min = opts.catalog_min
catalog_max = opts.catalog_max

outputDir = os.path.join(outputDir, '%d_%d' % (catalog_min, catalog_max))
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

plotDir = os.path.join(outputDir, 'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

sliceDir = os.path.join(outputDir, 'slices')
if not os.path.isdir(sliceDir):
    os.makedirs(sliceDir)

catalogPaths = glob.glob(os.path.join(modelPath, "*.*.%s" % featuresetname))
dictlist = []
for catalogPath in catalogPaths:
    modelName = catalogPath.split("/")[-1]
    cat1file = os.path.join(outputDir,'catalog_%s.fits' % modelName)

    if not os.path.isfile(cat1file):
        cat1 = load_catalog(catalogPath, catalog_min=catalog_min,
                            catalog_max=catalog_max, 
                            doParallel=opts.doParallel, Ncore=opts.Ncore)
        cat1.write(cat1file, format='fits')
    else:
        cat1 = Table.read(cat1file, format='fits')
    #    continue

    idx = np.where(cat1["prob"] >= 0.9)[0]
    print("Model %s: %.5f%%" % (modelName, 100*len(idx)/len(cat1["prob"])))

    df = cat1.to_pandas()
    df.rename(columns={"prob": modelName}, inplace=True)
    df.set_index('objid', inplace=True)
    dictlist.append(df)

    cat1file = os.path.join(sliceDir,'%s.h5' % modelName)
    df.loc[cat1["prob"] >= 0.9].to_hdf(cat1file, key='df', mode='w')

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

psums = df_merged.drop(columns=['d11.pnp.%s' % featuresetname,
                                'd11.vnv.%s' % featuresetname]).sum(axis=1)
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
