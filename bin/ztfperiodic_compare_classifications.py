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
#from tabulate import tabulate

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
from ztfperiodic.utils import get_kowalski_features_objids, get_kowalski_objids

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

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
    parser.add_option("--doPlotss",  action="store_true", default=False)
    parser.add_option("--doLowVar",  action="store_true", default=False)
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/")
    parser.add_option("-m","--modelPath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/xgboost/")

    parser.add_option("--crossmatch_distance",default=1.0,type=float)

    parser.add_option("--doField",  action="store_true", default=False)
    parser.add_option("-f","--field",default=853,type=int)

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=8,type=int)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

def load_h5(filename):

    model = filename.split("/")[-2]
    h5names = ["objid", model]
    try:
        with h5py.File(filename, 'r') as f:
            preds = f['preds'][()]
    except:
        return []
    data = {'objid': preds[:,0], model: preds[:,1]}
    data_tmp = pd.DataFrame(data)
    data_tmp.set_index('objid', inplace=True)
    if len(data_tmp) == 0:
        return []
    else:
        return data_tmp

def load_catalog(filenames,
                 doParallel=False, Ncore=8):

    if len(filenames) == 0:
        return []

    if doParallel:
        data_all = ProgressParallel(n_jobs=Ncore,use_tqdm=True,total=len(filenames))(delayed(load_h5)(filename) for filename in filenames)
    else:
        data_all = []
        for ii, filename in enumerate(filenames):
            if os.path.isfile(filename):
                data_tmp = load_h5(filename)
                if len(data_tmp) == 0: continue
                data_all.append(data_tmp)

    if len(data_all) == 0:
        return []

    # Merge the DataFrames
    df_merged = reduce(lambda  left,right: pd.merge(left,right, how='outer',
                                                    left_index=True,
                                                    right_index=True),
                       data_all)

    return df_merged

# Parse command line
opts = parse_commandline()

scriptpath = os.path.realpath(__file__)
inputDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"input")

if opts.doPlots:
    WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
    with h5py.File(WDcat, 'r') as f:
        gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
        parallax = f['parallax'][:]
    absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)

baseoutputDir = opts.outputDir
modelPaths = opts.modelPath.split(",")

if opts.doPlots:
    kow = []
    nquery = 10
    cnt = 0
    while cnt < nquery:
        try:
            TIMEOUT = 60
            protocol, host, port = "https", "gloria.caltech.edu", 443
            kow = Kowalski(username=opts.user, password=opts.pwd,
                           timeout=TIMEOUT,
                           protocol=protocol, host=host, port=port)
            break
        except:
            time.sleep(5)
        cnt = cnt + 1
    if cnt == nquery:
        raise Exception('Kowalski connection failed...')

catalogPaths = []
for modelPath in modelPaths:
    folders = glob.glob(os.path.join(modelPath,'*_*'))
    for folder in folders:
        if opts.doField:
            catalogPaths = catalogPaths + glob.glob(os.path.join(modelPath, "*", "%d_*.h5" % (opts.field)))
        else:
            catalogPaths = catalogPaths + glob.glob(os.path.join(modelPath, "*", "*.h5"))
   
modelList = []
catalogList = []
for catalogPath in catalogPaths:
    catalogSplit = catalogPath.split("/") 
    modelList.append("/".join(catalogSplit[:-1]))
    catalogList.append(catalogSplit[-1])
modelList = sorted(list(set(modelList)))
catalogList = sorted(list(set(catalogList)))

if opts.doField:
    outputDir = os.path.join(baseoutputDir, '%d' % (opts.field))
else:
    outputDir = os.path.join(baseoutputDir, 'all')
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

mergedDir = os.path.join(outputDir, 'merged')
if not os.path.isdir(mergedDir):
    os.makedirs(mergedDir)
mergedfile = os.path.join(outputDir,'catalog.h5')

baseplotDir = os.path.join(outputDir, 'plots')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

sliceDir = os.path.join(outputDir, 'slices')
if not os.path.isdir(sliceDir):
    os.makedirs(sliceDir)
  
if not os.path.isfile(mergedfile):
 
    dictlist = []
    for catalog in catalogList:
        print('Reading %s' % catalog)

        filenames = [os.path.join(model, catalog) for model in modelList]
        cat1file = os.path.join(mergedDir, catalog)

        if not os.path.isfile(cat1file):
            df = load_catalog(filenames,
                              doParallel=opts.doParallel, Ncore=opts.Ncore)
            if len(df) == 0: continue
 
            df.to_hdf(cat1file, key='df', mode='w')
        else:
            df = pd.read_hdf(cat1file)
        dictlist.append(df)

    merged = pd.concat(dictlist)
    merged.to_hdf(mergedfile, key='df', mode='w')

df_merged = pd.read_hdf(mergedfile)

for model in modelList:
    modelName = model.split("/")[-1]
    idx = np.where(df_merged[modelName] >= 0.9)[0]
    print("Model %s: %.5f%%" % (modelName, 100*len(idx)/len(df_merged[modelName])))

    if opts.doPlots:
        plotDir = os.path.join(baseplotDir, modelName)
        if not os.path.isdir(plotDir):
            os.makedirs(plotDir)
        cnt = 0
        for index, row in df_merged[modelName].iloc[idx].iteritems():
            if cnt > 10: continue

            objid, features = get_kowalski_features_objids([index], kow)
            period = features.period.values[0]
            amp = features.f1_amp.values[0]

            lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline = get_kowalski_objids([index], kow)
        
            if len(lightcurves) == 0: continue
        
            hjd, magnitude, err = lightcurves[0]
            absmag, bp_rp = absmags[0], bp_rps[0]
        
            phases = np.mod(hjd,2*period)/(2*period)
    
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
            if not np.isnan(bp_rp[0]) or not np.isnan(absmag[0]):
                ax2.errorbar(bp_rp[0],absmag[0],yerr=asymmetric_error,
                             c='r',zorder=1,fmt='o')
            ax2.set_xlim([-1,4.0])
            ax2.set_ylim([-5,18])
            ax2.invert_yaxis()
            fig.colorbar(hist2[3],ax=ax2)
            plt.suptitle('Period: %.5f days' % period)
            pngfile = os.path.join(plotDir,'%d.png' % objid)
            fig.savefig(pngfile, bbox_inches='tight')
            plt.close()
     
            cnt = cnt + 1
