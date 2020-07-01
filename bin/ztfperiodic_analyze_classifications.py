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
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_objids

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

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/bw/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.pnp.f.h5")

    parser.add_option("--doDifference",  action="store_true", default=False)
    parser.add_option("-d","--differencePath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.dscu.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.rrlyr.f.h5")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogPath = opts.catalogPath
differencePath = opts.differencePath   

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

scriptpath = os.path.realpath(__file__)
inputDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"input")

WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)
 
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

df = pd.read_hdf(catalogPath, 'df')
if opts.doDifference:
    differenceFiles = differencePath.split(",")
    for differenceFile in differenceFiles:
        df1 = pd.read_hdf(differenceFile, 'df')
        idx = df.index.difference(df1.index)
        df = df.loc[idx]

#objids = np.array(df.index).astype(int)
#objids_tmp, features = get_kowalski_features_objids(objids, kow)

for ii, (index, row) in enumerate(df.iterrows()): 
    if np.mod(ii,100) == 0:
        print('Loading %d/%d'%(ii,len(df)))

    objid, features = get_kowalski_features_objids([index], kow)
    period = features.period.values[0]
    amp = features.f1_amp.values[0]
    if (period < 0.1) or (period > 1.0): continue
    if (amp < 0.3): continue
    lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline = get_kowalski_objids([index], kow)

    hjd, magnitude, err = lightcurves[0]
    absmag, bp_rp = absmags[0], bp_rps[0]

    if opts.doPlots:
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
        if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
            ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                         c='r',zorder=1,fmt='o')
        ax2.set_xlim([-1,4.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        fig.colorbar(hist2[3],ax=ax2)
        plt.suptitle('Period: %.5f days' % period)
        pngfile = os.path.join(plotDir,'%d.png' % objid)
        fig.savefig(pngfile, bbox_inches='tight')
        plt.close()

