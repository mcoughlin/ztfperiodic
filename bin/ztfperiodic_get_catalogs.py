#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import astropy
from astropy.table import Table, vstack, unique
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/DESI")
    parser.add_option("--catalogs",default="/gdata/Data/PeriodSearches/v3/rosat/catalog/compare/catalog_GCE.fits,/gdata/Data/PeriodSearches/v3/xmm/catalog/compare/catalog_GCE.fits,/gdata/Data/PeriodSearches/v2/blue/catalog_GCE.fits")
    parser.add_option("--sigs",default="7.0,7.0,7.0")

    opts, args = parser.parse_args()

    return opts

def load_catalog(catalogs,sigs):

    filenames = catalogs.split(",")
    sigs = np.array(sigs.split(","),dtype=float)
    cnt = 0
    for ii, (filename, sig) in enumerate(zip(filenames,sigs)):    
        data_tmp = Table.read(filename, format='fits')
        idx = np.where(data_tmp["sig"] > sig)[0]
        data_tmp = data_tmp[idx]
        if cnt == 0:
            data = copy.copy(data_tmp)
        else:
            data = vstack([data,data_tmp])
        cnt = cnt + 1

    sig = data["sig"]
    idx = (1+np.arange(len(sig)))/len(sig+1)
    #sigsort = np.ceil(3.0*idx[np.argsort(sig)])
    sigsort = 3.0*idx[np.argsort(sig)]
    data["sigsort"] = sigsort
    data.sort("sigsort")
    data.reverse()

    ras, decs, sigs = data["ra"], data["dec"], data["sigsort"]

    objnames = []
    for ra, dec in zip(ras, decs):
        objname = "%.3f_%.3f" % (ra, dec)
        objnames.append(objname)
    data["objname"] = objnames
    data = unique(data, keys=['objname'], keep='first')

    data.sort("ra")
    
    return data

# Parse command line
opts = parse_commandline()
outputDir = opts.outputDir

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

cat = load_catalog(opts.catalogs,opts.sigs)

ras, decs, sigs = cat["ra"], cat["dec"], cat["sigsort"]

filename = os.path.join(outputDir,'MW.dat')
fid = open(filename, 'w')
for ra, dec, sig in zip(ras, decs, sigs):
    sig = 5
    fid.write('%.5f %.5f %d\n' % (ra, dec, sig))
fid.close()


