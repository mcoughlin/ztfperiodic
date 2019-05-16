#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy

import numpy as np

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import astropy
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output_quadrants/catalog/compare")
    parser.add_option("--catalog1",default="/home/mcoughlin/ZTF/output_quadrants/catalog/LS")
    parser.add_option("--catalog2",default="/home/mcoughlin/ZTF/output_quadrants/catalog/CE")
   
    parser.add_option("--sig1",default=1e6,type=float)
    parser.add_option("--sig2",default=7.0,type=float)    

    opts, args = parser.parse_args()

    return opts

def load_catalog(catalog):
    filenames = sorted(glob.glob(os.path.join(catalog,"*.dat")))[::-1]
    #filenames = filenames[:100]
    names = ["ra", "dec", "period", "sig", 
             "stats0", "stats1", "stats2", "stats3", "stats4",
             "stats5", "stats6", "stats7", "stats8", "stats9",
             "stats10", "stats11", "stats12", "stats13", "stats14",
             "stats15", "stats16", "stats17", "stats18", "stats19",
             "stats20", "stats21", "stats22", "stats23", "stats24",
             "stats25", "stats26", "stats27", "stats28", "stats29",
             "stats30", "stats31", "stats32", "stats33", "stats34",
             "stats35"]
    for ii, filename in enumerate(filenames):
        data_tmp = ascii.read(filename,names=names)
        if ii == 0:
            data = copy.copy(data_tmp)
        else:
            data = vstack([data,data_tmp])
    sig = data["sig"]
    idx = np.arange(len(sig))/len(sig)
    sigsort = idx[np.argsort(sig)]
    data["sigsort"] = sigsort
 
    return data

# Parse command line
opts = parse_commandline()

catalog1 = opts.catalog1
catalog2 = opts.catalog2
outputDir = opts.outputDir

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

cat1file = os.path.join(outputDir,'catalog_%s.fits' % catalog1.split("/")[-1])
cat2file = os.path.join(outputDir,'catalog_%s.fits' % catalog2.split("/")[-1])

if not os.path.isfile(cat1file):
    cat1 = load_catalog(catalog1)
    cat1.write(cat1file, format='fits')
else:
    cat1 = Table.read(cat1file, format='fits')

if not os.path.isfile(cat2file):
    cat2 = load_catalog(catalog2)
    cat2.write(cat2file, format='fits')
else:
    cat2 = Table.read(cat2file, format='fits')

catalog1 = SkyCoord(ra=cat1["ra"]*u.degree, dec=cat1["dec"]*u.degree, frame='icrs')
catalog2 = SkyCoord(ra=cat2["ra"]*u.degree, dec=cat2["dec"]*u.degree, frame='icrs')
idx,sep,_ = catalog1.match_to_catalog_sky(catalog2)

xs, ys, zs = [], [], []

filename = os.path.join(outputDir,'catalog.dat')
fid = open(filename,'w')
for i,ii,s in zip(np.arange(len(sep)),idx,sep):
    if s.arcsec > 1: continue
    ra, dec = cat1["ra"][i], cat1["dec"][i]
    sig1, sig2 = cat1["sig"][i], cat2["sig"][ii]
    sigsort1, sigsort2 = cat1["sigsort"][i], cat2["sigsort"][ii]
    period1, period2 = cat1["period"][i],cat2["period"][ii]

    if sig1 < opts.sig1: continue
    if sig2 < opts.sig2: continue

    xs.append(1.0/period1)
    ys.append(1.0/period2)
    ratio = np.min([sigsort1/sigsort2,sigsort2/sigsort1])
    zs.append(ratio)

    fid.write('%.5f %.5f %.5f %.5f %.5e %.5e\n' % (ra, dec,
                                                   period1, period2,
                                                   sig1, sig2))
fid.close() 

if opts.doPlots:
    pdffile = os.path.join(outputDir,'periods.pdf')
    cmap = cm.autumn

    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(1,1,1)
    sc = plt.scatter(xs,ys,c=zs,vmin=0.0,vmax=1.0,cmap=cmap,s=20,alpha=0.5)
    vals = np.linspace(np.min(xs), np.max(xs), 100)
    plt.plot(vals, vals, 'k--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    cbar = plt.colorbar(sc)
    cbar.set_label('min(LS/CE,CE/LS) significance')
    plt.xlabel('LS Frequency [1/days]')
    plt.ylabel('CE Frequency [1/days]')
    fig.savefig(pdffile)
    plt.close()

