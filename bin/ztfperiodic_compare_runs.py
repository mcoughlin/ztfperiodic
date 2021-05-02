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
font = {'size'   : 30}
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

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_quadrants_Primary_DR3_HC/catalog/compare/HC_vs_normal")
    parser.add_option("--catalog1",default="/home/michael.coughlin/ZTF/output_quadrants_Primary_DR3_HC/catalog/compare/*/catalog_EAOV.fits")
    parser.add_option("--catalog2",default="/home/michael.coughlin/ZTF/output_quadrants_Primary_DR3/catalog/compare/*/catalog_EAOV.fits")

    parser.add_option("--sig1",default=10.0,type=float)
    parser.add_option("--sig2",default=13.0,type=float)

    parser.add_option("--per1",default=240.0/86400,type=float)
    parser.add_option("--per2",default=0.0,type=float)

    parser.add_option("--crossmatch_distance",default=1.0,type=float)
   
    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

catalog1 = opts.catalog1
catalog2 = opts.catalog2
outputDir = opts.outputDir

name1 = 'v1'
name2 = 'v2'

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

filename = os.path.join(outputDir,'catalog.dat')
if not os.path.isfile(filename):

    catalog1files = glob.glob(catalog1)
    for ii, catalog1 in enumerate(catalog1files):
        if ii == 0:
            cat1 = Table.read(catalog1, format='fits')
            idx1 = np.where(cat1["sig"] >= opts.sig1)[0]
            cat1 = cat1[idx1]
            idx1 = np.where(cat1["period"] >= opts.per1)[0]
            cat1 = cat1[idx1]
        else:
            cat1_tmp = Table.read(catalog1, format='fits')
            idx1 = np.where(cat1_tmp["sig"] >= opts.sig1)[0]
            cat1_tmp = cat1_tmp[idx1]
            idx1 = np.where(cat1_tmp["period"] >= opts.per1)[0]
            cat1_tmp = cat1_tmp[idx1]
            cat1 = vstack([cat1, cat1_tmp])

    catalog2files = glob.glob(catalog2)
    for ii, catalog2 in enumerate(catalog2files):
        if ii == 0:
            cat2 = Table.read(catalog2, format='fits')
            idx2 = np.where(cat2["sig"] >= opts.sig2)[0]
            cat2 = cat2[idx2]
            idx2 = np.where(cat2["period"] >= opts.per2)[0]
            cat2 = cat2[idx2]
        else:
            cat2_tmp = Table.read(catalog2, format='fits')
            idx2 = np.where(cat2_tmp["sig"] >= opts.sig2)[0]
            cat2_tmp = cat2_tmp[idx2]
            idx2 = np.where(cat2_tmp["period"] >= opts.per2)[0]
            cat2_tmp = cat2_tmp[idx2]
            cat2 = vstack([cat2, cat2_tmp])

    print(len(cat1))
    print(len(cat2))

    catalog1 = SkyCoord(ra=cat1["ra"]*u.degree, dec=cat1["dec"]*u.degree, frame='icrs')
    catalog2 = SkyCoord(ra=cat2["ra"]*u.degree, dec=cat2["dec"]*u.degree, frame='icrs')
    idx,sep,_ = catalog1.match_to_catalog_sky(catalog2)
    
    xs, ys, zs = [], [], []

    fid = open(filename,'w')
    for i,ii,s in zip(np.arange(len(sep)),idx,sep):
        if s.arcsec > opts.crossmatch_distance: continue
      
        catnum = cat1["catnum"][i]
        objid = cat1["objid"][i]
        ra1, dec1 = cat1["ra"][i], cat1["dec"][i]
        ra2, dec2 = cat2["ra"][ii], cat2["dec"][ii]
        radiff = (ra1 - ra2)*3600.0
        decdiff = (dec1 - dec2)*3600.0
    
        sig1, sig2 = cat1["sig"][i], cat2["sig"][ii]
        sigsort1, sigsort2 = cat1["sigsort"][i], cat2["sigsort"][ii]
    
        period1, period2 = cat1["period"][i],cat2["period"][ii]
    
        xs.append(1.0/period1)
        ys.append(1.0/period2)
        ratio = np.min([sigsort1/sigsort2,sigsort2/sigsort1])
        zs.append(ratio)
    
        fid.write('%d %d %.5f %.5f %.5f %.5f %.5e %.5e\n' % (catnum, objid,
                                                             ra1, dec1,
                                                             period1, period2,
                                                             sig1, sig2))
    fid.close() 

data_out = np.loadtxt(filename)

if opts.doPlots:

    xs, ys = 1.0/data_out[:,4], 1.0/data_out[:,5]

    pdffile = os.path.join(outputDir,'diffs.pdf')
    idx = np.where(data_out[:,4] != data_out[:,5])[0]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.scatter(data_out[idx,6], data_out[idx,7], s=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(pdffile, bbox_inches='tight')
    plt.close()

    pdffile = os.path.join(outputDir,'periods.pdf')
    cmap = cm.autumn

    #xedges = np.logspace(np.log10(0.02),3.0,100)
    xedges = np.logspace(0,5.0,100)
    #yedges = np.logspace(np.log10(0.02),3.0,100)
    yedges = np.logspace(-3,2.0,100)
   
    H, xedges, yedges = np.histogram2d(xs, ys, bins=(xedges, yedges))
    print(np.min(xs), np.max(xs))
    print(np.min(ys), np.max(ys))

    H = H.T  # Let each row list bins with common y range.
    X, Y = np.meshgrid(xedges, yedges)
    #H[H==0] = np.nan

    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',0)

    fig = plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    c = plt.pcolormesh(X, Y, H, vmin=1.0,vmax=np.max(H),norm=LogNorm(),
                       cmap=cmap)
    plt.plot(np.logspace(-3,5,500), np.logspace(-3,5,500), 'k--', zorder=100)
    #print(np.logspace(-3,5,500))
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.xlim([0.02, 50])
    plt.xlim([5, 500])
    #plt.ylim([0.02, 50])
    plt.ylim([0.1, 50])
    #plt.fill_between([0.02, 10],[0.02,0.02],[500,500],color='gray',alpha=0.5)
    plt.xlabel('Frequency [1/days]', fontsize=24)
    plt.ylabel('Frequency [1/days]', fontsize=24)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.savefig(pdffile)
    plt.close()
   
    pdffile = os.path.join(outputDir,'periods.pdf')
    fig.savefig(pdffile, bbox_inches='tight')
    plt.close() 

    #xedges = np.logspace(np.log10(0.02),3.0,100)
    xedges = np.logspace(np.log10(0.02),4.0,100)
    #yedges = np.logspace(np.log10(0.02),3.0,100)
    yedges = np.logspace(-3,1.0,100)

    diff = np.abs(ys-xs)/ys
    diff[diff < 1e-3] = 1e-3

    H, xedges, yedges = np.histogram2d(xs, diff, bins=(xedges, yedges))
    H = H.T  # Let each row list bins with common y range.
    X, Y = np.meshgrid(xedges, yedges)
    #H[H==0] = np.nan

    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(1,1,1)
    c = plt.pcolormesh(X, Y, H, vmin=1.0,vmax=np.max(H),norm=LogNorm(),
                       cmap=cmap)
    print(np.logspace(-3,5,500))
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.xlim([0.02, 50])
    plt.xlim([0.02, 50])
    #plt.ylim([0.02, 50])
    #plt.ylim([0.02, 500])
    plt.xlabel('Frequency [1/days]', fontsize=24)
    plt.ylabel('Relative Difference in Computed Frequency', fontsize=24)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    pdffile = os.path.join(outputDir,'periods_relative.pdf')
    fig.savefig(pdffile, bbox_inches='tight')
    plt.close()
