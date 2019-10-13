#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import pickle

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
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("--doFermi",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output_lamost_multiepoch_snr/spectra/radial_velocities")
    parser.add_option("--catalog",default="/home/mcoughlin/ZTF/output_lamost_multiepoch_snr/spectra/GCE")
   
    parser.add_option("--sig",default=7.0,type=float)

    opts, args = parser.parse_args()

    return opts

def load_catalog(catalog,doFermi=False):

    if doFermi:
        filenames = sorted(glob.glob(os.path.join(catalog,"*/*.pkl")))[::-1]
    else:
        filenames = sorted(glob.glob(os.path.join(catalog,"*.pkl")))[::-1]

    velall = []

    cnt = 0
    for ii, filename in enumerate(filenames):
        filenameSplit = filename.split("/")
        catnum = filenameSplit[-1].replace(".dat","").split("_")[-1]

        with open(filename, 'rb') as handle:
            data_out = pickle.load(handle)
        for name in data_out.keys():
            if not "spectra" in data_out[name]: continue
            ra, dec = data_out[name]["RA"], data_out[name]["Dec"]
            period = data_out[name]["period"]
            significance = data_out[name]["significance"]
            spectra = data_out[name]["spectra"]
            vels, velerrs = [], []
            allneg, allpos = True, True
            diffs = []
            for key in spectra.keys():
                if len(spectra[key]) < 2: continue
                diff = np.sum(np.abs(np.diff(spectra[key][:,0])))
                diffs.append(diff) 
                for row in spectra[key]:
                    snr = row[0]/row[1]
                    Cpeak = row[2]
                    if np.abs(row[0]) > 1000: continue
                    if Cpeak < 0.5: continue
                    if snr < 3: continue
                    vels.append(row[0])
                    velerrs.append(row[1])
                    if row[0] >= 0:
                        allneg = False
                    if row[0] <= 0:
                        allpos = False
                    #print(ra, dec, period, significance, row)
            vels, velerrs = np.array(vels), np.array(velerrs)
            difftot = np.mean(diffs)
            if len(vels) == 0: continue
            if len(vels) == 1:
                vel, velerr = vels[0], velerrs[0]
            else:
                weights = 1/velerrs
                weights = weights / np.sum(weights)
                vel = np.average(vels, weights=weights)
                velerr = np.sqrt(np.average((vels-vel)**2, weights=weights))
            if vel == 0.0: continue
            snr = np.abs(vel/velerr)
            velall.append(vel)
            if difftot > 50:
                print(ra, dec, period, significance, vel)
            #if allneg or allpos:
            #    if (np.abs(vel) > 20) and (significance>8):
            #        print(ra, dec, period, significance, vel)
            #if snr > 3:
            #    print(ra, dec, period, significance, vel, velerr, snr)
        cnt = cnt + 1

    return velall

# Parse command line
opts = parse_commandline()

catalog = opts.catalog
outputDir = opts.outputDir

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

name = list(filter(None,catalog.split("/")))[-1]
catfile = os.path.join(outputDir,'catalog_%s.fits' % name)

if not os.path.isfile(catfile):
    vels = load_catalog(catalog,doFermi=opts.doFermi)

if opts.doPlots:
    pdffile = os.path.join(outputDir,'velocities.pdf')
    cmap = cm.autumn
    Nbins = 50
    bins = np.linspace(-200,200,Nbins)
    hist1, bin_edges = np.histogram(vels, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-10
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0    

    fig = plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    plt.step(bins,hist1,'-',color='k',linewidth=3,where='mid')
    ax.set_yscale('log')
    plt.xlabel('Velocities [km/s]')
    plt.ylabel('Counts')
    plt.ylim([1e-5,1])
    fig.savefig(pdffile)
    plt.close()

