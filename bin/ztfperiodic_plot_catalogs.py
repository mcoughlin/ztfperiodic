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
import matplotlib.patches as patches

import astropy
from astropy.table import Table, vstack, hstack
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

from ztfperiodic.utils import angular_distance
from ztfperiodic.utils import convert_to_hex

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("-o","--outputFile", default="/home/michael.coughlin/ZTF/output_quadrants_20Fields_DR5/catalog/compare/plots/853/catalog_EAOV_g.png")
    parser.add_option("-c","--catalogFile", default="/home/michael.coughlin/ZTF/output_quadrants_20Fields_DR5/catalog/compare/853/catalog_EAOV_g.fits")
    parser.add_option("-t","--threshold", type=float, default=20)

    opts, args = parser.parse_args()

    return opts

def ScatterPlotSigAboveThreshold(cat, threshold,  plotName):

    idx = np.where(cat["sig"] >= threshold)[0]
    ra, dec, sig = cat["ra"][idx], cat["dec"][idx], cat["sig"][idx]

    fig = plt.figure(figsize=(10,6))
    ax = fig.gca()

    cax = ax.scatter(x=ra, y=dec, c=sig, cmap=plt.cm.RdYlBu)

    cbar = fig.colorbar(cax, ticks = [30, 100, 200,300, 400])
    cbar.ax.set_yticklabels(['<30', '100', '200', '300', '>400'])
    cbar.set_label(f"Significance (values above {threshold}")

    plt.title(f"Scatter Plot (Significance above {threshold}", fontsize=20)
    plt.xlabel('RA', fontsize=16)
    plt.ylabel('DEC', fontsize=16)
    plt.show()
    plt.savefig(plotName)
    plt.close()

# Parse command line
opts = parse_commandline()
outputFile = opts.outputFile
catalogFile = opts.catalogFile

cat = Table.read(catalogFile, format='fits')
ScatterPlotSigAboveThreshold(cat, opts.threshold, outputFile)

