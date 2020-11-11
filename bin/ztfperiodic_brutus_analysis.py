#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import pickle
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

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/condor")

    opts, args = parser.parse_args()

    return opts


# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

catalogDir = os.path.join(outputDir,'catalog')
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

h5Dir = os.path.join(outputDir,'h5')
if not os.path.isdir(h5Dir):
    os.makedirs(h5Dir)

numpyDir = os.path.join(outputDir,'numpy')
if not os.path.isdir(numpyDir):
    os.makedirs(numpyDir)

nmpyfile = os.path.join(plotDir,'smf.npy')
if not os.path.isfile(nmpyfile):
    filenames = glob.glob(os.path.join(numpyDir,'*.npy'))
    smfrac = []
    for ii, filename in enumerate(filenames):

        objid = int(filename.split("/")[-1].split(".")[0].split("_")[0])
        filenameh5 = os.path.join(h5Dir, '%d.h5' % objid)

        # load results
        f = h5py.File(filenameh5, 'r')
        idxs_mist = f['model_idx'][:]  # model indices
        chi2_mist = f['obj_chi2min'][:]  # best-fit chi2
        nbands_mist = f['obj_Nbands'][:]  # number of bands in fit
        dists_mist = f['samps_dist'][:]  # distance samples
        reds_mist = f['samps_red'][:]  # A(V) samples
        dreds_mist = f['samps_dred'][:]  # R(V) samples
        lnps_mist = f['samps_logp'][:]  # log-posterior of samples

        print(f.keys())
        print(chi2_mist)
        print(stop)

        #if ii > 10000: continue
        if np.mod(ii,100) == 0:
            print("%d/%d"%(ii,len(filenames)))
    
        smfsamp = np.load(filename)
        frac = len(np.where(smfsamp > 0.5)[0])/len(smfsamp)
        #print('Fraction of smf > 0.5: %.5f' % (frac))
        smfrac.append(frac)

with open(nmpyfile, 'wb') as f:
    np.save(f, smfrac)

plt.figure(figsize=(12,8))
n, bins, patches = plt.hist(smfrac, 50, facecolor='green', alpha=0.75)
plt.xlabel('SMF > 0.5')
plt.ylabel('Counts')
plt.show()
plotName = os.path.join(plotDir,'smf.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')
