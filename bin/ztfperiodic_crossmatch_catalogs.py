#!/usr/bin/env python

import os, sys
import time
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py
from scipy.signal import lombscargle

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy import units as u
from astropy.coordinates import SkyCoord

import ztfperiodic
from ztfperiodic.period import CE
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import angular_distance
from ztfperiodic.utils import convert_to_hex
from ztfperiodic.periodsearch import find_periods

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--catalog_file",default="../catalogs/chandra_f.dat")
    parser.add_option("--catalog_file_1",default="../catalogs/fermi.dat")
    parser.add_option("--catalog_file_2",default="../catalogs/chandra.dat")

    opts, args = parser.parse_args()

    return opts

def read_catalog(catalog_file):

    amaj, amin, phi = None, None, None
    default_err = 5.0

    if ".dat" in catalog_file:
        lines = [line.rstrip('\n') for line in open(catalog_file)]
        names, ras, decs, errs = [], [], [], []
        if ("fermi" in catalog_file) or ("chandra" in catalog_file):
            amaj, amin, phi = [], [], []
        for line in lines:
            lineSplit = list(filter(None,line.split(" ")))
            if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file) or ("wd_rd" in catalog_file) or ("wd_bd" in catalog_file) or ("cyclotron" in catalog_file) or ("elm_wd" in catalog_file) or ("amcvn" in catalog_file) or ("sdb_dm" in catalog_file) or ("wdb_noneclipsing" in catalog_file):
                ra_hex, dec_hex = convert_to_hex(float(lineSplit[0])*24/360.0,delimiter=''), convert_to_hex(float(lineSplit[1]),delimiter='')
                if dec_hex[0] == "-":
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
                else:
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
                names.append(objname)
                ras.append(float(lineSplit[0]))
                decs.append(float(lineSplit[1]))
                errs.append(default_err)
            elif ("vlss" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                errs.append(err)
            elif ("fermi" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                errs.append(err)
                amaj.append(float(lineSplit[3]))
                amin.append(float(lineSplit[4]))
                phi.append(float(lineSplit[5]))
            elif ("chandra" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)/3600.0
                errs.append(err)
                amaj.append(float(lineSplit[3])/3600.0)
                amin.append(float(lineSplit[4])/3600.0)
                phi.append(float(lineSplit[5]))
            elif ("swift" in catalog_file) or ("xmm" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = float(lineSplit[3])
                errs.append(err)
            else:
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                errs.append(default_err)
        names = np.array(names)
        ras, decs, errs = np.array(ras), np.array(decs), np.array(errs)
        if ("fermi" in catalog_file) or ("chandra" in catalog_file):
            amaj, amin, phi = np.array(amaj), np.array(amin), np.array(phi)
    elif ".hdf5" in catalog_file:
        with h5py.File(catalog_file, 'r') as f:
            ras, decs = f['ra'][:], f['dec'][:]
        errs = default_err*np.ones(ras.shape)

        names = []
        for ra, dec in zip(ras, decs):
            ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)
        names = np.array(names)

    return names, ras, decs, errs, amaj, amin, phi

# Parse command line
opts = parse_commandline()

names_1, ras_1, decs_1, errs_1, amaj_1, amin_1, phi_1 = read_catalog(opts.catalog_file_1)
names_2, ras_2, decs_2, errs_2, amaj_2, amin_2, phi_2 = read_catalog(opts.catalog_file_2)

idxs = []
idys = []

if not amaj_1 is None:
    for ii, (name, ra, dec, err, amaj, amin, phi) in enumerate(zip(names_1, ras_1, decs_1, errs_1, amaj_1, amin_1, phi_1)):

        # 1 degree
        if amaj >= 30/60: continue

        if np.mod(ii,100) == 0:
            print('%d/ %d' % (ii, len(ras_1)))

        dist = angular_distance(ra, dec, ras_2, decs_2)
        ellipse = patches.Ellipse((ra, dec), 2*amaj, 2*amin, angle=phi)
        idx = np.where(dist <= 2*amaj)[0]
        for jj in idx:
            if not ellipse.contains_point((ras_2[jj],decs_2[jj])):
                continue
            idxs.append(jj)
            idys.append(ii)

            print(name, ra, dec, names_2[jj], ras_2[jj], decs_2[jj])

else:
    for ii, (name, ra, dec, err) in enumerate(zip(names_1, ras_1, decs_1, errs_1)):
        if np.mod(ii,100) == 0:
            print('%d/ %d' % (ii, len(ras_1)))

        dist = angular_distance(ra, dec, ras_2, decs_2)
        idx = np.where(dist <= err/3600.0)[0]
        for jj in idx:
            idxs.append(jj)
            idys.append(ii)

            print(name, ra, dec, names_2[jj], ras_2[jj], decs_2[jj])

idxs = np.unique(idxs)

lines = [line.rstrip('\n') for line in open(opts.catalog_file_2)]
fid = open(opts.catalog_file, 'w')
for ii, line in enumerate(lines):
    if ii in idxs:
        fid.write('%s\n' % line)
fid.close()

