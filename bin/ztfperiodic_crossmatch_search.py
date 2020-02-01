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
from astropy.table import Table

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

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_Fermi_GCE_LS_AOV/catalog/crossmatch")
    parser.add_option("--doFermi",  action="store_true", default=False)

    parser.add_option("--catalog",default="/home/michael.coughlin/ZTF/output_Fermi_GCE_LS_AOV/catalog/GCE_LS_AOV/")
    parser.add_option("--catalog_file",default="../catalogs/chandra.dat")

    parser.add_option("--sig",default=10.0,type=float)

    opts, args = parser.parse_args()

    return opts

def read_catalog(catalog_file):

    amaj, amin, phi = None, None, None
    default_err = 5.0

    if ".dat" in catalog_file:
        lines = [line.rstrip('\n') for line in open(catalog_file)]
        names, ras, decs, errs = [], [], [], []
        if ("fermi" in catalog_file) or ("chandra" in catalog_file) or ("rosat" in catalog_file) or ("swift" in catalog_file) or ("xmm" in catalog_file):
            amaj, amin, phi = [], [], []
        for line in lines:
            lineSplit = list(filter(None,line.split(" ")))
            if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file):
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
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)
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
                amaj.append(err/3600.0)
                amin.append(err/3600.0)
                phi.append(0.0)
            elif ("rosat" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                errs.append(default_err)
                amaj.append(default_err/3600.0)
                amin.append(default_err/3600.0)
                phi.append(0.0)
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

catalog = opts.catalog
doFermi = opts.doFermi

outputDir = opts.outputDir

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

if doFermi:
    filenames = sorted(glob.glob(os.path.join(catalog,"*/*.dat")))[::-1] + \
                sorted(glob.glob(os.path.join(catalog,"*/*.h5")))[::-1]
else:
    filenames = sorted(glob.glob(os.path.join(catalog,"*.dat")))[::-1] + \
                sorted(glob.glob(os.path.join(catalog,"*.h5")))[::-1]

cnames, ras, decs, errs, amajs, amins, phis = read_catalog(opts.catalog_file)

#filenames = filenames[:100]
names = ["name", "objid", "ra", "dec", "period", "sig", "pdot", "filt",
         "stats0", "stats1", "stats2", "stats3", "stats4",
         "stats5", "stats6", "stats7", "stats8", "stats9",
         "stats10", "stats11", "stats12", "stats13", "stats14",
         "stats15", "stats16", "stats17", "stats18", "stats19",
         "stats20", "stats21", "stats22", "stats23", "stats24",
         "stats25", "stats26", "stats27", "stats28", "stats29",
         "stats30", "stats31", "stats32", "stats33", "stats34",
         "stats35"]
h5names = ["objid", "ra", "dec", "period", "sig", "pdot",
           "stats0", "stats1", "stats2", "stats3", "stats4",
           "stats5", "stats6", "stats7", "stats8", "stats9",
           "stats10", "stats11", "stats12", "stats13", "stats14",
           "stats15", "stats16", "stats17", "stats18", "stats19",
           "stats20", "stats21", "stats22", "stats23", "stats24",
           "stats25", "stats26", "stats27", "stats28", "stats29",
           "stats30", "stats31", "stats32", "stats33", "stats34",
           "stats35"]

filename = os.path.join(outputDir,'crossmatch.dat')
fid = open(filename,'w')

cnt = 0
#filenames = filenames[:500]
for ff, filename in enumerate(filenames):
    if np.mod(ff,100) == 0:
        print('Loading file %d/%d' % (ff, len(filenames)))
 
    filenameSplit = filename.split("/")
    catnum = filenameSplit[-1].replace(".dat","").replace(".h5","").split("_")[-1]

    if "h5" in filename:
        try:
            with h5py.File(filename, 'r') as f:
                name = f['names'].value
                filters = f['filters'].value
                stats = f['stats'].value
        except:
            continue
        data_tmp = Table(rows=stats, names=h5names)
        data_tmp['name'] = name
        data_tmp['filt'] = filters
    else:
        data_tmp = ascii.read(filename,names=names)
    if len(data_tmp) == 0: continue

    idx = np.where(data_tmp["sig"] >= opts.sig)[0]
    data_tmp = data_tmp[idx]
    if len(data_tmp) == 0: continue

    print('Analyzing %s... %d significant objects.' % (filename, len(data_tmp)))
    idxs = []
    idys = []
    for ii, (name, ra, dec, err, amaj, amin, phi) in enumerate(zip(cnames, ras, decs, errs, amajs, amins, phis)):
        #if np.mod(ii,10000) == 0:
        #    print('Checking object %d/%d' % (ii, len(cnames)))

        dist = angular_distance(ra, dec, np.array(data_tmp["ra"]),
                                np.array(data_tmp["dec"]))
        idx = np.where(dist <= amaj)[0]
        if len(idx) == 0: continue
 
        ellipse = patches.Ellipse((ra, dec), amaj, amin, angle=phi)
        for jj in idx:
            if not ellipse.contains_point((data_tmp["ra"][jj],data_tmp["dec"][jj])):
                continue
            idxs.append(jj)
            idys.append(ii)

            print("%s %s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f" % (filename, name, ra, dec, err, amaj, amin, phi, data_tmp["ra"][jj],data_tmp["dec"][jj]))
            print("%s %s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f" % (filename, name, ra, dec, err, amaj, amin, phi, data_tmp["ra"][jj],data_tmp["dec"][jj]), file=fid, flush=True)
fid.close()
