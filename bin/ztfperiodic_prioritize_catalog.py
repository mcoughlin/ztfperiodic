
import os, sys
import glob
import optparse
import copy

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.io import ascii
from astropy.table import Table, Column

import pandas as pd
import numpy as np

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="/Users/mcoughlin/Desktop/Fermi/priority/")
    parser.add_option("-d","--dataDir",default="/Users/mcoughlin/Desktop/Fermi/catalog/GCE/")

    parser.add_option("--catalog_file",default="../catalogs/fermi.dat")

    parser.add_option("--sig",default=7.0,type=float)

    parser.add_option("--tmin",default=4.0*60.0/86400.0,type=float)
    parser.add_option("--tmax",default=10.0,type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
catalog_file = opts.catalog_file
dataDir = opts.dataDir
outputDir = opts.outputDir
tmin, tmax = opts.tmin, opts.tmax

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

names = ["name", "objid", "ra", "dec", "period", "sig", "pdot", "filt",
         "stats0", "stats1", "stats2", "stats3", "stats4",
         "stats5", "stats6", "stats7", "stats8", "stats9",
         "stats10", "stats11", "stats12", "stats13", "stats14",
         "stats15", "stats16", "stats17", "stats18", "stats19",
         "stats20", "stats21", "stats22", "stats23", "stats24",
         "stats25", "stats26", "stats27", "stats28", "stats29",
         "stats30", "stats31", "stats32", "stats33", "stats34",
         "stats35"]

nums, bs, nrows, indexes = [], [], [], []

lines = [line.rstrip('\n') for line in open(catalog_file)]
for ii, line in enumerate(lines):
    lineSplit = list(filter(None,line.split(" ")))
    name = lineSplit[0]
    ra, dec = float(lineSplit[1]), float(lineSplit[2])
    index = float(lineSplit[8])   
 
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    gal = coord.galactic
    b = gal.b.deg

    if np.abs(b) > 15: continue

    data = Table(names=names)
    filenames = sorted(glob.glob(os.path.join(dataDir,"%d/*.dat"%ii)))[::-1]
    cnt = 0
    for jj, filename in enumerate(filenames):
        data_tmp = ascii.read(filename,names=names)
        if len(data_tmp) == 0: continue

        data_tmp['name'] = data_tmp['name'].astype(str)
        data_tmp['filt'] = data_tmp['filt'].astype(str)

        if cnt == 0:
            data = copy.copy(data_tmp)
        else:
            data = vstack([data,data_tmp])
        cnt = cnt + 1

    idx = np.where(data["sig"] >= opts.sig)[0]
    data = data[idx]

    idx = np.where((data["period"] >= tmin) & (data["period"] <= tmax))[0]
    data = data[idx]

    if len(data) == 0: continue

    nums.append(ii)
    bs.append(b)
    nrows.append(len(data))
    indexes.append(index)

nums, bs, nrows, indexes = np.array(nums), np.array(bs), np.array(nrows), np.array(indexes)
idx = np.argsort(nrows)
nums, bs, nrows, indexes = nums[idx], bs[idx], nrows[idx], indexes[idx]

filename = os.path.join(outputDir,'priority.dat')
fid = open(filename, 'w')
for num, b, nrow, index in zip(nums, bs, nrows, indexes):
    fid.write('%d %.5f %d %.5f\n' % (num, b, nrow, index))
fid.close()
