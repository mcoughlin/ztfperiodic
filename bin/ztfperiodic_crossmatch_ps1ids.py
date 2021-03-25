
import os, sys
import glob
import optparse
import subprocess

import tables
import pandas as pd
import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches

import ztfperiodic.utils

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output_Fermi_forced_ML")

    parser.add_option("-s","--source_type",default="ps1")
    parser.add_option("--catalog_file",default="../catalogs/fermi_ML.dat")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()

# Parse command line
opts = parse_commandline()
outputDir = opts.outputDir

dir_path = os.path.dirname(os.path.realpath(__file__))
outputDir = opts.outputDir

idsDir = os.path.join(outputDir,'ids')
if not os.path.isdir(idsDir):
    os.makedirs(idsDir)

kow = Kowalski(username=opts.user, password=opts.pwd, timeout=120)

lines = [line.rstrip('\n') for line in open(opts.catalog_file)]
names, ras, decs, errs = [], [], [], []
if ("fermi" in opts.catalog_file):
    amaj, amin, phi = [], [], []
for line in lines:
    lineSplit = list(filter(None,line.split(" ")))
    if ("fermi" in opts.catalog_file):
        names.append(lineSplit[0])
        ras.append(float(lineSplit[1]))
        decs.append(float(lineSplit[2]))
        err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
        errs.append(err)
        amaj.append(float(lineSplit[3]))
        amin.append(float(lineSplit[4]))
        phi.append(float(lineSplit[5]))

for ii in range(len(names)):

    idsFile = os.path.join(idsDir,"%d.hdf5"%(ii))
    if not os.path.isfile(idsFile):

        ra, dec, radius = ras[ii], decs[ii], errs[ii]

        if dec < -30: continue
        if np.isnan(radius): continue
        print(ii, ra, dec, radius, amaj[ii], amin[ii])

        ellipse = patches.Ellipse((ra, dec), amaj[ii], amin[ii],
                                  angle=phi[ii])

        qu = { "query_type": "cone_search",
               "query": {"object_coordinates": {"radec": {'test': [ra,dec]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" },
                         "catalogs": { "PS1_DR1": { "filter": "{}", "projection": "{'_id': 1, 'raMean': 1, 'decMean': 1}"}}}}

        try:
            r = ztfperiodic.utils.database_query(kow, qu, nquery = 10)
        except:
            continue

        objids_fermi = []
        ras_fermi, decs_fermi = [], []
        r = r["data"]["PS1_DR1"]
        for obj in r['test']:
            if not ellipse.contains_point((obj['raMean'], obj['decMean'])):
                continue

            objids_fermi.append(obj['_id'])
            ras_fermi.append(obj['raMean'])
            decs_fermi.append(obj['decMean'])
        objids_fermi = np.array(objids_fermi)
        ras_fermi, decs_fermi = np.array(ras_fermi), np.array(decs_fermi)

        with h5py.File(idsFile, 'w') as hf:
            hf.create_dataset("objid",  data=objids_fermi)
            hf.create_dataset("ra",  data=ras_fermi)
            hf.create_dataset("dec",  data=decs_fermi)
