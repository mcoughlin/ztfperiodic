#!/usr/bin/env python

import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

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

    parser.add_option("-p","--python",default="python")
    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def filter_completed(quad_out, catalogDir):

    njobs, ncols = quad_out.shape
    tbd = []
    for ii, row in enumerate(quad_out):
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex, Ncatalog = row[4], row[5]
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            tbd.append(ii)
    quad_out = quad_out[tbd,:]
    return quad_out

# Parse command line
opts = parse_commandline()

dir_path = os.path.dirname(os.path.realpath(__file__))

outputDir = opts.outputDir

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)

qsubfile = os.path.join(qsubDir,'qsub.sub')
lines = [line.rstrip('\n') for line in open(qsubfile)]
jobline = "%s --doCheckLightcurves" % lines[-1]
joblineSplit = list(filter(None,jobline.split("algorithm")[-1].split(" ")))
algorithm = joblineSplit[0]

quadrantfile = os.path.join(qsubDir,'qsub.dat')

catalogDir = os.path.join(outputDir,'catalog',algorithm)
quad_out = np.loadtxt(quadrantfile)
for ii, row in enumerate(quad_out):
    field, ccd, quadrant = row[1], row[2], row[3]
    Ncatindex, Ncatalog = row[4], row[5]
    catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
    if not os.path.isfile(catalogFile):
        jobstr = jobline.replace("$PBS_ARRAYID","%d"%row[0])
        os.system(jobstr)
