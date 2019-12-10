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
    parser.add_option("-a","--algorithm",default="CE")
    parser.add_option("-l","--max_jobs",default=1000,type=int)

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def filter_completed(quad_out, catalogDir):

    njobs, ncols = quad_out.shape
    tbd = []
    for ii, row in enumerate(quad_out):
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex = row[4]
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            tbd.append(ii)
    quad_out = quad_out[tbd,:]
    return quad_out

# Parse command line
opts = parse_commandline()

dir_path = os.path.dirname(os.path.realpath(__file__))

algorithm = opts.algorithm
outputDir = opts.outputDir
max_jobs = opts.max_jobs

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)
catalogDir = os.path.join(outputDir,'catalog',algorithm)
qsubfile = os.path.join(qsubDir,'qsub.sub')
lines = [line.rstrip('\n') for line in open(qsubfile)]
jobline = lines[-1]

quadrantfile = os.path.join(qsubDir,'qsub.dat')

catalogDir = os.path.join(outputDir,'catalog',algorithm)
quad_out_original = np.loadtxt(quadrantfile)
quad_out = filter_completed(quad_out_original, catalogDir)       
njobs, ncols = quad_out.shape

if opts.doSubmit:
    while njobs > 0:
        quadrant_index = np.random.randint(0, njobs, size=1)
        idx = np.where(quad_out_original[:,0] == quad_out[quadrant_index,0])[0]
        row = quad_out_original[idx,:][0]
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex = row[4]

        jobstr = jobline.replace("$PBS_ARRAYID","%d"%row[0])
        os.system(jobstr)

        quad_out = filter_completed(quad_out, catalogDir)
        njobs, ncols = quad_out.shape
