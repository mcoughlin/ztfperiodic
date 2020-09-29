#!/usr/bin/env python

import os, sys
import time
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
    parser.add_option("-f","--filetype",default="slurm")

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def filter_completed(quad_out, catalogDir):

    start_time = time.time()

    njobs, ncols = quad_out.shape
    tbd = []
    for ii, row in enumerate(quad_out):
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex, Ncatalog = row[4], row[5]
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            tbd.append(ii)
    quad_out = quad_out[tbd,:]

    end_time = time.time()
    print('Checking completed jobs took %.2f seconds' % (end_time - start_time))

    return quad_out

# Parse command line
opts = parse_commandline()

dir_path = os.path.dirname(os.path.realpath(__file__))

outputDir = opts.outputDir
filetype = opts.filetype

qsubDir = os.path.join(outputDir,filetype)
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)
qsubfile = os.path.join(qsubDir,'%s.sub' % filetype)

lines = [line.rstrip('\n') for line in open(qsubfile)]
jobline = lines[-1]
joblineSplit = list(filter(None,jobline.split("algorithm")[-1].split(" ")))
algorithm = joblineSplit[0]

quadrantfile = os.path.join(qsubDir,'%s.dat' % filetype)

catalogDir = os.path.join(outputDir,'catalog',algorithm)
quad_out_original = np.loadtxt(quadrantfile)
quad_out = filter_completed(quad_out_original, catalogDir)       
njobs, ncols = quad_out.shape
print('%d jobs remaining...' % njobs)

counter = 0

if opts.doSubmit:
    while njobs > 0:
        quadrant_index = np.random.randint(0, njobs, size=1)
        idx = np.where(quad_out_original[:,0] == quad_out[quadrant_index,0])[0]
        row = quad_out_original[idx,:][0]
        field, ccd, quadrant = row[1], row[2], row[3]
        Ncatindex, Ncatalog = row[4], row[5]

        print(field, ccd, quadrant, Ncatindex, Ncatalog)
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            jobstr = jobline.replace("$PBS_ARRAYID","%d"%row[0])
            os.system(jobstr)

        counter = counter + 1
        print(counter)

        if np.mod(counter,500) == 0:
            quad_out = filter_completed(quad_out, catalogDir)
            njobs, ncols = quad_out.shape
            print('%d jobs remaining...' % njobs)
