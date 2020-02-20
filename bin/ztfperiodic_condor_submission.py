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
    parser.add_option("-g","--gpudev",default=0,type=int)
    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def filter_completed(quad_out, catalogDir):

    njobs, ncols = quad_out.shape
    tbd = []
    for ii, row in enumerate(quad_out):
        field, ccd, quadrant = row[0], row[1], row[2]
        Ncatindex, Ncatalog = row[3], row[4]
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            tbd.append(ii)
    quad_out = quad_out[tbd,:]
    return quad_out

def load_condor(condorfile):

    lines = [line.rstrip('\n') for line in open(condorfile)]    
    rows = []
    for line in lines:
        lineSplit = line.replace('"',"").split(" ")
        if lineSplit[0] == "VARS":
            field = int(lineSplit[3].split("=")[-1])
            ccd = int(lineSplit[4].split("=")[-1])
            quadrant = int(lineSplit[5].split("=")[-1])
            Ncatindex = int(lineSplit[6].split("=")[-1])
            Ncatalog = int(lineSplit[7].split("=")[-1])
            rows.append([field,ccd,quadrant,Ncatindex,Ncatalog])
    return np.array(rows)

# Parse command line
opts = parse_commandline()

dir_path = os.path.dirname(os.path.realpath(__file__))

outputDir = opts.outputDir

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

condorfile = os.path.join(condorDir,'condor.sub')
lines = [line.rstrip('\n') for line in open(condorfile)]
executable = lines[0].split(" ")[-1]
arguments = lines[3].replace("arguments = ","")
jobline = "CUDA_VISIBLE_DEVICES=%d %s %s %s" %(opts.gpudev, opts.python, executable, arguments)

argumentsSplit = list(filter(None,arguments.split("algorithm")[-1].split(" ")))
algorithm = argumentsSplit[0]

condorfile = os.path.join(condorDir,'condor.dag')
catalogDir = os.path.join(outputDir,'catalog',algorithm)

quad_out_original = load_condor(condorfile)
quad_out = filter_completed(quad_out_original, catalogDir)       
njobs, ncols = quad_out.shape
print('%d jobs remaining...' % njobs)

if opts.doSubmit:
    while njobs > 0:
        quadrant_index = np.random.randint(0, njobs, size=1)
        idx = np.where(quad_out_original[:,0] == quad_out[quadrant_index,0])[0]
        row = quad_out_original[idx,:][0]
        field, ccd, quadrant = row[0], row[1], row[2]
        Ncatindex, Ncatalog = row[3], row[4]

        jobstr = jobline.replace("$(field)","%d" % field).replace("$(ccd)", "%d" % ccd).replace("$(quadrant)", "%d" % quadrant).replace("$(Ncatalog)", "%d" % Ncatalog).replace("$(Ncatindex)", "%d" % Ncatindex)
        os.system(jobstr)

        quad_out = filter_completed(quad_out, catalogDir)
        njobs, ncols = quad_out.shape
        print('%d jobs remaining...' % njobs)
