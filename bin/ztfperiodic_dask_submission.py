#!/usr/bin/env python

import os, sys
import time
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

import dask
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, progress

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
    parser.add_option("-c","--CUDA_VISIBLE_DEVICES",default="0,1,2,3,4,5,6,7")

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def filter_completed(df, catalogDir):

    start_time = time.time()

    tbd = []
    for ii, (index, row) in enumerate(df.iterrows()):
        field, ccd, quadrant = row["field"], row["ccd"], row["quadrant"]
        Ncatindex, Ncatalog = row["Ncatindex"], row["Ncatalog"]
        idsFile = row["idsFile"]
        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
        if not os.path.isfile(catalogFile):
            tbd.append(ii)
    df = df.iloc[tbd]

    end_time = time.time()
    print('Checking completed jobs took %.2f seconds' % (end_time - start_time))

    return df

def run_job(row):

    field, ccd, quadrant = row["field"], row["ccd"], row["quadrant"]
    Ncatindex, Ncatalog = row["Ncatindex"], row["Ncatalog"]
    idsFile = row["idsFile"]

    print(field, ccd, quadrant, Ncatindex, Ncatalog)
    catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant,Ncatindex))
    if not os.path.isfile(catalogFile):
        jobstr = jobline.replace("$PBS_ARRAYID","%d"%row["job_number"])
        print(jobstr)
        os.system(jobstr)

if __name__ == '__main__':

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
    
    names = ["job_number", "field", "ccd", "quadrant",
             "Ncatindex", "Ncatalog", "idsFile"]
    
    catalogDir = os.path.join(outputDir,'catalog',algorithm)
    #quad_out_original = np.loadtxt(quadrantfile)
    df_original = pd.read_csv(quadrantfile, header=0, delimiter=' ',
                              names=names)
    df = filter_completed(df_original, catalogDir)       
    njobs = len(df)
    print('%d jobs remaining...' % njobs)
    
    counter = 0

    #client = Client(opts.scheduler)
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=opts.CUDA_VISIBLE_DEVICES,
                               threads_per_worker=1)

    client = Client(cluster)
    if opts.doSubmit:
        lazy_results = []
        for ii in range(njobs):
            row = df.iloc[ii]
            lazy_result = dask.delayed(run_job)(row)
            lazy_results.append(lazy_result)

        futures = dask.persist(*lazy_results)  # trigger computation in the background
        #dask.compute(*lazy_results)
        progress(futures)
