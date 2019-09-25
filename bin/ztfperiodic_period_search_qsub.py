
import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-p","--python",default="python")

    parser.add_option("--doGPU",  action="store_true", default=False)
    parser.add_option("--doCPU",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")
    parser.add_option("-d","--dataDir",default="/home/mcoughlin/ZTF/Matchfiles")
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-a","--algorithm",default="CE")

    parser.add_option("--doLightcurveStats",  action="store_true", default=False)
    parser.add_option("--doLongPeriod",  action="store_true", default=False)
    parser.add_option("--doCombineFilt",  action="store_true", default=False)
    parser.add_option("--doRemoveHC",  action="store_true", default=False)
    parser.add_option("--doUsePDot",  action="store_true", default=False)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=1000,type=int)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

if opts.doCPU:
    cpu_gpu_flag = "--doCPU"
else:
    cpu_gpu_flag = "--doGPU"

extra_flags = []
if opts.doLongPeriod:
    extra_flags.append("--doLongPeriod")
if opts.doLightcurveStats:
    extra_flags.append("--doLightcurveStats")
if opts.doCombineFilt:
    extra_flags.append("--doCombineFilt")
if opts.doRemoveHC:
    extra_flags.append("--doRemoveHC")
if opts.doUsePDot:
    extra_flags.append("--doUsePDot")
extra_flags = " ".join(extra_flags)

dataDir = opts.dataDir
outputDir = opts.outputDir
batch_size = opts.batch_size

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)

logDir = os.path.join(qsubDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

if opts.source_type == "quadrant":
    fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
    fields = [683,853,487,718,372,842,359,778,699,296]
    job_number = 0
    quadrantfile = os.path.join(qsubDir,'qsub.dat')
    fid = open(quadrantfile,'w')
    for field in fields:
        for ccd in ccds:
            for quadrant in quadrants:
                for ii in range(opts.Ncatalog):
                    fid.write('%d %d %d %d %d\n' % (job_number, field, ccd, quadrant, ii))

                    job_number = job_number + 1
    fid.close()

dir_path = os.path.dirname(os.path.realpath(__file__))


fid = open(os.path.join(qsubDir,'qsub.sub'),'w')
fid.write('#!/bin/bash\n')
fid.write('#PBS -l walltime=1:00:00,nodes=1:ppn=8:gpus=1,pmem=1000mb -q k40\n')
fid.write('#PBS -m abe\n')
fid.write('#PBS -M cough052@umn.edu\n')
fid.write('source /home/cough052/cough052/ZTF/ztfperiodic/setup.sh\n')
fid.write('cd $PBS_O_WORKDIR\n')
if opts.source_type == "quadrant":
    fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type quadrant --doQuadrantFile --quadrant_file %s --doRemoveBrightStars --stardist 10.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --quadrant_index $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,quadrantfile,opts.Ncatalog,opts.algorithm,extra_flags))
elif opts.source_type == "catalog":
    fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type catalog --catalog_file %s --doRemoveBrightStars --stardist 10.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --Ncatindex $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.catalog_file,opts.Ncatalog,opts.algorithm,extra_flags))
fid.close()
