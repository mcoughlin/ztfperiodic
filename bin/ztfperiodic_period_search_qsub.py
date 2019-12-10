
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

    parser.add_option("--doGPU",  action="store_true", default=False)
    parser.add_option("--doCPU",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")
    parser.add_option("--matchfileDir",default="/home/mcoughlin/ZTF/matchfiles/")
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-a","--algorithm",default="CE")

    parser.add_option("--doLightcurveStats",  action="store_true", default=False)
    parser.add_option("--doLongPeriod",  action="store_true", default=False)
    parser.add_option("--doCombineFilt",  action="store_true", default=False)
    parser.add_option("--doRemoveHC",  action="store_true", default=False)
    parser.add_option("--doHCOnly",  action="store_true", default=False)
    parser.add_option("--doUsePDot",  action="store_true", default=False)
    parser.add_option("--doSpectra",  action="store_true", default=False)
    parser.add_option("--doQuadrantScale",  action="store_true", default=False)

    parser.add_option("--doVariability",  action="store_true", default=False)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=10000.0,type=int)

    parser.add_option("--qid",default=None,type=int)
    parser.add_option("--fid",default=None,type=int)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")
    parser.add_option("-e","--email")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

dir_path = os.path.dirname(os.path.realpath(__file__))

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
if opts.doHCOnly:
    extra_flags.append("--doHCOnly")
if opts.doUsePDot:
    extra_flags.append("--doUsePDot")
if opts.doSpectra:
    extra_flags.append("--doSpectra")
if opts.doVariability:
    extra_flags.append("--doVariability")
    extra_flags.append("--doNotPeriodFind")
extra_flags = " ".join(extra_flags)

matchfileDir = opts.matchfileDir
outputDir = opts.outputDir
batch_size = opts.batch_size
Ncatalog = opts.Ncatalog

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)

if opts.doQuadrantScale:
    kow = Kowalski(username=opts.user, password=opts.pwd)

if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
        #ccds, quadrants = [1], [1]

        #fields = [683,853,487,718,372,842,359,778,699,296]
        fields = [718]
        job_number = 0
        quadrantfile = os.path.join(qsubDir,'qsub.dat')
        fid = open(quadrantfile,'w')
        for field in fields:
            for ccd in ccds:
                for quadrant in quadrants:
                    if opts.doQuadrantScale:
                        qu = {"query_type":"count_documents",
                              "query": {
                                  "catalog": 'ZTF_sources_20191101',
                                  "filter": {'field': {'$eq': int(field)},
                                             'ccd': {'$eq': int(ccd)},
                                             'quad': {'$eq': int(quadrant)}
                                             }
                                       } 
                             }                                            
                        r = ztfperiodic.utils.database_query(kow, qu, nquery = 1)
                        if not "result_data" in r: continue
                        nlightcurves = r['result_data']['query_result']

                        Ncatalog = int(np.ceil(float(nlightcurves)/opts.Nmax))
                    for ii in range(Ncatalog):
                        fid.write('%d %d %d %d %d %d\n' % (job_number, field, ccd, quadrant, ii, Ncatalog))
    
                        job_number = job_number + 1
        fid.close()
elif opts.lightcurve_source == "matchfiles":
    bands = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'J'}
    directory="%s/*/*/*.pytable"%opts.matchfileDir
    job_number = 0
    quadrantfile = os.path.join(qsubDir,'qsub.dat')
    fid = open(quadrantfile,'w')
    for f in glob.iglob(directory):
        if not opts.qid is None:
            if not ("rc%02d"%opts.qid) in f:
                continue
        if not opts.fid is None:
            if not ("z%s"%bands[opts.fid]) in f:
                continue
        for ii in range(Ncatalog):    
            fid.write('%d %s %d\n' % (job_number, f, ii))
 
            job_number = job_number + 1
    fid.close()

fid = open(os.path.join(qsubDir,'qsub.sub'),'w')
fid.write('#!/bin/bash\n')
fid.write('#PBS -l walltime=1:00:00,nodes=1:ppn=24:gpus=1,pmem=5290mb -q k40\n')
fid.write('#PBS -m abe\n')
fid.write('#PBS -M cough052@umn.edu\n')
fid.write('source /home/cough052/cough052/ZTF/ztfperiodic/setup.sh\n')
fid.write('cd $PBS_O_WORKDIR\n')
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type quadrant --doQuadrantFile --quadrant_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --quadrant_index $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,quadrantfile,opts.Ncatalog,opts.algorithm,extra_flags))
    elif opts.source_type == "catalog":
        fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type catalog --catalog_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --Ncatindex $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.catalog_file,opts.Ncatalog,opts.algorithm,extra_flags))
elif opts.lightcurve_source == "matchfiles":
    fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d -l matchfiles --doRemoveTerrestrial --doQuadrantFile --quadrant_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --quadrant_index $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,quadrantfile,opts.Ncatalog,opts.algorithm,extra_flags))
fid.close()

fid = open(os.path.join(qsubDir,'qsub_submission.sub'),'w')
fid.write('#!/bin/bash\n')
fid.write('#PBS -l walltime=1:00:00,nodes=1:ppn=24:gpus=1,pmem=5290mb -q k40\n')
fid.write('#PBS -m abe\n')
fid.write('#PBS -M cough052@umn.edu\n')
fid.write('source /home/cough052/cough052/ZTF/ztfperiodic/setup.sh\n')
fid.write('cd $PBS_O_WORKDIR\n')
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('%s/ztfperiodic_job_submission.py --outputDir %s -a %s --doSubmit\n' % (dir_path, outputDir, opts.algorithm))
fid.close()
