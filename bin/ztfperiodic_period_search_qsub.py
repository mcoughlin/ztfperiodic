
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

    parser.add_option("--doLongPeriod",  action="store_true", default=False)
    parser.add_option("--doCombineFilt",  action="store_true", default=False)
    parser.add_option("--doRemoveHC",  action="store_true", default=False)
    parser.add_option("--doHCOnly",  action="store_true", default=False)
    parser.add_option("--doUsePDot",  action="store_true", default=False)
    parser.add_option("--doSpectra",  action="store_true", default=False)
    parser.add_option("--doQuadrantScale",  action="store_true", default=False)

    parser.add_option("--doVariability",  action="store_true", default=False)
    parser.add_option("--doCutNObs",  action="store_true", default=False)
    parser.add_option("-n","--NObs",default=500,type=int)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=1000.0,type=int)

    parser.add_option("--doCrossMatch",  action="store_true", default=False)

    parser.add_option("--qid",default=None,type=int)
    parser.add_option("--fid",default=None,type=int)

    parser.add_option("--doDocker",  action="store_true", default=False)
    parser.add_option("--doRsyncFiles",  action="store_true", default=False)

    parser.add_option("--doPercentile",  action="store_true", default=False)
    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")
    parser.add_option("-e","--email")

    parser.add_option("-f","--filetype",default="slurm")
    parser.add_option("--queue_type",default="v100")

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
if opts.doRsyncFiles:
    extra_flags.append("--doRsyncFiles")
if opts.doCrossMatch:
    extra_flags.append("--doCrossMatch")
if opts.doPercentile:
    extra_flags.append("--doPercentile")
if opts.doParallel:
    extra_flags.append("--doParallel")
if opts.doPlots:
    extra_flags.append("--doPlots")
extra_flags = " ".join(extra_flags)

matchfileDir = opts.matchfileDir
outputDir = opts.outputDir
batch_size = opts.batch_size
Ncatalog = opts.Ncatalog
algorithm = opts.algorithm

catalogDir = os.path.join(outputDir,'catalog',algorithm)

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)

idsDir = os.path.join(qsubDir,'ids')
if not os.path.isdir(idsDir):
    os.makedirs(idsDir)

if opts.doCutNObs:
    nobsfile = "../input/nobs.dat"
    nobs_data = np.loadtxt(nobsfile)

    if opts.doHCOnly:
        idx = np.where((nobs_data[:,7] >= opts.NObs) | (nobs_data[:,8] >= opts.NObs) | (nobs_data[:,9] >= opts.NObs))[0]
    else:
        idx = np.where((nobs_data[:,4] >= opts.NObs) | (nobs_data[:,5] >= opts.NObs) | (nobs_data[:,6] >= opts.NObs))[0]

    nobs_data = nobs_data[idx,:]
    fields_list = nobs_data[:,0]

if opts.doQuadrantScale:
    kow = Kowalski(username=opts.user, password=opts.pwd)

if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
        #ccds, quadrants = [1], [1]

        fields1 = [683,853,487,718,372,842,359,778,699,296]
        fields2 = [841,852,682,717,488,423,424,563,562,297,700,777]
        fields3 = [851,848,797,761,721,508,352,355,364,379]
        fields4 = [1866,1834,1835,1804,1734,1655,1565]

        fields_complete = fields1 + fields2 + fields3 + fields4
        fields = np.arange(1600,1700)
        fields = np.setdiff1d(fields,fields_complete)

        fields = [400]
        fields = np.arange(250,882)
        fields = np.arange(600,700)
        fields = np.arange(750,800)
        #fields = np.arange(300,305)

        job_number = 0
        quadrantfile = os.path.join(qsubDir,'qsub.dat')
        fid = open(quadrantfile,'w')
        for field in fields:
            if opts.doCutNObs:
                if not field in fields_list:
                    continue
            print('Running field %d' % field)
            for ccd in ccds:
                for quadrant in quadrants:
                    if opts.doQuadrantScale:
                        qu = {"query_type":"count_documents",
                              "query": {
                                  "catalog": 'ZTF_sources_20200401',
                                  "filter": {'field': {'$eq': int(field)},
                                             'ccd': {'$eq': int(ccd)},
                                             'quad': {'$eq': int(quadrant)}
                                             }
                                       } 
                             }                                            
                        r = ztfperiodic.utils.database_query(kow, qu, nquery = 10)
                        if not "data" in r: continue
                        nlightcurves = r['data']

                        Ncatalog = int(np.ceil(float(nlightcurves)/opts.Nmax))

                        idsFile = os.path.join(idsDir,"%d_%d_%d.npy"%(field, ccd, quadrant))
                        if not os.path.isfile(idsFile):
                            print(idsFile)
                            qu = {"query_type":"find",
                                  "query": {"catalog": 'ZTF_sources_20200401',
                                            "filter": {'field': {'$eq': int(field)},
                                                       'ccd': {'$eq': int(ccd)},
                                                       'quad': {'$eq': int(quadrant)}
                                                      },
                                            "projection": "{'_id': 1}"},
                                 }
                            r = ztfperiodic.utils.database_query(kow, qu, nquery = 10)
                            objids = []
                            for obj in r['data']:
                                objids.append(obj['_id'])
                            np.save(idsFile, objids)
                    
                    for ii in range(Ncatalog):
                        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant, ii))
                        if os.path.isfile(catalogFile):
                            print('%s already exists... continuing.' % catalogFile)
                            continue

                        fid.write('%d %d %d %d %d %d %s\n' % (job_number, field, ccd, quadrant, ii, Ncatalog, idsFile))
    
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
fid.write('#PBS -l walltime=23:59:59,nodes=1:ppn=24:gpus=1,pmem=5290mb -q k40\n')
fid.write('#PBS -m abe\n')
fid.write('#PBS -M cough052@umn.edu\n')
fid.write('source /home/cough052/cough052/ZTF/ztfperiodic/setup.sh\n')
fid.write('cd $PBS_O_WORKDIR\n')
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type objid --doQuadrantFile --quadrant_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --quadrant_index $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,quadrantfile,algorithm,extra_flags))
    elif opts.source_type == "catalog":
        fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type catalog --catalog_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --Ncatalog %d --Ncatindex $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.catalog_file,opts.Ncatalog,algorithm,extra_flags))
elif opts.lightcurve_source == "matchfiles":
    fid.write('%s/ztfperiodic_period_search.py %s --outputDir %s --batch_size %d -l matchfiles --doRemoveTerrestrial --doQuadrantFile --quadrant_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --Ncatalog %d --quadrant_index $PBS_ARRAYID --algorithm %s %s\n'%(dir_path,cpu_gpu_flag,outputDir,batch_size,quadrantfile,opts.Ncatalog,algorithm,extra_flags))
fid.close()

fid = open(os.path.join(qsubDir,'qsub_submission.sub'),'w')
fid.write('#!/bin/bash\n')
if opts.queue_type == "v100":
    fid.write('#PBS -l walltime=23:59:59,nodes=1:ppn=24:gpus=1,pmem=3000mb -q v100\n')
elif opts.queue_type == "k40":
    fid.write('#PBS -l walltime=23:59:59,nodes=1:ppn=24:gpus=1,pmem=5290mb -q k40\n')
else:
    print('queue_type must be v100 or k40')
    exit(0)
fid.write('#PBS -m abe\n')
fid.write('#PBS -M cough052@umn.edu\n')
fid.write('source /home/cough052/cough052/ZTF/ztfperiodic/setup.sh\n')
fid.write('cd $PBS_O_WORKDIR\n')
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('%s/ztfperiodic_job_submission.py --outputDir %s --filetype %s --doSubmit\n' % (dir_path, outputDir, opts.filetype))
fid.close()
