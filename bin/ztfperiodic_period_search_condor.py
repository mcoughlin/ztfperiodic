
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

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=1000.0,type=int)

    parser.add_option("--doVariability",  action="store_true", default=False)
    parser.add_option("--doCutNObs",  action="store_true", default=False)
    parser.add_option("-n","--NObs",default=500,type=int)

    parser.add_option("--doUseMatchfileFile",  action="store_true", default=False)

    parser.add_option("--qid",default=None,type=int)
    parser.add_option("--fid",default=None,type=int)

    parser.add_option("--doDocker",  action="store_true", default=False)
    parser.add_option("--doRsyncFiles",  action="store_true", default=False)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
Ncatalog = opts.Ncatalog

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
extra_flags = " ".join(extra_flags)

matchfileDir = opts.matchfileDir
outputDir = opts.outputDir
batch_size = opts.batch_size
algorithm = opts.algorithm

catalogDir = os.path.join(outputDir,'catalog',algorithm)

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

if opts.doCutNObs:
    nobsfile = "../input/nobs.dat"
    nobs_data = np.loadtxt(nobsfile)
    idx = np.where((nobs_data[:,4] >= opts.NObs) | (nobs_data[:,5] >= opts.NObs))[0]
    nobs_data = nobs_data[idx,:]
    fields_list = nobs_data[:,0]

dir_path = os.path.dirname(os.path.realpath(__file__))

condordag = os.path.join(condorDir,'condor.dag')
fid = open(condordag,'w') 
condorsh = os.path.join(condorDir,'condor.sh')
fid1 = open(condorsh,'w') 

job_number = 0

if opts.doQuadrantScale:
    kow = Kowalski(username=opts.user, password=opts.pwd)

if opts.lightcurve_source == "Kowalski":

    if opts.source_type == "quadrant":
        fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
        fields1 = [683,853,487,718,372,842,359,778,699,296]
        fields2 = [841,852,682,717,488,423,424,563,562,297,700,777]
        fields3 = [851,848,797,761,721,508,352,355,364,379]
        fields4 = [1866,1834,1835,1804,1734,1655,1565]

        fields_complete = fields1 + fields2 + fields3 + fields4
        fields = np.arange(300,400)
        fields = np.setdiff1d(fields,fields_complete)

        for field in fields:
            print('Running field %d' % field)
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
                        catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant, ii))
                        if os.path.isfile(catalogFile):
                            print('%s already exists... continuing.' % catalogFile)
                            continue

                        if opts.doDocker:
                            fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic %s --outputDir %s --program_ids 1,2,3 --field %d --ccd %d --quadrant %d --user %s --pwd %s --batch_size %d -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --doRemoveTerrestrial --doPlots --doRemoveBrightStars --doLightcurveStats %s\n'%(cpu_gpu_flag, outputDir, field, ccd, quadrant, opts.user, opts.pwd,opts.batch_size, Ncatalog, ii, opts.algorithm, extra_flags))
                        else:
                            fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --program_ids 1,2,3 --field %d --ccd %d --quadrant %d --user %s --pwd %s --batch_size %d -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --doRemoveTerrestrial --doPlots --doRemoveBrightStars --doLightcurveStats %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, field, ccd, quadrant, opts.user, opts.pwd,opts.batch_size, Ncatalog, ii, opts.algorithm, extra_flags))
    
                        fid.write('JOB %d condor.sub\n'%(job_number))
                        fid.write('RETRY %d 3\n'%(job_number))
                        fid.write('VARS %d jobNumber="%d" field="%d" ccd="%d" quadrant="%d" Ncatindex="%d" Ncatalog="%d"\n'%(job_number,job_number,field, ccd, quadrant, ii, Ncatalog))
                        fid.write('\n\n')
                        job_number = job_number + 1

    elif opts.source_type == "catalog":
        for ii in range(Ncatalog):
            if opts.doDocker:
                fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic %s --outputDir %s --user %s --pwd %s --batch_size %d -l Kowalski --source_type catalog --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --catalog_file %s --doLightcurveStats --doPlots --Ncatalog %d --Ncatindex %d %s\n'%(cpu_gpu_flag, outputDir, opts.user, opts.pwd,opts.batch_size, opts.algorithm, opts.catalog_file,opts.Ncatalog,ii,extra_flags))
            else:
                fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --user %s --pwd %s --batch_size %d -l Kowalski --source_type catalog --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --catalog_file %s --doLightcurveStats --doPlots --Ncatalog %d --Ncatindex %d %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, opts.user, opts.pwd,opts.batch_size, opts.algorithm, opts.catalog_file,opts.Ncatalog,ii,extra_flags))

            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" Ncatindex="%d" Ncatalog="%d"\n'%(job_number,job_number, ii, Ncatalog))
            fid.write('\n\n')
            job_number = job_number + 1

elif opts.lightcurve_source == "matchfiles":
    bands = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'J'}
    if opts.doUseMatchfileFile:
        filename = "../input/matchfiles.txt"
        filenames = [line.rstrip('\n') for line in open(filename)]
    else:
        directory="%s/*/*/*.pytable"%opts.matchfileDir
        filenames = [f for f in glob.iglob(directory)]
    for f in filenames:
        field_id = int(f.split("/")[-1].split("_")[1])
        if opts.doCutNObs:
            if not field_id in fields_list:
                continue
        if not opts.qid is None:
            if not ("rc%02d"%opts.qid) in f:
                continue
        if not opts.fid is None:
            if not ("z%s"%bands[opts.fid]) in f:
                continue
        for ii in range(Ncatalog):
            if opts.doDocker:
                fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic %s --outputDir %s --matchFile %s -l matchfiles --source_type quadrants --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --doLightcurveStats --Ncatalog %d --Ncatindex %d --matchfileDir %s %s\n'%(cpu_gpu_flag, outputDir, f, opts.algorithm, opts.Ncatalog,ii, matchfileDir, extra_flags))
            else:
                fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --matchFile %s -l matchfiles --source_type quadrants --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --doLightcurveStats --Ncatalog %d --Ncatindex %d --matchfileDir %s %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, f, opts.algorithm, opts.Ncatalog,ii, matchfileDir, extra_flags))

            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" matchFile="%s" Ncatindex="%d" Ncatalog="%d"\n'%(job_number,job_number,f,ii,opts.Ncatalog))
            fid.write('\n\n')
            job_number = job_number + 1

fid1.close()
fid.close()

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = %s/ztfperiodic_period_search.py\n'%dir_path)
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('arguments = %s --outputDir %s --batch_size %d --field $(field) --ccd $(ccd) --quadrant $(quadrant) --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --doRemoveBrightStars --program_ids 1,2,3 --doPlots --doLightcurveStats --algorithm %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.algorithm,extra_flags))
    elif opts.source_type == "catalog":
        fid.write('arguments = %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type catalog --catalog_file %s --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --Ncatindex $(Ncatindex) --algorithm %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.catalog_file,opts.Ncatalog,opts.algorithm,extra_flags))
elif opts.lightcurve_source == "matchfiles":
    fid.write('arguments = %s --outputDir %s --batch_size %d --matchFile $(matchFile) -l matchfiles --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --doRemoveTerrestrial --doRemoveBrightStars --program_ids 1,2,3 --doPlots --doLightcurveStats --matchfileDir %s --algorithm %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,matchfileDir,opts.algorithm,extra_flags))
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 8192\n');
if opts.doCPU:
    fid.write('request_cpus = 1\n');
else:
    fid.write('request_gpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /local/michael.coughlin/folding.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

