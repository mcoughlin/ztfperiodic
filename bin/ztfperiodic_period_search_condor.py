
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

if opts.doLongPeriod:
    long_period_flag = "--doLongPeriod"
else:
    long_period_flag = ""

if opts.doLightcurveStats:
    stats_flag = "--doLightcurveStats"
else:
    stats_flag = ""

dataDir = opts.dataDir
outputDir = opts.outputDir
batch_size = opts.batch_size

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

dir_path = os.path.dirname(os.path.realpath(__file__))

condordag = os.path.join(condorDir,'condor.dag')
fid = open(condordag,'w') 
condorsh = os.path.join(condorDir,'condor.sh')
fid1 = open(condorsh,'w') 

job_number = 0

if opts.lightcurve_source == "Kowalski":

    if opts.source_type == "quadrant":
        fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
        fields = [683]
        for field in fields:
            for ccd in ccds:
                for quadrant in quadrants:
                    for ii in range(opts.Ncatalog):
                        fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --field %d --ccd %d --quadrant %d --user %s --pwd %s --batch_size %d -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --doLightcurveStats %s %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, field, ccd, quadrant, opts.user, opts.pwd,opts.batch_size,opts.Ncatalog, ii, opts.algorithm,long_period_flag,stats_flag))
    
                        fid.write('JOB %d condor.sub\n'%(job_number))
                        fid.write('RETRY %d 3\n'%(job_number))
                        fid.write('VARS %d jobNumber="%d" field="%d" ccd="%d" quadrant="%d" Ncatindex="%d"\n'%(job_number,job_number,field, ccd, quadrant, ii))
                        fid.write('\n\n')
                        job_number = job_number + 1
    elif opts.source_type == "catalog":
        for ii in range(opts.Ncatalog):
            fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --user %s --pwd %s --batch_size %d -l Kowalski --source_type catalog --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 10.0 --program_ids 1,2,3 --catalog_file %s --doPlots --Ncatalog %d --Ncatindex %d %s %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, opts.user, opts.pwd,opts.batch_size, opts.algorithm, opts.catalog_file,opts.Ncatalog,ii,long_period_flag,stats_flag))

            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" Ncatindex="%d"\n'%(job_number,job_number, ii))
            fid.write('\n\n')
            job_number = job_number + 1

elif opts.lightcurve_source == "matchfiles":
    directory="%s/*/*/*"%opts.dataDir
    for f in glob.iglob(directory):
        fid1.write('%s %s/ztfperiodic_period_search.py %s --outputDir %s --matchFile %s -l matchfiles --doSaveMemory --doRemoveTerrestrial --doRemoveBrightStars --doLightcurveStats --algorithm %s %s %s\n'%(opts.python, dir_path, cpu_gpu_flag, outputDir, f, opts.algorithm,long_period_flag,stats_flag))

        fid.write('JOB %d condor.sub\n'%(job_number))
        fid.write('RETRY %d 3\n'%(job_number))
        fid.write('VARS %d jobNumber="%d" matchFile="%s"\n'%(job_number,job_number,f))
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
        fid.write('arguments = %s --outputDir %s --batch_size %d --field $(field) --ccd $(ccd) --quadrant $(quadrant) --Ncatalog %d --Ncatindex $(Ncatindex) --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --doRemoveBrightStars --program_ids 1,2,3 --doPlots --doLightcurveStats --algorithm %s %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,opts.Ncatalog,opts.user,opts.pwd,opts.algorithm,long_period_flag,stats_flag))
    elif opts.source_type == "catalog":
        fid.write('arguments = %s --outputDir %s --batch_size %d --user %s --pwd %s -l Kowalski --doSaveMemory --doRemoveTerrestrial --source_type catalog --catalog_file %s --doRemoveBrightStars --stardist 10.0 --program_ids 1,2,3 --doPlots --Ncatalog %d --Ncatindex $(Ncatindex) --algorithm %s %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,opts.user,opts.pwd,opts.catalog_file,opts.Ncatalog,opts.algorithm,long_period_flag,stats_flag))
else:
    fid.write('arguments = %s --outputDir %s --batch_size %d --matchFile $(matchFile) -l matchfiles --doSaveMemory --doRemoveTerrestrial --doRemoveBrightStars --doLightcurveStats --algorithm %s %s %s\n'%(cpu_gpu_flag,outputDir,batch_size,opts.algorithm,long_period_flag,stats_flag))
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 8192\n');
if opts.doCPU:
    fid.write('request_cpus = 1\n');
else:
    fid.write('request_gpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /usr1/mcoughlin/folding.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

