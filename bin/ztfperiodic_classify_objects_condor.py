
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
    parser.add_option("-b","--batch_size",default=1,type=int)
    parser.add_option("-a","--algorithm",default="xgboost")

    parser.add_option("--doQuadrantScale",  action="store_true", default=False)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=10000.0,type=int)

    parser.add_option("-m","--modelPath",default="/home/michael.coughlin/ZTF/ZTFVariability/pipeline/saved_models/")

    parser.add_option("--doDocker",  action="store_true", default=False)

    parser.add_option("-f","--featuresetname",default="b")
    parser.add_option("-d","--dbname",default="ZTF_source_features_20191101")
    parser.add_option("-q","--query_type",default="ids")
    parser.add_option("-i","--ids_file",default="/home/michael.coughlin/ZTF/ZTFVariability/ids/ids.20fields.npy")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
Ncatalog = opts.Ncatalog

outputDir = opts.outputDir
batch_size = opts.batch_size
algorithm = opts.algorithm
modelPath = opts.modelPath
featuresetname = opts.featuresetname
dbname = opts.dbname
ids_file = opts.ids_file

if opts.query_type == "skiplimit":
    if dbname == 'ZTF_source_features_20191101_20_fields':
        nlightcurves = 34681547
    elif dbname == 'ZTF_source_features_20191101':
        nlightcurves = 578676249
    else:
        print('dbname %s not known... exiting.')
        exit(0)
elif opts.query_type == "ids":
    data_out = np.load(ids_file)
    nlightcurves = len(data_out)
else:
    print('query_type is not skiplimit or ids... exiting.')
    exit(0)

catalogDir = os.path.join(outputDir,'catalog',algorithm)

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

modelFiles = glob.glob(os.path.join(modelPath, "d11*.%s.*model" % featuresetname))

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
        if opts.doQuadrantScale:
            #qu = {"query_type":"count_documents",
            #      "query": {
            #          "catalog": 'ZTF_source_features_20191101',
            #          "filter": {}
            #          }
            #     }
            #r = ztfperiodic.utils.database_query(kow, qu, nquery = 1)
            #nlightcurves = r['result_data']['query_result']

            #qu = {"query_type":"count_documents",
            #      "query": {
            #          "catalog": 'ZTF_source_features_20191101_20_fields',
            #          "filter": {}
            #          }
            #     }
            #r = ztfperiodic.utils.database_query(kow, qu, nquery = 1)
            #nlightcurves = r['result_data']['query_result']

            Ncatalog = int(np.ceil(float(nlightcurves)/opts.Nmax))

        for ii in range(Ncatalog):
            modelFiles_tmp = []
            for modelFile in modelFiles:
                modelName = modelFile.replace(".model","").split("/")[-1]

                catalogFile = os.path.join(catalogDir,modelName, "%d.h5"%(ii))
                if os.path.isfile(catalogFile):
                    print('%s already exists... continuing.' % catalogFile)
                    continue

                modelFiles_tmp.append(modelFile)

            if opts.doDocker:
                fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic --outputDir %s --program_ids 1,2,3 --field %d --ccd %d --quadrant %d --user %s --pwd %s --batch_size %d -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --doRemoveTerrestrial --doPlots %s\n'%(outputDir, field, ccd, quadrant, opts.user, opts.pwd,opts.batch_size, Ncatalog, ii, opts.algorithm, extra_flags))
            else:
                fid1.write('%s %s/ztfperiodic_classify_objects.py --outputDir %s --user %s --pwd %s -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --dbname %s --doPlots --modelFiles %s --query_type %s --ids_file %s\n'%(opts.python, dir_path, outputDir, opts.user, opts.pwd, Ncatalog, ii, opts.algorithm, dbname, ",".join(modelFiles_tmp),opts.query_type,ids_file))
        
            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" Ncatindex="%d" Ncatalog="%d" modelFiles="%s"\n'%(job_number,job_number,ii, Ncatalog, ",".join(modelFiles_tmp)))
            fid.write('\n\n')
            job_number = job_number + 1

    elif opts.source_type == "catalog":
        for ii in range(Ncatalog):
            modelFiles_tmp = []
            for modelFile in modelFiles:
                modelName = modelFile.replace(".model","").split("/")[-1]

                catalogFile = os.path.join(catalogDir,modelName, "%d.h5"%(ii))
                if os.path.isfile(catalogFile):
                    print('%s already exists... continuing.' % catalogFile)
                    continue

                modelFiles_tmp.append(modelFile)

            if opts.doDocker:
                fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic %s --outputDir %s --user %s --pwd %s --batch_size %d -l Kowalski --source_type catalog --algorithm %s --doRemoveTerrestrial --doRemoveBrightStars --stardist 13.0 --program_ids 1,2,3 --catalog_file %s --doPlots --Ncatalog %d --Ncatindex %d %s\n'%(cpu_gpu_flag, outputDir, opts.user, opts.pwd,opts.batch_size, opts.algorithm, opts.catalog_file,opts.Ncatalog,ii,extra_flags))
            else:
                fid1.write('%s %s/ztfperiodic_classify_objects.py --outputDir %s --user %s --pwd %s -l Kowalski --source_type catalog --algorithm %s --catalog_file %s --doPlots --Ncatalog %d --Ncatindex %d --dbname %s --modelFiles %s\n'%(opts.python, dir_path, outputDir, opts.user, opts.pwd, opts.algorithm, opts.catalog_file, opts.Ncatalog,ii,dbname,",".join(modelFiles_tmp)))

            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" Ncatindex="%d" Ncatalog="%d" modelFiles="%s"\n'%(job_number,job_number, ii, Ncatalog, ",".join(modelFiles_tmp)))
            fid.write('\n\n')
            job_number = job_number + 1

fid1.close()
fid.close()

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = %s/ztfperiodic_classify_objects.py\n'%dir_path)
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
if opts.lightcurve_source == "Kowalski":
    if opts.source_type == "quadrant":
        fid.write('arguments = --outputDir %s --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --user %s --pwd %s -l Kowalski --doPlots --algorithm %s --dbname %s --modelFiles $(modelFiles) --query_type %s --ids_file %s\n'%(outputDir,opts.user,opts.pwd,opts.algorithm,dbname,opts.query_type,ids_file))
    elif opts.source_type == "catalog":
        fid.write('arguments = --outputDir %s --user %s --pwd %s -l Kowalski --source_type catalog --catalog_file %s --doPlots --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --algorithm %s --dbname %s --modelFiles $(modelFiles)\n'%(outputDir,opts.user,opts.pwd,opts.catalog_file,opts.algorithm,dbname))
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 8192\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /local/michael.coughlin/folding.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

