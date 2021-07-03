
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
    parser.add_option("-a","--algorithm",default="dnn")

    parser.add_option("--doQuadrantScale",  action="store_true", default=False)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=1000000.0,type=int)

    parser.add_option("-m","--modelPath",default="/home/mcoughlin/ZTF/labels_d14/models/")

    parser.add_option("--doDocker",  action="store_true", default=False)

    parser.add_option("-f","--featuresetname",default="b")
    parser.add_option("-d","--dbname",default="ZTF_source_features_DR3")
    parser.add_option("-q","--query_type",default="ids")
    parser.add_option("-i","--ids_file",default="/home/michael.coughlin/ZTF/ZTFVariability/ids/ids.20fields.npy")

    parser.add_option("--doLoadIDFile",  action="store_true", default=False)

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

idsDir = os.path.join(condorDir,'ids')
if not os.path.isdir(idsDir):
    os.makedirs(idsDir)
idsDir = "/home/michael.coughlin/ZTF/output_quadrants_Primary_DR5/condor/ids"

if algorithm == "dnn":
    modelFiles = []
    
    if featuresetname == "ontological":
        varclasses = ['puls', 'dscu', 'ceph', 'rrlyr', 'rrlyrab', 'rrlyrc', 'rrlyrbl', 'rrlyrd', 'srv', 'bis', 'blyr', 'rscvn', 'agn', 'yso', 'wuma', 'ell']
    elif featuresetname == "phenomenological":
        varclasses = ['vnv', 'pnp', 'i', 'e', 'ea', 'eb', 'ew', 'fla', 'bogus', 'dip', 'lpv', 'saw', 'sine']
    for varclass in varclasses:
        for trainingset in ['d14', 'd12', 'd11', 'd10']:
            modelFile = glob.glob(os.path.join(modelPath, "%s*%s*h5" % (varclass, trainingset)))        
            if len(modelFile) > 0:
                modelFiles.append(modelFile[0])
                break
elif algorithm == "xgboost":
    modelFiles = glob.glob(os.path.join(modelPath, "d11*.%s.*model" % featuresetname))

dir_path = os.path.dirname(os.path.realpath(__file__))

condordag = os.path.join(condorDir,'condor.dag')
fid = open(condordag,'w') 
condorsh = os.path.join(condorDir,'condor.sh')
fid1 = open(condorsh,'w') 

job_number = 0

if opts.doQuadrantScale:
    TIMEOUT = 60
    protocol, host, port = "https", "gloria.caltech.edu", 443
    kow = Kowalski(username=opts.user, password=opts.pwd,
                   timeout=TIMEOUT,
                   protocol=protocol, host=host, port=port)

if opts.lightcurve_source == "Kowalski":

    if opts.source_type == "quadrant":

        fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
        fields1 = [683,853,487,718,372,842,359,778,699,296]
        fields2 = [841,852,682,717,488,423,424,563,562,297,700,777]
        #fields3 = [851,848,797,761,721,508,352,355,364,379]
        #fields4 = [1866,1834,1835,1804,1734,1655,1565]

        #fields = fields1 + fields2
        #fields_complete = fields1 + fields2 # + fields3 + fields4
        #fields = np.arange(300,400)
        #fields = np.arange(700,800)
        #fields = np.setdiff1d(fields,fields_complete)

        #fields = [700]
        fields = np.arange(250,882)
        #fields = np.arange(350,500)
        #fields = np.arange(686,880)
        #fields = [400]

        for field in fields:
            print('Running field %d' % field)
            for ccd in ccds:
                for quadrant in quadrants:
                    if opts.doQuadrantScale:
                        if opts.doLoadIDFile:
                            idsFile = os.path.join(idsDir,"%d_%d_%d.npy"%(field, ccd, quadrant))
                            if not os.path.isfile(idsFile):
                                continue
                            objids = np.load(idsFile)
                            nlightcurves = len(objids)
                            Ncatalog = int(np.ceil(float(nlightcurves)/opts.Nmax))
                        else:
                            #qu = {"query_type":"count_documents",
                            #      "query": {
                            #          "catalog": 'ZTF_source_features_DR3',
                            #          "filter": {'field': {'$eq': int(field)},
                            #                     'ccd': {'$eq': int(ccd)},
                            #                     'quad': {'$eq': int(quadrant)}
                            #                     }
                            #               }
                            #     }
                            #r = ztfperiodic.utils.database_query(kow, qu, nquery = 1)
                            #if not "data" in r: continue
                            #nlightcurves = r['data']
                            #if nlightcurves == 0: continue

                            Ncatalog = int(np.ceil(float(nlightcurves)/opts.Nmax))

                            idsFile = os.path.join(idsDir,"%d_%d_%d.npy"%(field, ccd, quadrant))
                            if not os.path.isfile(idsFile):
                                print(idsFile)
                                qu = {"query_type":"find",
                                      "query": {"catalog": 'ZTF_source_features_DR3',
                                                "filter": {'field': {'$eq': int(field)},
                                                           'ccd': {'$eq': int(ccd)},
                                                           'quad': {'$eq': int(quadrant)}
                                                          },
                                                "projection": "{'_id': 1}"},
                                     }
                                qu = {"query_type":"find",
                                      "query": {"catalog": 'ZTF_source_features_DR3',
                                                "filter": {},
                                                "kwargs": {"limit": 1000},
                                                "projection": "{'_id': 1}"},
                                     }
                                r = {}
                                r['data'] = []
                                while len(r['data']) == 0:
                                    r = ztfperiodic.utils.database_query(kow, qu, nquery = 100)
                                objids = []
                                for obj in r['data']:
                                    objids.append(obj['_id'])
                                np.save(idsFile, objids)

                    for ii in range(Ncatalog):
                        modelFiles_tmp = []
                        for modelFile in modelFiles:
                            modelName = modelFile.replace(".model","").split("/")[-1]
            
                            catalogFile = os.path.join(catalogDir,modelName, "%d_%d_%d_%d.h5"%(field, ccd, quadrant, ii))
                            if os.path.isfile(catalogFile):
                                print('%s already exists... continuing.' % catalogFile)
                                continue
            
                            modelFiles_tmp.append(modelFile)
                        if len(modelFiles_tmp) == 0: continue
            
                        if opts.doDocker:
                            fid1.write('nvidia-docker run --runtime=nvidia python-ztfperiodic --outputDir %s --program_ids 1,2,3 --field %d --ccd %d --quadrant %d --user %s --pwd %s --batch_size %d -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --doRemoveTerrestrial --doPlots %s\n'%(outputDir, field, ccd, quadrant, opts.user, opts.pwd,opts.batch_size, Ncatalog, ii, opts.algorithm, extra_flags))
                        else:
                            fid1.write('%s %s/ztfperiodic_classify_objects.py --outputDir %s --user %s --pwd %s -l Kowalski --source_type quadrant --Ncatalog %d --Ncatindex %d --algorithm %s --dbname %s --doPlots --modelFiles %s --query_type %s --ids_file %s\n'%(opts.python, dir_path, outputDir, opts.user, opts.pwd, Ncatalog, ii, opts.algorithm, dbname, ",".join(modelFiles_tmp),opts.query_type,idsFile))
                    
                        fid.write('JOB %d condor.sub\n'%(job_number))
                        fid.write('RETRY %d 3\n'%(job_number))
                        fid.write('VARS %d jobNumber="%d" Ncatindex="%d" Ncatalog="%d" modelFiles="%s" idsFile="%s"\n'%(job_number,job_number,ii, Ncatalog, ",".join(modelFiles_tmp),idsFile))
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
        fid.write('arguments = --outputDir %s --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --user %s --pwd %s -l Kowalski --doPlots --algorithm %s --dbname %s --modelFiles $(modelFiles) --query_type %s\n'%(outputDir,opts.user,opts.pwd,opts.algorithm,dbname,opts.query_type))
    elif opts.source_type == "catalog":
        fid.write('arguments = --outputDir %s --user %s --pwd %s -l Kowalski --source_type catalog --catalog_file %s --doPlots --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --algorithm %s --dbname %s --modelFiles $(modelFiles)\n'%(outputDir,opts.user,opts.pwd,opts.catalog_file,opts.algorithm,dbname))
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 16384\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /local/michael.coughlin/folding.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

