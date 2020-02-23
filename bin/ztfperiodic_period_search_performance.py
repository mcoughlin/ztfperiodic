
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

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")
    parser.add_option("-a","--algorithm",default="CE")

    parser.add_option("--doQuadrantScale",  action="store_true", default=False)

    parser.add_option("-l","--lightcurve_source",default="Kowalski")
    parser.add_option("-s","--source_type",default="quadrant")
    parser.add_option("--catalog_file",default="../input/xray.dat")
    parser.add_option("--Ncatalog",default=13.0,type=int)
    parser.add_option("--Nmax",default=1000.0,type=int)

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--doReadFiles",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
Ncatalog = opts.Ncatalog
outputDir = opts.outputDir
algorithm = opts.algorithm

catalogDir = os.path.join(outputDir,'catalog',algorithm)

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

if opts.doQuadrantScale:
    kow = Kowalski(username=opts.user, password=opts.pwd)

nobsfile = os.path.join(condorDir,'nobs.dat')

if opts.doReadFiles:
    fid = open(nobsfile,'w')
    
    if opts.lightcurve_source == "Kowalski":
    
        if opts.source_type == "quadrant":
            fields, ccds, quadrants = np.arange(1,880), np.arange(1,17), np.arange(1,5)
            fields1 = [683,853,487,718,372,842,359,778,699,296]
            #fields1 = [487,718]
            fields2 = [841,852,682,717,488,423,424,563,562,297,700,777]
            fields = fields1 + fields2
            #fields = [600]
            #fields = [718]
            #fields = [851,848,797,761,721,508,352,355,364,379]

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
     
                        nobs = 0
                        for ii in range(Ncatalog):
                            catalogFile = os.path.join(catalogDir,"%d_%d_%d_%d.h5"%(field, ccd, quadrant, ii))
                            if os.path.isfile(catalogFile):
                                try:
                                    with h5py.File(catalogFile, 'r') as f:
                                        name = f['names'].value
                                        filters = f['filters'].value
                                        stats = f['stats'].value
                                    nobs = nobs + len(name)
                                except:
                                    continue
                        if nobs > 0:
                            print('%d %d' % (nobs, float(nlightcurves)), file=fid, flush=True)
    fid.close()

data_out = np.loadtxt(nobsfile)
nobs = np.sum(data_out[:,0])
ntot = np.sum(data_out[:,1])

print('Nobs: %d, Ntot: %d' % (nobs, ntot))

