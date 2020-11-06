#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import pickle
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import astropy
from astropy.table import Table, vstack
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.lcstats import sawtooth_decomposition, make_s
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query
from ztfperiodic.utils import galex_query

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("--doSawtooth",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/condor")
    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2_ids_DR2/catalog/compare/rrlyr/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/slices_merged/pnp.d11_dnn_v2_20200627.h5.h5")

    parser.add_option("--doDifference",  action="store_true", default=False)
    parser.add_option("-d","--differencePath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.dscu.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.rrlyr.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ea.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.eb.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ew.f.h5")

    parser.add_option("--doIntersection",  action="store_true", default=False)
    parser.add_option("-i","--intersectionPath",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/slices_merged/wuma.d12_dnn_v2_20200921.h5.h5")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--doBrutus",  action="store_true", default=False)
    parser.add_option("-b","--brutusPath",default="/home/michael.coughlin/ZTF/brutus/data/DATAFILES/")

    parser.add_option("--Nmax",default=100.0,type=int)

    opts, args = parser.parse_args()

    return opts


# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogPath = opts.catalogPath
differencePath = opts.differencePath   
intersectionPath = opts.intersectionPath

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

catalogDir = os.path.join(outputDir,'catalog')
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

kow = []
nquery = 10
cnt = 0
while cnt < nquery:
    try:
        kow = Kowalski(username=opts.user, password=opts.pwd)
        break
    except:
        time.sleep(5)
    cnt = cnt + 1
if cnt == nquery:
    raise Exception('Kowalski connection failed...')

if ".h5" in catalogPath:
    try:
        df = pd.read_hdf(catalogPath, 'df')
    except:
        df = pd.read_hdf(catalogPath, 'df_merged')

elif ".fits" in catalogPath:
    tab = Table.read(catalogPath, format='fits')
    df = tab.to_pandas()
    df.set_index('objid',inplace=True)

if opts.doDifference:
    differenceFiles = differencePath.split(",")
    for differenceFile in differenceFiles:
        df1 = pd.read_hdf(differenceFile, 'df')
        idx = df.index.difference(df1.index)
        df = df.loc[idx]
if opts.doIntersection:
    intersectionFiles = intersectionPath.split(",")
    for intersectionFile in intersectionFiles:
        if ".h5" in catalogPath:
            df1 = pd.read_hdf(intersectionFile, 'df')
        else:
            tab = Table.read(intersectionFile, format='fits')
            df1 = tab.to_pandas()
            df1.set_index('objid',inplace=True)

        idx = df.index.intersection(df1.index)
        df = df.loc[idx]

        idx = df1.index.intersection(df.index)
        df1 = df1.loc[idx]

mag = ['AllWISE__w1mpro', 'AllWISE__w2mpro',
       'AllWISE__w3mpro', 'AllWISE__w4mpro',
       'PS1_DR1__gMeanPSFMag',
       'PS1_DR1__rMeanPSFMag',
       'PS1_DR1__iMeanPSFMag',
       'PS1_DR1__zMeanPSFMag',
       'PS1_DR1__yMeanPSFMag']
       
magerr = ['AllWISE__w1sigmpro', 'AllWISE__w2sigmpro', 
          'AllWISE__w3sigmpro', 'AllWISE__w4sigmpro',
          'PS1_DR1__gMeanPSFMagErr',
          'PS1_DR1__rMeanPSFMagErr',
          'PS1_DR1__iMeanPSFMagErr',
          'PS1_DR1__zMeanPSFMagErr',
          'PS1_DR1__yMeanPSFMagErr']

parallax = ['Gaia_DR2__parallax', 'Gaia_DR2__parallax_error']

catalogFile = os.path.join(outputDir,"catalog.h5")
if not os.path.isfile(catalogFile):
    objids = np.array(df.index)
    #objids = objids[:100]
    
    objids, features = get_kowalski_features_objids(objids, kow,
                                                 featuresetname='all')
    
    snr = features["Gaia_DR2__parallax"]/features["Gaia_DR2__parallax_error"]
    idx = np.where((snr >= 3) & ~np.isnan(snr))[0]
    
    objids = np.array(objids.iloc[idx])
    features = features.iloc[idx]
    
    ra = np.array(features['ra']),
    dec = np.array(features['dec'])
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galcoords = np.vstack([coord.galactic.l.deg, coord.galactic.b.deg]).T
    
    mag = np.array(features[mag])
    magerr = np.array(features[magerr])
    parallax = np.array(features[parallax])
    
    with h5py.File(catalogFile, 'w') as hf:
        hf.create_dataset("objids",  data=objids)
        hf.create_dataset("mag",  data=mag)
        hf.create_dataset("magerr",  data=magerr)
        hf.create_dataset("parallax",  data=parallax)
        hf.create_dataset("galcoords", data=galcoords)
    
with h5py.File(catalogFile, 'r') as f:
    mag, magerr, parallax = f['mag'][:], f['magerr'][:], f['parallax'][:]
    objids, galcoords = f['objids'][:], f['galcoords'][:]
    
dir_path = os.path.dirname(os.path.realpath(__file__))

condordag = os.path.join(condorDir,'condor.dag')
fid = open(condordag,'w')
condorsh = os.path.join(condorDir,'condor.sh')
fid1 = open(condorsh,'w')

job_number = 0
Ncatalog = int(np.ceil(float(len(objids))/opts.Nmax))
for ii in range(Ncatalog):

    fid1.write('python %s/ztfperiodic_brutus.py --doPlots --outputDir %s --Ncatalog %d --Ncatindex %d --catalogFile %s\n' % (dir_path,outputDir,Ncatalog,ii,catalogFile))

    fid.write('JOB %d condor.sub\n'%(job_number))
    fid.write('RETRY %d 3\n'%(job_number))
    fid.write('VARS %d jobNumber="%d" Ncatindex="%d" Ncatalog="%d"\n'%(job_number,job_number, ii, Ncatalog))
    fid.write('\n\n')
    job_number = job_number + 1    

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = %s/ztfperiodic_brutus.py\n'%dir_path)
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
fid.write('arguments = --outputDir %s --Ncatalog $(Ncatalog) --Ncatindex $(Ncatindex) --catalogFile %s\n'%(outputDir, catalogFile))
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 8192\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /local/michael.coughlin/brutus.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()
