#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
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

    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/bw/")
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/rrlyr/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/catalog_d11.vnv.f.fits")

    parser.add_option("--doDifference",  action="store_true", default=False)
    parser.add_option("-d","--differencePath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.dscu.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.rrlyr.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.ea.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.eb.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/slices/d11.ew.f.h5")

    parser.add_option("--doIntersection",  action="store_true", default=False)
    parser.add_option("-i","--intersectionPath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/catalog_d11.rrlyr.f.fits")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts

def query_variability(kow):

    cuts = {
        'pnp': {'$gt': 0.9},
        'rrlyr': {'$lt': 0.9},
        'dscu': {'$lt': 0.9},
        'e': {'$lt': 0.9},
        'ea': {'$lt': 0.9},
        'eb': {'$lt': 0.9},
        'ew': {'$lt': 0.9},
    }

    cuts = {
        'vnv': {'$gt': 0.9},
        'rrlyr': {'$gt': 0.9},
    }

    q = {
        "query_type": "aggregate",
        "query": {
            "catalog": "ZTF_source_classifications_20191101",
            "pipeline": [
                {
                    '$match': {
                        #'pnp': {
                        #    '$elemMatch': {
                        #        'version': 'd10_dnn_v2_20200616', 
                        #        'value': cuts['pnp']
                        #    }
                        #},
                        'vnv': {
                            '$elemMatch': {
                                'version': 'd11_dnn_v2_20200627',
                                'value': cuts['vnv']
                            }
                        }, 
                        'rrlyr': {
                            '$elemMatch': {
                                'version': 'd11_dnn_v2_20200627', 
                                'value': cuts['rrlyr']
                            }
                        }, 
                        #'dscu': {
                        #    '$elemMatch': {
                        #        'version': 'd10_dnn_v2_20200616', 
                        #        'value': cuts['dscu']
                        #    }
                        #}, 
                        #'e': {
                        #    '$elemMatch': {
                        #        'version': 'd10_dnn_v2_20200616', 
                        #        'value': cuts['e']
                        #    }
                        #}, 
    #                     'ea': {
    #                         '$elemMatch': {
    #                             'version': 'd10_dnn_v2_20200616', 
    #                             'value': cuts['ea']
    #                         }
    #                     }, 
    #                     'eb': {
    #                         '$elemMatch': {
    #                             'version': 'd10_dnn_v2_20200616', 
    #                             'value': cuts['eb']
    #                         }
    #                     }, 
    #                     'ew': {
    #                         '$elemMatch': {
    #                             'version': 'd10_dnn_v2_20200616', 
    #                             'value': cuts['ew']
    #                         }
    #                     }
                    }
                }, 
                {
                    '$lookup': {
                        'from': 'ZTF_source_features_20191101', 
                        'localField': '_id', 
                        'foreignField': '_id', 
                        'as': 'features'
                    }
                }, 
                #{
                #    '$match': {
                #        'features.period': {
                #            '$gte': 0.1, 
                #            '$lte': 1
                #        }, 
                #        'features.f1_amp': {
                #            '$gte': 0.3
                #        }
                #    }
                #},
                {
                    '$project': {
                        '_id': 1, 
                        'ra': {
                            '$arrayElemAt': [
                                '$features.ra', 0
                            ]
                        }, 
                        'dec': {
                            '$arrayElemAt': [
                                '$features.dec', 0
                            ]
                        }, 
                        'period': {
                            '$arrayElemAt': [
                                '$features.period', 0
                            ]
                        }, 
                        'f1_amp': {
                            '$arrayElemAt': [
                                '$features.f1_amp', 0
                            ]
                        },
                        'vnv': {
                            '$arrayElemAt': [
                                '$vnv', 0
                            ]
                        },
                        'rrlyr': {
                            '$arrayElemAt': [
                                '$rrlyr', 0
                            ]
                        }
                    }
                }
            ]
        }
    }
    data = kow.query(q).get("result_data", dict()).get("query_result", dict())
    return data

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogPath = opts.catalogPath
differencePath = opts.differencePath   
intersectionPath = opts.intersectionPath

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

scriptpath = os.path.realpath(__file__)
inputDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"input")

WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)
 
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
    df = pd.read_hdf(catalogPath, 'df')
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

idx1 = np.where(df["prob"] >= 0.9)[0]
idx2 = np.where(df1["prob"] >= 0.9)[0]
idx = np.intersect1d(idx1, idx2)
objids_1 = df1.iloc[idx].index

variability = query_variability(kow)
objids_2, probs_2_vnv, probs_2_rrlyr  = [], [], []
for obj in variability:
    objids_2.append(obj["_id"])

objids_2 = np.array(objids_2)

objids = np.union1d(objids_1,objids_2)

objfile = os.path.join(plotDir, 'objids.dat')
fid = open(objfile, 'w')
for objid in objids:
    
    idx = np.where(np.array(df.index) == objid)[0]
    df_slice = df.iloc[idx]
    df1_slice = df1.iloc[idx]

    dnn_classifications = get_kowalski_classifications_objids([objid], kow)

    if len(df_slice) > 0:
        prob_1_vnv, prob_1_rrlyr = df_slice["prob"], df1_slice["prob"]
    else:
        prob_1_vnv, prob_1_rrlyr = -1, -1
    if len(dnn_classifications) > 0:
        prob_2_vnv, prob_2_rrlyr = dnn_classifications["vnv"], dnn_classifications["rrlyr"]
    else:
        prob_2_vnv, prob_2_rrlyr = -1, -1
    fid.write('%d %.5f %.5f %.5f %.5f\n' % (objid, prob_1_vnv, prob_1_rrlyr,
                                            prob_2_vnv, prob_2_rrlyr))
fid.close()

print(stop)
objids_tmp, features = get_kowalski_features_objids(objids, kow)
h5file = os.path.join(plotDir, 'features.h5')
features.to_hdf(h5file, key='df', mode='w')
print(stop)

variability = query_variability(kow)
objids_2 = []
for obj in variability:
    objids_2.append(obj["_id"])

objids_1 = []
plotFiles = glob.glob(os.path.join(plotDir,'*.png'))
for plotFile in plotFiles:
    objid = plotFile.split("/")[-1].replace(".png","")
    objids_1.append(int(objid))

#print(len(objids_1),len(objids_2))
#print(len(np.setdiff1d(objids_1,objids_2))/len(objids_1))

for ii, (index, row) in enumerate(df.iterrows()): 
    if np.mod(ii,100) == 0:
        print('Loading %d/%d'%(ii,len(df)))

    objid, features = get_kowalski_features_objids([index], kow)
    period = features.period.values[0]
    amp = features.f1_amp.values[0]
    if (period < 0.1) or (period > 1.0): continue
    if (amp < 0.3): continue
    lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline = get_kowalski_objids([index], kow)

    objids_1.append(index)

    hjd, magnitude, err = lightcurves[0]
    absmag, bp_rp = absmags[0], bp_rps[0]

    if opts.doPlots:
        phases = np.mod(hjd,2*period)/(2*period)

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        ax1.errorbar(phases, magnitude,err,ls='none',c='k')
        period2=period
        ymed = np.nanmedian(magnitude)
        y10, y90 = np.nanpercentile(magnitude,10), np.nanpercentile(magnitude,90)
        ystd = np.nanmedian(err)
        ymin = y10 - 7*ystd
        ymax = y90 + 7*ystd
        ax1.set_ylim([ymin,ymax])
        ax1.invert_yaxis()
        asymmetric_error = np.atleast_2d([absmag[1], absmag[2]]).T
        hist2 = ax2.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
        if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
            ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                         c='r',zorder=1,fmt='o')
        ax2.set_xlim([-1,4.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        fig.colorbar(hist2[3],ax=ax2)
        plt.suptitle('Period: %.5f days' % period)
        pngfile = os.path.join(plotDir,'%d.png' % objid)
        fig.savefig(pngfile, bbox_inches='tight')
        plt.close()


