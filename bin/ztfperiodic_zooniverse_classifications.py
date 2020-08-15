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

from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import combine_lcs
from ztfperiodic.zooniverse import Subjects

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

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/top_sources")
    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2_ids_DR2/catalog/compare/rrlyr/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.pnp.f.h5")

    parser.add_option("--doDifference",  action="store_true", default=False)
    parser.add_option("-d","--differencePath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.dscu.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.rrlyr.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ea.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.eb.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ew.f.h5")

    parser.add_option("--doIntersection",  action="store_true", default=False)
    parser.add_option("-i","--intersectionPath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ceph.f.h5")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--doSubjectSet",  action="store_true", default=False)
    parser.add_option("--zooniverse_user")
    parser.add_option("--zooniverse_pwd")
    parser.add_option("--zooniverse_id",default=13032,type=int)

    parser.add_option("-N","--Nexamples",default=10,type=int)

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

intersectionType = intersectionPath.split("/")[-1].replace(".h5","")
outputDir = os.path.join(outputDir, intersectionType)

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

if opts.doSubjectSet:
    subs = Subjects(username=opts.zooniverse_user,
                    password=opts.zooniverse_pwd,
                    project_id=opts.zooniverse_id) 

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

col = df1.columns[0]
idx = np.argsort(df1[col])[::-1]
idx = np.array(idx).astype(int)[:opts.Nexamples]

idy = df.index.intersection(df1.iloc[idx].index)
df = df.loc[idy]
df1 = df1.iloc[idx]

fs = 24
colors = ['g','r','y']
symbols = ['x', 'o', '^']
fids = [1,2,3]
bands = {1: 'g', 2: 'r', 3: 'i'}

if opts.doSubjectSet:
   image_list, metadata_list, subject_set_name = [], [], intersectionType 

objfile = os.path.join(plotDir, 'objids.dat')
objfid = open(objfile, 'w')
for ii, (index, row) in enumerate(df.iterrows()): 
    if np.mod(ii,100) == 0:
        print('Loading %d/%d'%(ii,len(df)))

    objid, features = get_kowalski_features_objids([index], kow)
    period = features.period.values[0]
    lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline = get_kowalski_objids([index], kow)
    ra, dec = coordinates[0][0], coordinates[0][1]

    lightcurves_all = get_kowalski(ra, dec, kow,
                                   min_epochs=20)
    lightcurves_combined = combine_lcs(lightcurves_all)

    hjd, magnitude, err = lightcurves[0]
    absmag, bp_rp = absmags[0], bp_rps[0]
    gaia = gaia_query(ra, dec, 5/3600.0)
    d_pc, gofAL = None, None
    if gaia:
        Plx = gaia['Plx'].data.data[0] # mas
        gofAL = gaia["gofAL"].data.data[0]
        # distance in pc
        if Plx > 0 :
            d_pc = 1 / (Plx*1e-3)

    if opts.doPlots:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        plt.axes(ax1)
        bands_count = np.zeros((len(fids),1))
        for jj, (fid, color, symbol) in enumerate(zip(fids, colors, symbols)):
            for ii, key in enumerate(lightcurves_all.keys()):
                lc = lightcurves_all[key]
                if not lc["fid"][0] == fid: continue
                idx = np.where(lc["fid"][0] == fids)[0]
                if bands_count[idx] == 0:
                    plt.errorbar(np.mod(lc["hjd"], 2.0*period)/(2.0*period), lc["mag"],yerr=lc["magerr"],fmt='%s%s' % (color,symbol), label=bands[fid])
                else:
                    plt.errorbar(np.mod(lc["hjd"], 2.0*period)/(2.0*period), lc["mag"],yerr=lc["magerr"],fmt='%s%s' % (color,symbol))
                bands_count[idx] = bands_count[idx] + 1
        plt.xlabel('Phase', fontsize = fs)
        plt.ylabel('Magnitude [ab]', fontsize = fs)
        plt.legend(prop={'size': 20})
        ax1.tick_params(axis='both', which='major', labelsize=fs)
        ax1.tick_params(axis='both', which='minor', labelsize=fs)
        ax1.invert_yaxis()
        plt.title("Period = %.3f days"%period, fontsize = fs)

        plt.axes(ax2)
        asymmetric_error = np.atleast_2d([absmag[1], absmag[2]]).T
        hist2 = ax2.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
        if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
            ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                         c='r',zorder=1,fmt='o')
        ax2.set_xlim([-1,4.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        cbar = fig.colorbar(hist2[3],ax=ax2)
        cbar.set_label('Object Count')
        ax2.set_xlabel('Gaia BP - RP')
        ax2.set_ylabel('Gaia $M_G$')

        if (not d_pc is None) and (not gofAL is None):
            ax2.set_title("d = %d [pc], gof = %.1f"%(d_pc, gofAL), fontsize = fs)

        plt.tight_layout()
        pngfile = os.path.join(plotDir,'%d.png' % objid)
        fig.savefig(pngfile, bbox_inches='tight')
        plt.close()

    objfid.write('%d %.10f %.10f %.10f\n' % (index, ra, dec, period))
    print('%d %.10f %.10f %.10f' % (index, ra, dec, period))

    if opts.doSubjectSet:
        image_list.append(pngfile)
        mdict = {'candidate': int(objid),
                 'ra': ra, 'dec': dec, 
                 'period': period}
        metadata_list.append(mdict)
objfid.close()

if opts.doSubjectSet:
    ret = subs.add_new_subject(image_list,
                               metadata_list,
                               subject_set_name=subject_set_name)


