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
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/var/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/catalog_d11.vnv.c.fits")

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
    q = {
        "query_type": "aggregate",
        "query": {
            "catalog": "ZTF_source_classifications_20191101",
            "pipeline": [
                {
                    '$match': {
                        'pnp': {
                            '$elemMatch': {
                                'version': 'd10_dnn_v2_20200616', 
                                'value': cuts['pnp']
                            }
                        }, 
                        'rrlyr': {
                            '$elemMatch': {
                                'version': 'd10_dnn_v2_20200616', 
                                'value': cuts['rrlyr']
                            }
                        }, 
                        'dscu': {
                            '$elemMatch': {
                                'version': 'd10_dnn_v2_20200616', 
                                'value': cuts['dscu']
                            }
                        }, 
                        'e': {
                            '$elemMatch': {
                                'version': 'd10_dnn_v2_20200616', 
                                'value': cuts['e']
                            }
                        }, 
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
                {
                    '$match': {
                        'features.period': {
                            '$gte': 0.1, 
                            '$lte': 1
                        }, 
                        'features.f1_amp': {
                            '$gte': 0.3
                        }
                    }
                },
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

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

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

cat = Table.read(catalogPath, format='fits')
idx1 = np.where(cat["prob"]<=0.1)[0]
idx2 = np.where(cat["prob"]>=0.9)[0]

nsamples = 1000

objids1 = cat["objid"][idx1]
objids2 = cat["objid"][idx2]

idx1 = np.random.choice(len(objids1), size=nsamples)
idx2 = np.random.choice(len(objids2), size=nsamples)

objids1 = objids1[idx1]
objids2 = objids2[idx2]

objids1_tmp, features1 = get_kowalski_features_objids(objids1, kow)
objids1_tmp, features2 = get_kowalski_features_objids(objids2, kow)

feature_set11 = ['median', 'wmean', 'chi2red', 'roms', 'wstd', 'norm_peak_to_peak_amp',
           'norm_excess_var', 'median_abs_dev', 'iqr', 'f60', 'f70', 'f80', 'f90',
           'skew', 'smallkurt', 'inv_vonneumannratio', 'welch_i', 'stetson_j',
           'stetson_k', 'ad', 'sw']

keys = ["wstd", 'norm_peak_to_peak_amp', 'norm_excess_var', 'skew', 'inv_vonneumannratio', 'welch_i', 'stetson_j', 'stetson_k', 'ad', 'sw'] 

labels = [r"$m_{\mathrm{var}}$","Norm. Peak to Peak Amp.","Norm. Excess Var.","Skew","Inv. Von Neumann","Welch/Stetson I","Stetson J","Stetson K","Anderson-Darling","Shapiro-Wilk"]

color2 = 'coral'
color1 = 'cornflowerblue'

#labels = []
plotName = "%s/features.pdf"%(plotDir)
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
for ii, key in enumerate(keys):
    val1, val2 = features1[key], features2[key]
    val1_min, val1_max = np.percentile(val1, 5), np.percentile(val1, 95)
    val2_min, val2_max = np.percentile(val2, 5), np.percentile(val2, 95)

    val1 = (val1-val1_min)/(val1_max-val1_min)
    val2 = (val2-val2_min)/(val2_max-val2_min)

    val1 = val1[(val1 >=0) & (val1 <=1)]
    val2 = val2[(val2 >=0) & (val2 <=1)]

    idx = 2*ii
    parts = plt.violinplot(val1,[idx-0.25])
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

    perc50 = np.percentile(val1,50)
    if ii == 0:
        plt.plot([idx-0.5,idx],[perc50,perc50],'--',color=color1,label="Non-Variable", linewidth=3)
    else:
        plt.plot([idx-0.5,idx],[perc50,perc50],'--',color=color1, linewidth=3)

    parts = plt.violinplot(val2,[idx+0.25])
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color2)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color2)
        pc.set_edgecolor(color2)

    perc50 = np.percentile(val2,50)
    if ii == 0:
        plt.plot([idx,idx+0.5],[perc50,perc50],'--',color=color2,label="Variable", linewidth=3)
    else:
        plt.plot([idx,idx+0.5],[perc50,perc50],'--',color=color2, linewidth=3)

    #labels.append(key)

    #print(key, 10**perc50)

ax.set_xticks(2*np.arange(len(keys)))
ax.set_xticklabels(labels)

plt.xlabel('Features',fontsize=32)
plt.ylabel('Values',fontsize=24)
plt.grid()
plt.legend(loc=2, prop={'size': 24})
ax.get_yaxis().set_visible(False)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24, rotation=75)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

