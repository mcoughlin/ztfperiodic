#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
from functools import reduce
import traceback

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

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

from zvm import zvm

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/bw/")
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_ids_DR2/catalog/compare/bw/")

    parser.add_option("-p","--program_name",default="high_amplitude_weird_DR2")
    parser.add_option("-d","--program_description",default="confident predictions of periodic but unidentified objects in DR2, 20200723")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--zvm_user")
    parser.add_option("--zvm_pwd")

    parser.add_option("--sigthresh",default=None,type=float)

    opts, args = parser.parse_args()

    return opts

def get_source(ztf_id):
    q = {
        "query_type": "aggregate",
        "query": {
            "catalog": "ZTF_sources_20200401",
            "pipeline": [
                {
                    "$match": {
                        "_id": ztf_id
                    }
                },
                {
                    "$lookup": {
                        "from": "ZTF_source_features_20191101",
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": "features"
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "ra": 1,
                        "dec": 1,
                        "period": "$features.period"
                    }
                }
            ]
        }
    }
    sources = kow.query(q).get("result_data", dict()).get("query_result", dict())
    if len(sources) > 0:
        source = sources[0]
        if len(source['period']) > 0:
            source['period'] = source['period'][0]
        else:
            source['period'] = np.nan
    else:
        source = []

    return source

def save_source(zvmarshal, source, zvm_program_id, verbose = False):

    for ii in range(3):
        try:
            # see if there is a saved source to merge with
            q = {"query_type": "cone_search",
                 "object_coordinates": {
                     "radec": {'source': [source["ra"], source["dec"]]},
                     "cone_search_radius": 2,
                     "cone_search_unit": "arcsec"
                 },
                 "catalogs": {
                     "sources": {
                         "filter": {'zvm_program_id': zvm_program_id},
                         "projection": {'_id': 1}
                     }
                 }
                }
            r = zvmarshal.query(query=q)
            data = r['result']['result_data']['sources']['source']
            if len(data) > 0:
                # saved to this program? merge
                source_id = data[0]['_id']
                ztf_source_id = int(source["_id"])
                r = zvmarshal.api(endpoint=f'sources/{source_id}',
                                  method='post',
                                  data={'source_id': source_id,
                                        'action': 'merge',
                                        '_id': ztf_source_id})
                # save period:
                if not np.isnan(source['period']):
                    r = zvmarshal.api(endpoint=f'sources/{source_id}',
                                      method='post',
                                      data={'source_id': source_id,
                                            'action': 'add_period',
                                            'period': source['period'],
                                            'period_unit': 'Days'})
                # display(JSON(r, expanded=False))
            else:
                # not previously saved? save by position:
                r = zvmarshal.api(endpoint='sources', method='put',
                                  data={'ra': source["ra"],
                                        'dec': source["dec"],
                                        'prefix': 'ZTF', 'naming': 'random',
                                        'zvm_program_id': int(zvm_program_id),
                                        'automerge': True})
                source_id = r['result']['_id']
                if verbose:
                    print('saved: ', source_id, int(source["_id"]))

                # set period
                if not np.isnan(source['period']):
                    r = zvmarshal.api(endpoint=f'sources/{source_id}',
                                      method='post',
                                      data={'source_id': source_id,
                                            'action': 'add_period',
                                            'period': source['period'],
                                            'period_unit': 'Days'})

                # add classifications:
                # set labels
                # labels = []
                # if verbose:
                #     print(row['variable'], row['non-variable'])
                # if row['variable'] > 0:
                #     labels.append({'type': 'phenomenological',
                #                    'label': 'variable',
                #                    'value': row['variable']})
                # if row['non-variable'] > 0:
                #     labels.append({'type': 'phenomenological',
                #                    'label': 'non-variable',
                #                    'value': row['non-variable']})
                #
                # r = z.api(endpoint=f'sources/{source_id}', method='post',
                #           data={'source_id': source_id, 'action': 'set_labels',
                #                 'labels': labels})

            break
        except Exception as e:
            if verbose:
                print(e)
                _err = traceback.format_exc()
                print(_err)
            continue

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir

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

nquery = 10
cnt = 0
while cnt < nquery:
    try:
        zvmarshal = zvm(username=str(opts.zvm_user),
                        password=str(opts.zvm_pwd),
                        verbose=True, host="rico.caltech.edu")
        break
    except:
        time.sleep(5)
    cnt = cnt + 1
if cnt == nquery:
    raise Exception('zvm connection failed...')

zvm_program_id = -1
r = zvmarshal.api(endpoint='programs', method='get', data={'format': 'json'})
for prog in r:
    if prog["name"] == opts.program_name:
        zvm_program_id = prog["_id"]
        break
if zvm_program_id < 0:
    print('Creating new program...')
    r = zvmarshal.api(endpoint='programs', method='put',
                      data={'program_name': str(opts.program_name),
                            'program_description': str(opts.program_description)
                           }
                     )
    zvm_program_id = r["result"]["_id"]

objids = []
periods = []

filenames = glob.glob(os.path.join(plotDir,'*.png'))
if len(filenames) == 0:
    filedirs = glob.glob(os.path.join(outputDir,'*-*'))
    for ii, filedir in enumerate(filedirs):
        filenames = glob.glob(os.path.join(filedir,'*.png'))
        for jj, filename in enumerate(filenames):
            if np.mod(jj, 10) == 0:
                print('Dir %d/%d File %d/%d' % (ii+1,len(filedirs),
                                                jj+1,len(filenames)))
            filenameSplit = filename.split("/")[-1].split(".png")[0].split("_")
            sig, ra, dec, period, filt = np.array(filenameSplit,
                                                  dtype=float)

            if not opts.sigthresh is None:
                if sig < opts.sigthresh: continue

            lcs = get_kowalski(ra, dec, kow, radius = 1.0)
            for objid in lcs.keys():
                objids.append(objid)
                periods.append(period)
 
else:
    for filename in filenames:
        filenameSplit = filename.split("/")[-1].split(".png")[0]
        objids.append(int(filenameSplit))
        periods.append(-1)

objids = np.array(objids)
periods = np.array(periods)
idx = np.argsort(objids)
objids, periods = objids[idx], periods[idx]

for ii, (objid, period) in enumerate(zip(objids, periods)):
    if np.mod(ii, 10) == 0:
        print('Pushed %d/%d' % (ii, len(objids)))

    source = get_source(int(objid))
    if len(source) == 0:
        print('No info for %d... continuing.' % objid)
        continue
    if np.isnan(source['period']) and (period > 0):
        source['period'] = period   
    save_source(zvmarshal, source, zvm_program_id, verbose = True)    
