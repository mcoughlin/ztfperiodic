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

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids

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
        source['period'] = source['period'][0]
    else:
        source = []

    return source

def save_source(irow):
    i, row = irow

    verbose = False

    for ii in range(3):
        try:
            # see if there is a saved source to merge with
            q = {"query_type": "cone_search",
                 "object_coordinates": {
                     "radec": {'source': [row.ra, row.dec]},
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
            r = z.query(query=q)
            data = r['result']['result_data']['sources']['source']
            if len(data) > 0:
                # saved to this program? merge
                source_id = data[0]['_id']
                ztf_source_id = int(row._id)
                r = z.api(endpoint=f'sources/{source_id}', method='post',
                          data={'source_id': source_id, 'action': 'merge', '_id': ztf_source_id})
                # save period:
                if not np.isnan(row['period']):
                    r = z.api(endpoint=f'sources/{source_id}', method='post',
                              data={'source_id': source_id, 'action': 'add_period',
                                    'period': row['period'], 'period_unit': 'Days'})
                # display(JSON(r, expanded=False))
            else:
                # not previously saved? save by position:
                r = z.api(endpoint='sources', method='put',
                          data={'ra': row.ra, 'dec': row.dec,
                                'prefix': 'ZTF', 'naming': 'random',
                                'zvm_program_id': int(row.zvm_program_id),
                                'automerge': True})
                source_id = r['result']['_id']
                if verbose:
                    print('saved: ', source_id, int(row._id))

                # set period
                if not np.isnan(row['period']):
                    r = z.api(endpoint=f'sources/{source_id}', method='post',
                              data={'source_id': source_id, 'action': 'add_period',
                                    'period': row['period'], 'period_unit': 'Days'})

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

filenames = glob.glob(os.path.join(plotDir,'*.png'))
objids = []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].split(".png")[0]
    objids.append(int(filenameSplit))

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

zvmarshal = []
nquery = 10
cnt = 0
while cnt < nquery:
    try:
        zvmarshal = zvm(username=opts.zvm_user, password=opts.zvm_pwd)
        break
    except:
        time.sleep(5)
    cnt = cnt + 1
if cnt == nquery:
    raise Exception('zvm connection failed...')

#r = zvmarshal.api(
#    endpoint='programs', method='put', 
#    data={
#        'program_name': opts.program_name,
#        'program_description': opts.program_description
#    }
#)

for objid in objids:
    source = get_source(objid)
    print(source)


