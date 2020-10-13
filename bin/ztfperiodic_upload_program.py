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
    parser.add_option("--doUpload",  action="store_true", default=False)

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
    sources = kow.query(q).get("data", dict())
    if len(sources) > 0:
        source = sources[0]
        if len(source['period']) > 0:
            source['period'] = source['period'][0]
        else:
            source['period'] = np.nan
    else:
        source = []

    return source

def save_source(zvmarshal, source, zvm_program_id, verbose = False,
                classification = None):

    for ii in range(3):
        if True:
        #try:
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
            if (classification is not None) and (not classification == "OTHER"):
                labels = []
                if classification in ["EA", "EB", "EW"]:
                    labels.append({'type': 'phenomenological',
                                   'label': 'variable',
                                   'value': 1.0})
                    labels.append({'type': 'phenomenological',
                                   'label': 'periodic',
                                   'value': 1.0})
                    labels.append({'type': 'phenomenological',
                                   'label': 'eclipsing',
                                   'value': 1.0})     
                    labels.append({'type': 'phenomenological',
                                   'label': classification,
                                   'value': 1.0})
                    labels.append({'type': 'intrinsic',
                                   'label': "binary star",
                                   'value': 1.0})
                    if classification == "EA":
                        labels.append({'type': 'intrinsic',
                                       'label': "detached eclipsing MS-MS",
                                       'value': 1.0})
                    elif classification == "EB":
                        labels.append({'type': 'intrinsic',
                                       'label': "Beta Lyr",
                                       'value': 1.0})
                    elif classification == "EW":
                        labels.append({'type': 'intrinsic',
                                       'label': "W Uma",
                                       'value': 1.0})
                elif classification in ["RR","DSCT"]:
                    labels.append({'type': 'phenomenological',
                                   'label': 'variable',
                                   'value': 1.0})
                    labels.append({'type': 'phenomenological',
                                   'label': 'periodic',
                                   'value': 1.0})
                    labels.append({'type': 'phenomenological',
                                   'label': 'sawtooth',
                                   'value': 1.0})
                    labels.append({'type': 'pulsator',
                                   'label': 'sawtooth',
                                   'value': 1.0})
                    if classification == "RR":
                        labels.append({'type': 'intrinsic',
                                       'label': "RR Lyrae",
                                       'value': 1.0})
                    elif classification == "DSCT":
                        labels.append({'type': 'intrinsic',
                                       'label': "Delta Scu",
                                       'value': 1.0})
                elif classification in ["BOGUS"]:
                    labels.append({'type': 'phenomenological',
                                   'label': 'bogus',
                                   'value': 1.0})
 
                r = zvmarshal.api(endpoint=f'sources/{source_id}',
                                  method='post',
                                  data={'source_id': source_id,
                                        'action': 'set_labels',
                                        'labels': labels})
            break
        #except Exception as e:
        #    if verbose:
        #        print(e)
        #        _err = traceback.format_exc()
        #        print(_err)
        #    continue

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

infoDir = os.path.join(outputDir,'info')
if not os.path.isdir(infoDir):
    os.makedirs(infoDir)

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

print('Using program ID: %d' % (zvm_program_id))

objids = []
periods = []
classifications = None

compareFile = os.path.join(outputDir,'catalog.dat')
if os.path.isfile(compareFile):
    try:
        data_out = np.loadtxt(compareFile)
        objids = data_out[:,1]
        periods = data_out[:,4]
    except:
        data_out = Table.read(compareFile, format='ascii',
                              names=('objids', 'periods', 'classifications'))

        objids, periods = data_out["objids"].tolist(), data_out["periods"].tolist()
        classifications = data_out["classifications"].tolist()

else:
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
if classifications is not None:
    classifications = [classifications[ii] for ii in idx]

fid = open(os.path.join(infoDir, 'objs.dat'), 'w')

for ii, (objid, period) in enumerate(zip(objids, periods)):
    if np.mod(ii, 50) == 0:
        print('Pushed %d/%d' % (ii, len(objids)))

    source = get_source(int(objid))
    if len(source) == 0:
        print('No info for %d... continuing.' % objid)
        continue
    if np.isnan(source['period']) and (period > 0):
        source['period'] = period  

    fid.write('%d %.10f %.10f %.10f\n' % (objid, source['ra'], source['dec'],
                                          source['period'])) 

    
    if opts.doUpload:
        if classifications is not None: 
            save_source(zvmarshal, source, zvm_program_id, verbose = True,
                        classification = classifications[ii])   
        else:
            save_source(zvmarshal, source, zvm_program_id, verbose = True)

fid.close() 
