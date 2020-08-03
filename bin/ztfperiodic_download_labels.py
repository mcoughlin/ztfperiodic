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

from joblib import Parallel, delayed
from tqdm import tqdm

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

from zvm import zvm

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    #parser.add_option("--doUpload",  action="store_true", default=False)

    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields/catalog/compare/bw/")
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/labels")

    parser.add_option("-t","--tag",default="d12")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--zvm_user")
    parser.add_option("--zvm_pwd")

    parser.add_option("--doParallel",  action="store_true", default=False)
    parser.add_option("-n","--Ncore",default=8,type=int)

    opts, args = parser.parse_args()

    return opts

def database_query(kow, qu, nquery = 5):
    r = {}
    cnt = 0
    while cnt < nquery:
        r = kow.query(query=qu)
        if "result_data" in r:
            break
        time.sleep(5)
        cnt = cnt + 1
    return r

def get_program_ids(tag):

    if tag == 'vnv_d1':
        zvm_program_ids = [21, 24, 36]
    elif tag == 'vnv_d2':
        zvm_program_ids = [21, 22, 23, 24, 25, 26, 27, 28]
    elif tag == 'vnv_d3':
        zvm_program_ids = [21, 22, 23, 24, 25, 26, 27, 28, 36]
    elif tag == 'vnv_d5':
        zvm_program_ids = [21, 22, 23, 24, 25, 26, 27, 28, 36, 37, 38, 39]
    elif tag == 'vnv_d6':
        zvm_program_ids = [21, 22, 23, 24, 25, 26, 27, 28, 36, 40, 41, 42]
    elif (tag == 'vnv_d7') or (tag == 'vnv_d8'):
        zvm_program_ids = [
            3, 4, 6, 9, 21, 22, 23, 24, 25, 26, 27, 28, 36, 40, 41, 42,
            44, 45, 46, 47, 48, 49
        ]
    elif tag == 'd9':
        zvm_program_ids = [
            3, 4, 6, 9, 21, 22, 23, 24, 25, 26, 27, 28, 36, 40, 41, 42,
            43, 44, 45, 46, 47, 48, 49, 51, 53, 58, 60, 61
        ]
    elif tag == 'd10':
        zvm_program_ids = [
            3, 4, 6, 9, 21, 22, 23, 24, 25, 26, 27, 28, 36, 40, 41, 42,
            43, 44, 45, 46, 47, 48, 49, 51, 53, 58, 60, 61, 62
        ]
    elif tag == 'd11':
        zvm_program_ids = [
            3, 4, 5, 6, 9,
            11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 36, 37,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            51, 53, 54, 56, 58,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            81, 82, 89,
            91
        ]
    elif tag == 'd12':
        zvm_program_ids = [
            3, 4, 5, 6, 9,
            11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 36, 37,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            51, 53, 54, 56, 58,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            81, 82, 89,
            91,
            94, 95, 96, 97, 98, 101, 102
        ]

    return zvm_program_ids

def is_ingested(ztf_id):
    # todo: select sources with a threshold on n_obs
    q = {'query_type': 'count_documents',
         'query': {
             'catalog': catalogs['features'],
             'filter': {'_id': int(ztf_id)}
         }}
    r = database_query(kow,q)
    if not bool(r):
        return {}
    r = r.get('result_data').get('query_result')

    q = {'query_type': 'find',
         'query': {
             'catalog': catalogs['sources'],
             'filter': {'_id': int(ztf_id)},
             'projection': {'_id':0, 'ra': 1, 'dec': 1}
         }}
    radec = database_query(kow,q)
    if radec is None:
        return {}    
    radec = radec.get('result_data').get('query_result')

    if len(radec) > 0:
        return {'ztf_id': int(ztf_id),
                'ingested': r,
                'ra': radec[0]['ra'],
                'dec': radec[0]['dec']}
    else:
        return {}

def get_labels(zvm_id):
    q = {'query_type': 'find',
         'query': {
             'catalog': 'sources',
             'filter': {
                 '_id': zvm_id
             },
             'projection': {
                 'labels': 1
             }
         }
        }

    r = zvmarshal.query(query=q).get('result').get('result_data').get('query_result')
    #print(r)
    return {'zvm_id': zvm_id, 'labels': r[0]['labels']}

def get_features(ztf_id):
    # fixme: cross_matches are omitted for now
    q = {'query_type': 'find',
         'query': {
             'catalog': catalogs['features'],
             'filter': {'_id': int(ztf_id)},
             'projection': {'coordinates': 0, 'cross_matches': 0}
         }}

    r = database_query(kow,q)
    if not bool(r):
        return {}
    r = r.get('result_data').get('query_result')[0]

    return r

# Parse command line
opts = parse_commandline()

catalogs = {'features': 'ZTF_source_features_20191101',
            'sources': 'ZTF_sources_20200401'}

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

dfmfile = os.path.join(outputDir, 'df_m.hdf5')
if not os.path.isfile(dfmfile):

    r = zvmarshal.api(endpoint='programs', method='get', data={'format': 'json'})
    df_p = pd.DataFrame.from_records(r)
    
    for zvm_program_id in df_p['_id']:
        # number of objects in dataset:
        q = {'query_type': 'count_documents',
             'query': {
                 'catalog': 'sources',
                 'filter': {
                     'zvm_program_id': zvm_program_id
                 }
             }
            }
        r = zvmarshal.query(query=q).get('result').get('result_data').get('query_result')
        w = df_p['_id'] == zvm_program_id
        df_p.loc[w, 'n'] = r
    
        # number of labeled objects in dataset:
        q = {'query_type': 'count_documents',
             'query': {
                 'catalog': 'sources',
                 'filter': {
                     'zvm_program_id': zvm_program_id,
                     'labels.0': {'$exists': True}
                 }
             }
            }
        r = zvmarshal.query(query=q).get('result').get('result_data').get('query_result')
        w = df_p['_id'] == zvm_program_id
        df_p.loc[w, 'n_l'] = r
    
    zvm_program_ids = get_program_ids(opts.tag)
    
    q = {'query_type': 'find',
         'query': {
             'catalog': 'sources',
             'filter': {
                 'zvm_program_id': {'$in': zvm_program_ids},
                 'labels.0': {'$exists': True}
             },
             'projection': {
                 'lc.id': 1
             }
         }
        }
    r = zvmarshal.query(query=q).get('result').get('result_data').get('query_result')
    
    ids = []
    for s in r:
        for i in s['lc']:
            ids.append({'ztf_id': i['id'], 'zvm_id': s['_id']})
    df_i = pd.DataFrame.from_records(ids)
    #df_i = df_i[:10000]
    
    print('Checking objects for features...')
    if opts.doParallel:
        r = ProgressParallel(n_jobs=opts.Ncore,use_tqdm=True,total=len(df_i['ztf_id'].unique()))(delayed(is_ingested)(ztf_id) for ztf_id in df_i['ztf_id'].unique())
    else:
        r = []
        for ii, ztf_id in enumerate(df_i['ztf_id'].unique()):
            if np.mod(ii, 100) == 0:
                print('Checked object %d/%d' % (ii+1, len(df_i['ztf_id'].unique())))
            r.append(is_ingested(ztf_id))
    
    df_r = pd.DataFrame.from_records(r)    
    df_m = pd.merge(df_r, df_i, on='ztf_id')
    df_m.to_hdf(dfmfile, key='df_merged', mode='w')
else:
    df_m = pd.read_hdf(dfmfile, key='df_merged')

zvm_ids = set(df_m.loc[df_m['ingested'] == 1, 'zvm_id'].values)

dflabelsfile = os.path.join(outputDir, 'df_zvm_labels.hdf5')
if not os.path.isfile(dflabelsfile):
    print('Checking ZVM for labels...')
    if opts.doParallel:
        l = ProgressParallel(n_jobs=opts.Ncore,use_tqdm=True,total=len(zvm_ids))(delayed(get_labels)(zvm_id) for zvm_id in zvm_ids)
    else:   
        l = []
        for ii, zvm_id in enumerate(zvm_ids):
            if np.mod(ii, 100) == 0:
                print('Getting labels for object %d/%d' % (ii+1, len(zvm_ids)))
            l.append(get_labels(zvm_id))

    df_zvm_labels = pd.DataFrame(l)
    df_zvm_labels.to_hdf(dflabelsfile, key='df_merged', mode='w')
else:
    df_zvm_labels = pd.read_hdf(dflabelsfile, key='df_merged')

labels_source = []
labels = set()

for tu in df_zvm_labels.itertuples():
    labs = tu.labels
    # multiple users may have classified the same object
    # compute the mean values for each label
    _l = {ll['label']: {'v': 0, 'n': 0} for ll in labs}
    for ll in labs:
        _l[ll['label']]['v'] += ll['value']
        _l[ll['label']]['n'] += 1

        labels.add((ll['type'], ll['label']))
    for ll, vv in _l.items():
        _l[ll] = vv['v'] / vv['n']

    labels_source.append(dict(**{'zvm_id': tu.zvm_id}, **_l))

df_labels_source = pd.DataFrame.from_records(labels_source).fillna(0)
df_labels = pd.merge(df_m, df_labels_source,
                     on='zvm_id').drop_duplicates('ztf_id').reset_index(drop=True)

df_label_stats = pd.DataFrame(labels,
                              columns=['type',
                                       'label']).sort_values(by=['type',
                                                                 'label']).reset_index(drop=True)
df_label_stats['number'] = 0

for dl in df_labels.columns.values[5:]:
    ww = df_label_stats['label'] == dl
    df_label_stats.loc[ww, 'number'] = ((df_labels[dl] > 0.0) & (df_labels['ingested'] == 1)).sum()

ztf_ids = sorted(df_m.loc[df_m['ingested'] == 1, 'ztf_id'].unique())

dffeaturesfile = os.path.join(outputDir, 'df_features.hdf5')
if not os.path.isfile(dffeaturesfile):
    print('Pulling labels...')
    if opts.doParallel:
        features = ProgressParallel(n_jobs=opts.Ncore,use_tqdm=True,total=len(ztf_ids))(delayed(get_features)(ztf_id) for ztf_id in ztf_ids)
    else:
        features = []
        for ii, ztf_id in enumerate(ztf_ids):
            if np.mod(ii, 100) == 0:
                print('Getting features for object %d/%d' % (ii+1, len(ztf_ids)))
            features.append(get_features(ztf_id))

    df_features = pd.DataFrame(features).fillna(0)
    df_features.rename(columns={"_id": "ztf_id"}, inplace=True)
    df_features.to_hdf(dffeaturesfile, key='df_merged', mode='w')
else:
    df_features = pd.read_hdf(dffeaturesfile, key='df_merged')

### Merge labels and features into single df
df_ds = pd.merge(df_labels[df_labels['ingested'] == 1], df_features, on='ztf_id')
# keep ra/dec's of the individual light curves (ztf_id)
df_ds.drop(['ra_x', 'dec_x'], axis=1, inplace=True)
df_ds.rename(columns={"ra_y": "ra", "dec_y": "dec"}, inplace=True)
df_ds.fillna(0, inplace=True)
df_ds.reset_index(inplace=True)

datasetfile = os.path.join(outputDir, 'dataset.%s.csv' % opts.tag)
df_ds.to_csv(datasetfile, index=False)

