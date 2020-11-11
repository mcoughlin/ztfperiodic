#!/usr/bin/env python

import os, sys
import json
import glob
import optparse
import copy
import time
import h5py
from functools import reduce
import traceback
import pickle

import numpy as np
import pandas as pd
from ast import literal_eval

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

from tqdm.keras import TqdmCallback

import tensorflow as tf
import tensorflow_addons as tfa

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_featuresetnames
from ztfperiodic.classifier import Dataset, DNN_v2 


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/labels")
    parser.add_option("-t","--tag",default="d12")
    parser.add_option("-n","--normFile",default="/home/michael.coughlin/ZTF/ZTFVariability/pipeline/saved_models/norms.20200615.json")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

catalogs = {'features': 'ZTF_source_features_20191101',
            'sources': 'ZTF_sources_20200401'}

outputDir = opts.outputDir

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

modelsDir = os.path.join(outputDir,'models')
if not os.path.isdir(modelsDir):
    os.makedirs(modelsDir)

logsDir = os.path.join(outputDir,'logs')
if not os.path.isdir(logsDir):
    os.makedirs(logsDir)

datasetfile = os.path.join(outputDir, 'dataset.%s.csv' % opts.tag)

target_labels = {'agn': 'AGN',
                 'bis': 'binary star',
                 'blyr': 'Beta Lyr',
                 'ceph': ['Cepheid', 'Cepheid type-II'],
                 'dscu': 'Delta Scu',
                 'e': 'eclipsing',
                 'ea': 'EA',
                 'eb': 'EB',
                 'ew': 'EW',
                 'fla': 'flaring',
                 'i': 'irregular',
                 'pnp': 'periodic',
                 'puls': 'pulsator',
                 'rrlyr': 'RR Lyrae',
                 'rscvn': 'RS CVn',
                 'vnv': 'variable',
                 'yso': 'YSO'
                }

target_labels = {'pnp': 'periodic',
                 'vnv': 'variable'}

for label in sorted(target_labels.keys()):
    print('Analyzing %s' % label)
 

    target_label = target_labels[label]       
    modelFile = os.path.join(modelsDir, '%s.%s_dnn_v2.h5' % (label, opts.tag))

    if os.path.isfile(modelFile): continue

    if label in ['agn', 'bis', 'blyr', 'ceph', 'dscu', 'puls', 'rrlyr',
                 'rscvn', 'yso']:
        featuresetname = 'ontological'
    elif label in ['e', 'ea', 'eb', 'ew', 'fla', 'i', 'pnp', 'vnv']:
        featuresetname = 'phenomenological'
    
    features = get_featuresetnames(featuresetname)[1:] # remove dmdt
    
    pklfile = datasetfile.replace("csv", featuresetname) + ".pkl"
    if not os.path.isfile(pklfile):
        ds = Dataset(path_dataset=datasetfile, 
                     features=features,
                     verbose=True)
        pickle.dump(ds, open(pklfile, "wb" ) )
    ds = pickle.load( open(pklfile, "rb" ) )
    print(ds)

    if label in ["bis", "pnp", "vnv", "e", "ew"]:
        balance = None
    else:
        balance = 2.5
    
    threshold = 0.7
    # balance = None
    # weight_per_class = True
    weight_per_class = False
    test_size = 0.1
    val_size = 0.1
    random_state = 42
    # batch_size = 32
    # shuffle_buffer_size = 32
    batch_size = 64
    shuffle_buffer_size = 64
    epochs = 300
    
    datasets, indexes, steps_per_epoch, class_weight = ds.make(
        target_label=target_label, threshold=threshold,
        balance=balance, weight_per_class=weight_per_class,
        test_size=test_size, val_size=val_size, random_state=random_state,
        path_norms=opts.normFile, 
        batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size,
        epochs=epochs
    )
    
    classifier = DNN_v2(name=label)
    classifier.setup(features_shape=(len(features), ), dmdt_shape=(26, 26, 1), 
                     dense_branch=True, conv_branch=True,
                     loss='binary_crossentropy', optimizer='adam', lr=3e-4, momentum=0.9,
                     monitor='val_loss', patience=30,
    #                  loss='binary_crossentropy', optimizer='adam', lr=2e-4, momentum=0.88,
    #                  monitor='val_loss', patience=5,
                     callbacks=('early_stopping', 'tensorboard'),
                     tag=opts.tag, logdir=logsDir)
    
    #load_pre_trained = True
    load_pre_trained = False
    if load_pre_trained:
        # DNN_v2:
        # saved_model = 'saved_models/rrlyr.d9_dnn_v2_20200615.h5'
    #     saved_model = 'saved_models/rrlyr.d10_dnn_v6_20200616.h5'
        saved_model = 'saved_models/rrlyr.d11_dnn_v2_20200627.h5'
        saved_model = '/home/mcoughlin/ZTF/ZTFVariability/pipeline/saved_models/rrlyr.d12_dnn_v2_20200921.h5'
    
        classifier.model = tf.keras.models.load_model(saved_model)
    
    print(classifier.model.summary())
    
    class_weight = {0: 1, 1: 1}
    # class_weight = {0: 2, 1: 1}
    # class_weight = {0: 1, 1: 2.5}
    # class_weight = {0: 1.1, 1: 1}
    epochs = 300
    # epochs = 20
    
    classifier.meta['callbacks'].append(TqdmCallback(verbose=1))
    
    classifier.train(datasets['train'], datasets['val'], 
                     steps_per_epoch['train'], steps_per_epoch['val'],
                     epochs=epochs, class_weight=class_weight, verbose=0)
    
    stats = classifier.evaluate(datasets['test'],
                                callbacks=[tfa.callbacks.TQDMProgressBar()],
                                verbose=0)
    print(stats)
    classifier.save(output_path=modelsDir,
                    output_format='hdf5', tag='%s_dnn_v2' % opts.tag)
    
    data = tf.data.Dataset.from_tensor_slices(
        ({'features': ds.df_ds[list(ds.features)].values, 'dmdt': ds.dmdt}, ds.target)
    )
    predictions = classifier.predict(data.batch(len(ds.df_ds)))
    pt = np.hstack((predictions, ds.target))
    pt_thresholded = np.rint(pt)
    w = np.logical_xor(pt_thresholded[:, 0], pt_thresholded[:, 1])
    
    print(len(w), np.sum(w), np.sum(w)/len(w))
    
    print(list(set(ds.df_ds.loc[w, 'zvm_id'].values.tolist())))
