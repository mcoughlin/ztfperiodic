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

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_featuresetnames
from ztfperiodic.classifier import DNN_v2 

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

    parser.add_option("-n","--normFile",default="/home/michael.coughlin/ZTF/ZTFVariability/pipeline/saved_models/norms.20200615.json")

    parser.add_option("-l","--target_label",default='RR Lyrae')

    opts, args = parser.parse_args()

    return opts

class Dataset(object):

    def __init__(
        self, path_dataset,
        path_labels: str = '../labels',
        features=(
            'ad', 'chi2red', 'f1_a', 'f1_amp', 'f1_b',
            'f1_bic', 'f1_phi0', 'f1_power', 'f1_relamp1', 'f1_relamp2',
            'f1_relamp3', 'f1_relamp4', 'f1_relphi1', 'f1_relphi2', 'f1_relphi3',
            'f1_relphi5', 'f60', 'f70', 'f80', 'f90', 'inv_vonneumannratio', 'iqr',
            'median', 'median_abs_dev',
            # 'n',
            'norm_excess_var', 'norm_peak_to_peak_amp', 'pdot', 'period', 'roms', 'significance',
            'skew', 'smallkurt', 'stetson_j', 'stetson_k', 'sw', 'welch_i', 'wmean',
            'wstd', 'n_ztf_alerts', 'mean_ztf_alert_braai'
        ),
        verbose=False,
        **kwargs
    ):
        """
        load csv file produced by labels*.ipynb

        :param tag:
        :param path_labels:
        :param features:
        :param verbose:
        """
        self.verbose = verbose
        self.features = features

        if self.verbose:
            print(f'Loading {path_dataset}...')
        self.df_ds = pd.read_csv(path_dataset)
        if self.verbose:
            print(self.df_ds[list(features)].describe())
 
        dmdt = []
        if self.verbose:
            print('Moving dmdt\'s to a dedicated numpy array...')
            for i in tqdm(self.df_ds.itertuples(), total=len(self.df_ds)):
                dmdt.append(np.asarray(literal_eval(self.df_ds['dmdt'][i.Index])))
        else:
            for i in self.df_ds.itertuples():
                dmdt.append(np.asarray(literal_eval(self.df_ds['dmdt'][i.Index])))
        self.dmdt = np.array(dmdt)
        self.dmdt = np.expand_dims(self.dmdt, axis=-1)

        # drop in df_ds:
        self.df_ds.drop(columns='dmdt', inplace=True)
        self.df_ds.fillna(0, inplace=True)

    @staticmethod
    def threshold(a, t: float = 0.5):
        b = np.zeros_like(a)
        b[np.array(a) > t] = 1
        return b

    def make(
        self, target_label: str = 'variable', threshold: float = 0.5, balance=None, weight_per_class: bool = True,
        test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42,
        path_norms=None, batch_size: int = 256, shuffle_buffer_size: int = 256, epochs: int = 300,
        **kwargs
    ):
        """
        make datasets for target_label

        :param target_label:
        :param threshold:
        :param balance:
        :param weight_per_class:
        :param test_size:
        :param val_size:
        :param random_state:
        :param path_norms: json file with norms to use to normalize features. if None, norms are computed
        :param batch_size
        :param shuffle_buffer_size
        :param epochs
        :return:
        """

        # Note: Dataset.from_tensor_slices method requires the target variable to be of the int type.
        # TODO: see what to do about it when trying label smoothing in the future.

        # target = np.asarray(list(map(int, np.rint(self.df_ds[target_label].values))))
        target = np.asarray(list(map(int, self.threshold(self.df_ds[target_label].values, t=threshold))))

        self.target = np.expand_dims(target, axis=1)

        neg, pos = np.bincount(target.flatten())
        total = neg + pos
        if self.verbose:
            print(f'Examples:\n  Total: {total}\n  Positive: {pos} ({100 * pos / total:.2f}% of total)\n')

        w_pos = np.rint(self.df_ds[target_label].values) == 1
        index_pos = self.df_ds.loc[w_pos].index
        if target_label == 'variable':
            # 'variable' is a special case: there is an explicit 'non-variable' label:
            w_neg = np.asarray(list(map(int, self.threshold(self.df_ds['non-variable'].values, t=threshold)))) == 1
        else:
            w_neg = ~w_pos
        index_neg = self.df_ds.loc[w_neg].index

        # balance positive and negative examples if there are more negative than positive?
        index_neg_dropped = None
        if balance:
            neg_sample_size = int(np.sum(w_pos) * balance)
            index_neg = self.df_ds.loc[w_neg].sample(n=neg_sample_size, random_state=1).index
            index_neg_dropped = self.df_ds.loc[list(set(self.df_ds.loc[w_neg].index) - set(index_neg))].index

        ds_indexes = index_pos.to_list() + index_neg.to_list()

        # Train/validation/test split (we will use an 81% / 9% / 10% data split by default):

        train_indexes, test_indexes = train_test_split(ds_indexes, shuffle=True,
                                                       test_size=test_size, random_state=random_state)
        train_indexes, val_indexes = train_test_split(train_indexes, shuffle=True,
                                                      test_size=val_size, random_state=random_state)

        # Normalize features (dmdt's are already L2-normalized) (?using only the training samples?).
        # Obviously, the same norms will have to be applied at the testing and serving stages.

        # load/compute feature norms:
        if not path_norms:
            norms = {feature: np.linalg.norm(self.df_ds.loc[ds_indexes, feature]) for feature in self.features}
            for feature, norm in norms.items():
                if np.isnan(norm) or norm == 0.0:
                    norms[feature] = 1.0
            if self.verbose:
                print('Computed feature norms:\n', norms)
        else:
            with open(path_norms, 'r') as f:
                norms = json.load(f)
            if self.verbose:
                print(f'Loaded feature norms from {path_norms}:\n', norms)

        for feature, norm in norms.items():
            self.df_ds[feature] /= norm

        # replace zeros with median values
        if kwargs.get('zero_to_median', False):
            for feature in norms.keys():
                if feature in ('pdot', 'n_ztf_alerts'):
                    continue
                wz = self.df_ds[feature] == 0.0
                if wz.sum() > 0:
                    if feature == 'mean_ztf_alert_braai':
                        median = 0.5
                    else:
                        median = self.df_ds.loc[~wz, feature].median()
                    self.df_ds.loc[wz, feature] = median

        # make tf.data.Dataset's:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[train_indexes, self.features].values, 'dmdt': self.dmdt[train_indexes]},
             target[train_indexes])
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[val_indexes, self.features].values, 'dmdt': self.dmdt[val_indexes]},
             target[val_indexes])
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[test_indexes, self.features].values, 'dmdt': self.dmdt[test_indexes]},
             target[test_indexes])
        )
        dropped_negatives = tf.data.Dataset.from_tensor_slices(
            ({'features': self.df_ds.loc[index_neg_dropped, self.features].values,
              'dmdt': self.dmdt[index_neg_dropped]},
             target[index_neg_dropped])
        ) if balance else None

        # Shuffle and batch the datasets:
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epochs)
        val_dataset = val_dataset.batch(batch_size).repeat(epochs)
        test_dataset = test_dataset.batch(batch_size)

        dropped_negatives = dropped_negatives.batch(batch_size) if balance else None

        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'dropped_negatives': dropped_negatives,
        }

        indexes = {
            'train': np.array(train_indexes),
            'val': np.array(val_indexes),
            'test': np.array(test_indexes),
            'dropped_negatives': np.array(index_neg_dropped.to_list()) if index_neg_dropped is not None else None,
        }

        # How many steps per epoch?

        steps_per_epoch_train = len(train_indexes) // batch_size - 1
        steps_per_epoch_val = len(val_indexes) // batch_size - 1
        steps_per_epoch_test = len(test_indexes) // batch_size - 1

        steps_per_epoch = {'train': steps_per_epoch_train,
                           'val': steps_per_epoch_val,
                           'test': steps_per_epoch_test}
        if self.verbose:
            print(f'Steps per epoch: {steps_per_epoch}')

        # Weight training data depending on the number of samples?
        # Very useful for imbalanced classification, especially when in the cases with a small number of examples.

        if weight_per_class:
            # weight data class depending on number of examples?
            # num_training_examples_per_class = np.array([len(target) - np.sum(target), np.sum(target)])
            num_training_examples_per_class = np.array([len(index_neg), len(index_pos)])

            assert 0 not in num_training_examples_per_class, 'found class without any examples!'

            # fewer examples -- larger weight
            weights = (1 / num_training_examples_per_class) / np.linalg.norm((1 / num_training_examples_per_class))
            normalized_weight = weights / np.max(weights)

            class_weight = {i: w for i, w in enumerate(normalized_weight)}

        else:
            # working with binary classifiers only
            class_weight = {i: 1 for i in range(2)}

        return datasets, indexes, steps_per_epoch, class_weight


# Parse command line
opts = parse_commandline()

catalogs = {'features': 'ZTF_source_features_20191101',
            'sources': 'ZTF_sources_20200401'}

outputDir = opts.outputDir

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

datasetfile = os.path.join(outputDir, 'dataset.%s.csv' % opts.tag)

if opts.target_label in ["RR Lyrae"]:
    featuresetname = 'ontological'
else:
    featuresetname = 'phenomenological'

features = get_featuresetnames(featuresetname)[1:] # remove dmdt

pklfile = datasetfile + ".pkl"
if not os.path.isfile(pklfile):
#if True:
    ds = Dataset(path_dataset=datasetfile, 
                 features=features,
                 verbose=True)
    pickle.dump(ds, open(pklfile, "wb" ) )
ds = pickle.load( open(pklfile, "rb" ) )
print(ds)

threshold = 0.7
balance = 2.5
# balance = 1.1
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
    target_label=opts.target_label, threshold=threshold,
    balance=balance, weight_per_class=weight_per_class,
    test_size=test_size, val_size=val_size, random_state=random_state,
    path_norms=opts.normFile, 
    batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size,
    epochs=epochs
)

classifier = DNN_v2(name='rrlyr')
classifier.setup(features_shape=(len(features), ), dmdt_shape=(26, 26, 1), 
                 dense_branch=True, conv_branch=True,
                 loss='binary_crossentropy', optimizer='adam', lr=3e-4, momentum=0.9,
                 monitor='val_loss', patience=30,
#                  loss='binary_crossentropy', optimizer='adam', lr=2e-4, momentum=0.88,
#                  monitor='val_loss', patience=5,
                 callbacks=('early_stopping', 'tensorboard'),
                 tag=opts.tag, logdir='logs')

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
classifier.save(output_path='saved_models',
                output_format='hdf5', tag=f'rrlyrae_dnn_v2_20200616')

# classifier.save(output_path='saved_models', output_format='hdf5', tag=f'{tag}_p_dnn_v2_20200627')


