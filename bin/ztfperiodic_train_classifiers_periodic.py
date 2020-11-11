#!/usr/bin/env python

import os, sys
import json
import glob
import optparse
import joblib
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

import torch.multiprocessing as mp

from torch.multiprocessing import current_process
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from ztfperiodic.periodicnetwork.model.iresnet import Classifier as iresnet
from ztfperiodic.periodicnetwork.model.itcn import Classifier as itcn
from ztfperiodic.periodicnetwork.model.rnn import Classifier as rnn
from ztfperiodic.periodicnetwork.data import MyDataset as MyDataset

from ztfperiodic.periodicnetwork.light_curve import LightCurve
from ztfperiodic.periodicnetwork.util import *
from ztfperiodic.periodicnetwork.train import train

# Based on: https://github.com/kmzzhang/periodicnetwork/blob/master/train.py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/labels_periodic")
    parser.add_option("-t","--tag",default="d12")

    parser.add_option('--L', type=int, default=128,
                        help='training sequence length')
    parser.add_option('--filename', type=str, default='test.pkl',
                        help='dataset filename. file is expected in ./data/')
    parser.add_option('--frac-train', type=float, default=0.8,
                        help='training sequence length')
    parser.add_option('--frac-valid', type=float, default=0.25,
                        help='training sequence length')
    parser.add_option('--train-batch', type=int, default=32,
                        help='training sequence length')
    parser.add_option('--varlen_train', action='store_true', default=False,
                        help='enable variable length training')
    parser.add_option('--input', type=str, default='dtf',
                        help='input representation of data. combination of t/dt/f/df/g.')
    parser.add_option('--n_test', type=int, default=1,
                        help='number of different sequence length to test')
    parser.add_option('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_option('--dropout', type=float, default=0,
                        help='dropout rate')
    parser.add_option('--dropout-classifier', type=float, default=0,
                        help='dropout rate')
    parser.add_option('--permute', action='store_true', default=False,
                        help='data augmentation')
    parser.add_option('--clip', type=float, default=-1,
                        help='gradient clipping')
    parser.add_option('--max_epoch', type=int, default=50,
                        help='maximum number of training epochs')
    parser.add_option('--min_maxpool', type=int, default=2,
                        help='minimum length required for maxpool operation.')
    parser.add_option('--ngpu', type=int, default=1,
                        help='number of gpu devices to use. neg number refer to particular single device number')
    parser.add_option('--njob', type=int, default=1,
                        help='maximum number of networks to train on each gpu')
    parser.add_option('--K', type=int, default=8,
                        help='number of data partition to use')
    parser.add_option('--pseed', type=int, default=0,
                        help='random seed for data partition (only when K = 1)')
    parser.add_option('--network', type=str, default='iresnet',
                        help='name of the neural network to train')
    parser.add_option('--kernel', type=int, default=2,
                        help='kernel size')
    parser.add_option('--depth', type=int, default=7,
                        help='network depth')
    parser.add_option('--n_layer', type=int, default=2,
                        help='(iresnet/resnet only) number of convolution per residual block')
    parser.add_option('--hidden', type=int, default=128,
                        help='hidden dimension')
    parser.add_option('--hidden-classifier', type=int, default=32,
                        help='hidden dimension for final layer')
    parser.add_option('--max_hidden', type=int, default=128,
                        help='(iresnet/resnet only) maximum hidden dimension')
    parser.add_option('--two_phase', action='store_true', default=False,
                        help='')
    parser.add_option('--print_every', type=int, default=-1,
                        help='')
    parser.add_option('--seed', type=int, default=0,
                        help='random seed for network seed and random partition')
    parser.add_option('--cudnn_deterministic', action='store_true', default=False,
                        help='')
    parser.add_option('--min_sample', type=int, default=0,
                        help='minimum number of pre-segmented light curve per class')
    parser.add_option('--max_sample', type=int, default=100000,
                        help='maximum number of pre-segmented light curve per class during testing')
    parser.add_option('--retrain', action='store_true', default=False,
                        help='continue training from checkpoint')
    parser.add_option('--no-log', action='store_true', default=False,
                        help='continue training from checkpoint')
    parser.add_option('--note', type=str, default='',
                        help='')
    parser.add_option('--project-name', type=str, default='',
                        help='for weights and biases tracking')
    parser.add_option('--decay-type', type=str, default='plateau',
                        help='')
    parser.add_option('--patience', type=int, default=5,
                        help='patience for learning decay')
    parser.add_option('--early_stopping', type=int, default=0,
                        help='terminate training if loss does not improve by 10% after waiting this number of epochs')
    
    opts, args = parser.parse_args()

    return opts

def get_device(path):
    device = os.listdir(path)[0]
    os.remove(path+'/'+device)
    return device

def get_network(n_classes):

    if opts.network in ['itcn', 'iresnet']:
        padding = 'cyclic'
    else:
        padding = 'zero'

    if opts.network in ['itcn', 'tcn']:
        clf = itcn(
            num_inputs=n_inputs,
            num_class=n_classes,
            depth=opts.depth,
            hidden_conv=opts.hidden,
            hidden_classifier=opts.hidden_classifier,
            dropout=opts.dropout,
            kernel_size=opts.kernel,
            dropout_classifier=opts.dropout_classifier,
            aux=3,
            padding=padding
        ).type(dtype)

    elif opts.network in ['iresnet', 'resnet']:
        clf = iresnet(
            n_inputs,
            n_classes,
            depth=opts.depth,
            nlayer=opts.n_layer,
            kernel_size=opts.kernel,
            hidden_conv=opts.hidden,
            max_hidden=opts.max_hidden,
            padding=padding,
            min_length=opts.min_maxpool,
            aux=3,
            dropout_classifier=opts.dropout_classifier,
            hidden=opts.hidden_classifier
        ).type(dtype)

    elif opts.network in ['gru', 'lstm']:
        clf = rnn(
            num_inputs=n_inputs,
            hidden_rnn=opts.hidden,
            num_layers=opts.depth,
            num_class=n_classes,
            hidden=opts.hidden_classifier,
            rnn=opts.network.upper(),
            dropout=opts.dropout,
            aux=3
        ).type(dtype)

    return clf

def train_helper(param):
    global map_loc
    train_index, test_index, name = param
    split = [chunk for i in train_index for chunk in data[i].split(opts.L, opts.L) if data[i].label is not None]
    for lc in split:
        lc.period_fold()
    unique_label, count = np.unique([lc.label for lc in split], return_counts=True)
    print('------------after segmenting into L={}------------'.format(opts.L))
    print(unique_label)
    print(count)

    # shape: (N, L, 3)
    X_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([convert_label[chunk.label] for chunk in split])

    x, means, scales = getattr(PreProcessor, opts.input)(np.array(X_list), periods)
    print('shape of the training dataset array:', x.shape)
    mean_x = x.reshape(-1, n_inputs).mean(axis=0)
    std_x = x.reshape(-1, n_inputs).std(axis=0)
    x -= mean_x
    x /= std_x
    if opts.varlen_train:
        x = np.array(X_list)
    if opts.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L-1)

    aux = np.c_[means, scales, np.log10(periods)]
    aux_mean = aux.mean(axis=0)
    aux_std = aux.std(axis=0)
    aux -= aux_mean
    aux /= aux_std
    scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])
    if not opts.varlen_train:
        scales_all = None
    else:
        np.save(name + '_scales.npy', scales_all)

    train_idx, val_idx = train_test_split(label, 1 - opts.frac_valid, -1)
    if opts.ngpu < 0:
        torch.cuda.set_device(int(-1*opts.ngpu))
        map_loc = 'cuda:{}'.format(int(-1*opts.ngpu))

    print('Using ', torch.cuda.current_device())
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    sys.stdout = sys.__stdout__
    train_dset = MyDataset(x[train_idx], aux[train_idx], label[train_idx])
    val_dset = MyDataset(x[val_idx], aux[val_idx], label[val_idx])
    train_loader = DataLoader(train_dset, batch_size=opts.train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, batch_size=128, shuffle=False, drop_last=False)

    split = [chunk for i in test_index for chunk in data[i].split(opts.L, opts.L)]
    for lc in split:
        lc.period_fold()

    # shape: (N, L, 3)
    x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    x, means, scales = getattr(PreProcessor, opts.input)(np.array(x_list), periods)

    # whiten data
    x -= mean_x
    x /= std_x
    if opts.varlen_train:
        x = np.array(X_list)
    if opts.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L)

    label = np.array([convert_label[chunk.label] for chunk in split])
    aux = np.c_[means, scales, np.log10(periods)]
    aux -= aux_mean
    aux /= aux_std

    test_dset = MyDataset(x, aux, label)
    test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

    mdl = get_network(n_classes)
    if not opts.no_log:
        import wandb
        wandb.init(project=opts.project_name, config=opts, name=name)
        wandb.watch(mdl)

    modelname = name+opts.note+'.pth'
    if opts.retrain or not os.path.isfile(modelname):
        if opts.retrain:
            mdl.load_state_dict(torch.load(name + '.pth', map_location=map_loc))
            opts.lr *= 0.01
        optimizer = optim.Adam(mdl.parameters(), lr=opts.lr)
        torch.manual_seed(opts.seed)
        train(mdl, optimizer, train_loader, val_loader, test_loader, opts.max_epoch,
              print_every=opts.print_every, save=True, filename=name+opts.note, patience=opts.patience,
              early_stopping_limit=opts.early_stopping, use_tqdm=True, scales_all=scales_all, clip=opts.clip,
              retrain=opts.retrain, decay_type=opts.decay_type, monitor='accuracy', log=not opts.no_log,
              perm=opts.permute)

    # load the model with the best validation accuracy for testing on the test set
    mdl.load_state_dict(torch.load(modelname, map_location=map_loc))

    # Evaluate model on sequences of different length
    accuracy_length = np.zeros(len(lengths))
    accuracy_class_length = np.zeros(len(lengths))
    mdl.eval()
    with torch.no_grad():
        for j, length in enumerate(lengths):
            split = [chunk for i in test_index for chunk in data[i].split(length, length)]
            # num_chunks = np.array([len(data[i].split(length, length)) for i in test_index])
            # num_chunks = num_chunks[num_chunks != 0]
            # assert np.sum(num_chunks) == len(split)
            for lc in split:
                lc.period_fold()

            # shape: (N, L, 3)
            x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
            periods = np.array([lc.p for lc in split])
            x, means, scales = getattr(PreProcessor, opts.input)(np.array(x_list), periods)

            # whiten data
            x -= mean_x
            x /= std_x
            if opts.two_phase:
                x = np.concatenate([x, x], axis=1)
            x = np.swapaxes(x, 2, 1)
            # shape: (N, 3, L)

            label = np.array([convert_label[chunk.label] for chunk in split])
            aux = np.c_[means, scales, np.log10(periods)]
            aux -= aux_mean
            aux /= aux_std

            test_dset = MyDataset(x, aux, label)
            test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)
            softmax = torch.nn.Softmax(dim=1)
            predictions = []
            ground_truths = []
            probs = []
            for i, d in enumerate(test_loader):
                x, aux_, y = d
                logprob = mdl(x.type(dtype), aux_.type(dtype))
                predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                probs.extend(list(softmax(logprob).detach().cpu().numpy()))
                ground_truths.extend(list(y.numpy()))

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)
            #pred_perobj = [np.argmax(np.log(probs[sum(num_chunks[:i]):sum(num_chunks[:i + 1])]).sum(axis=0))
            #               for i in range(len(num_chunks))]
            #gt_perobj = [ground_truths[sum(num_chunks[:i])] for i in range(len(num_chunks))]
            #

            if len(lengths) == 1:
                np.save('{}_predictions.npy'.format(name), np.c_[predictions, ground_truths])

            accuracy_length[j] = (predictions == ground_truths).mean()
            accuracy_class_length[j] = np.array(
                [(predictions[ground_truths == l] == ground_truths[ground_truths == l]).mean()
                 for l in np.unique(ground_truths)]).mean()
    if opts.ngpu > 1:
        return_device(path, device)
    return accuracy_length, accuracy_class_length


if __name__ == '__main__':

    # Parse command line
    opts = parse_commandline()

    outputDir = opts.outputDir
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    plotDir = os.path.join(outputDir,'plots')
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)   

    datafilename = os.path.join(outputDir, 'lightcurves.pkl')
 
    if opts.network == 'resnet' or opts.network == 'iresnet':
        save_name = '{}-K{}-D{}-NL{}-H{}-MH{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(opts.network,
            opts.kernel,
            opts.depth,
            opts.n_layer,
            opts.hidden,
            opts.max_hidden,
            opts.L,
            int(opts.varlen_train),
            opts.input,
            opts.lr,
            opts.hidden_classifier,
            max(opts.dropout, opts.dropout_classifier),
            int(opts.two_phase))
    else:
        save_name = '{}-K{}-D{}-H{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(opts.network,
            opts.kernel,
            opts.depth,
            opts.hidden,
            opts.L,
            int(opts.varlen_train),
            opts.input,
            opts.lr,
            opts.clip,
            opts.dropout,
            int(opts.two_phase))
    
    if current_process().name != 'MainProcess':
        if opts.njob > 1 or opts.ngpu > 1:
            path = 'device'+save_name+opts.note
            device = get_device(path)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device[0])
    else:
        print('save filename:')
        print(save_name)
    
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        dint = torch.cuda.LongTensor
        map_loc = 'cuda:0'
    else:
        assert opts.ngpu == 1
        dtype = torch.FloatTensor
        dint = torch.LongTensor
        map_loc = 'cpu'
    
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        dint = torch.cuda.LongTensor
        map_loc = 'cuda:0'
    else:
        assert opts.ngpu == 1
        dtype = torch.FloatTensor
        dint = torch.LongTensor
        map_loc = 'cpu'
    
    if opts.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if 'asassn' in datafilename:
        opts.max_sample = 20000
    if opts.n_test == 1:
        lengths = [opts.L]
    else:
        lengths = np.linspace(16, opts.L * 2, opts.n_test).astype(np.int)
        if opts.L not in lengths:
            lengths = np.sort(np.append(lengths, opts.L))
    data = joblib.load(datafilename)
    #with open('data/{}'.format(datafilename), 'rb') as handle:
    #    data = pickle.load(handle)
    # sanity check on dataset
    for lc in data:
        positive = lc.errors > 0
        positive *= lc.errors < 99
        lc.times = lc.times[positive]
        lc.measurements = lc.measurements[positive]
        lc.errors = lc.errors[positive]
        lc.p = lc.best_period
    
    if 'macho' in datafilename:
        for lc in data:
            if 'LPV' in lc.label:
                lc.label = "LPV"
    
    # Generate a list all labels for train/test split
    unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
    use_label = unique_label[count >= opts.min_sample]
    
    n_classes = len(use_label)
    new_data = []
    for cls in use_label:
        class_data = [lc for lc in data if lc.label == cls]
        new_data.extend(class_data[:min(len(class_data), opts.max_sample)])
    data = new_data
    
    all_label_string = [lc.label for lc in data]
    unique_label, count = np.unique(all_label_string, return_counts=True)
    print('------------before segmenting into L={}------------'.format(opts.L))
    print(unique_label)
    print(count)
    convert_label = dict(zip(use_label, np.arange(len(use_label))))
    all_labels = np.array([convert_label[lc.label] for lc in data])
    
    if opts.input in ['dtdfg', 'dtfg', 'dtfe']:
        n_inputs = 3
    elif opts.input in ['df', 'f', 'g']:
        n_inputs = 1
    else:
        n_inputs = 2

    jobs = []
    np.random.seed(opts.seed)
    for i in range(opts.K):
        if opts.K == 1:
            i = opts.pseed
        trains, tests = train_test_split(all_labels, train_size=opts.frac_train, random_state=i)
        jobs.append((trains, tests, '{}/{}-{}'.format(outputDir, save_name, i)))
    try:
        os.mkdir(outputDir)
    except:
        pass
    if opts.ngpu <= 1 and opts.njob == 1:
        results = []
        for j in jobs:
            results.append(train_helper(j))
    else:
        create_device('device'+save_name+opts.note, opts.ngpu, opts.njob)
        ctx = mp.get_context('spawn')
        with ctx.Pool(opts.ngpu * opts.njob) as p:
            results = p.map(train_helper, jobs)
        shutil.rmtree('device' + save_name+opts.note)
    results = np.array(results)
    results_all = np.c_[lengths, results[:, 0, :].T]
    results_class = np.c_[lengths, results[:, 1, :].T]

    np.save('{}/{}{}-results.npy'.format(outputDir, save_name, opts.note), results_all)
    np.save('{}/{}{}-results-class.npy'.format(outputDir, save_name, opts.note), results_class)

    results = '{}/{}-{}_predictions.npy'.format(outputDir, save_name, i)
    data_out = np.load(results)
    
    X, Y = np.meshgrid(np.arange(len(unique_label)+1), np.arange(len(unique_label)+1))
    plt.figure()
    cm = confusion_matrix(data_out[:,1], data_out[:,0], normalize='true')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.pcolor(X, Y, cm, cmap=plt.get_cmap('cool'), vmin=0, vmax=1)
    #cax = ax.matshow(cm)
    fig.colorbar(cax)
    plt.xticks(np.arange(len(unique_label))+0.5, unique_label, rotation='vertical')
    plt.yticks(np.arange(len(unique_label))+0.5, unique_label, rotation='horizontal')
    plt.xlim([0, len(unique_label)])
    plt.ylim([0, len(unique_label)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plotName = os.path.join(plotDir,'confusion.pdf')
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()
