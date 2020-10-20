#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

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
    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/top_sources/labeling_guide/json_plots")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/top_sources/labeling_guide/json")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogPath = opts.catalogPath

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

fs = 24
colors = ['g','r','y']
symbols = ['x', 'o', '^']
fids = [1,2,3]
bands = {1: 'g', 2: 'r', 3: 'i'}
inv_bands = {v: k for k, v in bands.items()}

jsonFiles = glob.glob(os.path.join(catalogPath, '*.json'))

for ii, jsonFile in enumerate(jsonFiles): 
    with open(jsonFile) as json_file:
        data_json = json.load(json_file)
    lc = {}
    lc["hjd"], lc["mag"], lc["magerr"], lc["fid"] = [], [], [], []

    for data in data_json["data"]["scatterPlot"]["data"]:
        dat = data["seriesData"]
        opt = data["seriesOptions"]
        band, period = opt["label"], opt["period"]

        for d in dat:
            lc["hjd"].append(d["x"])
            lc["mag"].append(-d["y"])
            lc["magerr"].append(d["yerr"])
            lc["fid"].append(inv_bands[band])

    lc["hjd"] = np.array(lc["hjd"])
    lc["mag"] = np.array(lc["mag"])
    lc["magerr"] = np.array(lc["magerr"])
    lc["fid"] = np.array(lc["fid"])

    index = int(jsonFile.split("/")[-1].replace(".json",""))
    pngfile = os.path.join(outputDir,'%d.png' % index)
    if opts.doPlots:
        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()
        for jj, (fid, color, symbol) in enumerate(zip(fids, colors, symbols)):
            idx = np.where(lc["fid"] == fid)[0]
            plt.errorbar(np.mod(lc["hjd"][idx], 2.0*period)/(2.0*period), lc["mag"][idx],yerr=lc["magerr"][idx],fmt='%s%s' % (color,symbol), label=bands[fid])
        plt.xlabel('Phase', fontsize = fs)
        plt.ylabel('Magnitude [ab]', fontsize = fs)
        plt.legend(prop={'size': 20})
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.tick_params(axis='both', which='minor', labelsize=fs)
        ax.invert_yaxis()
        plt.title("Period = %.3f days"%period, fontsize = fs)
        plt.savefig(pngfile, bbox_inches='tight')
        plt.close()
