#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import json
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

import astropy
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import ascii, fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.utils import get_kowalski_list
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import combine_lcs
from ztfperiodic.zooniverse import ZooProject


# HR plot options
color_dict = {'blue': '#217CA3',
              'mustard': '#E29930',
              'asphalt': '#32384D',
              'shawdow': '#211F30'}

density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color_dict['blue'], (1, 1, 1, 0)]).reversed()

n = 255
minTrunk = 0.35
maxTrunk = 1.0
trunc_cmap = LinearSegmentedColormap.from_list(
         'trunc(density_cmap,0.35,1)',
         density_cmap(np.linspace(minTrunk, maxTrunk, n)))

def arc_patch(xy, width, height, theta1 = 0, theta2 = 180, resolution=50, 
              **kwargs):

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((width*np.cos(theta)  + xy[0], 
                        height*np.sin(theta) + xy[1]))
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True, **kwargs)

    return poly



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
    parser.add_option("--doLCFile",  action="store_true", default=False)
    parser.add_option("--doFakeData",  action="store_true", default=False)

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
    parser.add_option("--zooniverse_id",default=4878,type=int)

    parser.add_option("-N","--Nexamples",default=10,type=int)

    parser.add_option("-t","--tag",default="v1")

    parser.add_option("--doThreshold",  action="store_true", default=False)
    parser.add_option("--threshold_min",default=0.7, type=float)
    parser.add_option("--threshold_max",default=1.0, type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogPath = opts.catalogPath
differencePath = opts.differencePath   
intersectionPath = opts.intersectionPath

if ".h5" in catalogPath:
    intersectionType = intersectionPath.split("/")[-1].replace(".h5","")
elif ".fits" in catalogPath:
    intersectionType = intersectionPath.split("/")[-1].replace(".fits","")
elif ".csv" in catalogPath:
    intersectionType = catalogPath.split("/")[-1].replace(".csv","")

outputDir = os.path.join(outputDir, intersectionType)

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

jsonDir = os.path.join(outputDir,'json')
if not os.path.isdir(jsonDir):
    os.makedirs(jsonDir)

scriptpath = os.path.realpath(__file__)
inputDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"input")

WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)

HRcat = os.path.join(inputDir,'GaiaHR.h5')
with h5py.File(HRcat,'r') as f: 
    counts = f['counts'][:]
    xedges = f['xedges'][:]
    yedges = f['yedges'][:]

kow = []
nquery = 10
cnt = 0
while cnt < nquery:
    try:
        kow_features = Kowalski(username=opts.user, password=opts.pwd)

        TIMEOUT = 60
        protocol, host, port = "https", "gloria.caltech.edu", 443
        kow_lcs = Kowalski(username=opts.user, password=opts.pwd,
                           timeout=TIMEOUT,
                           protocol=protocol, host=host, port=port)
        break
    except:
        time.sleep(5)
    cnt = cnt + 1
if cnt == nquery:
    raise Exception('Kowalski connection failed...')

if opts.doSubjectSet:
    zoo = ZooProject(username=opts.zooniverse_user,
                     password=opts.zooniverse_pwd,
                     project_id=opts.zooniverse_id) 

if ".h5" in catalogPath:
    df = pd.read_hdf(catalogPath, 'df')
elif ".fits" in catalogPath:
    tab = Table.read(catalogPath, format='fits')
    df = tab.to_pandas()
    df.set_index('objid',inplace=True)

    df = df[(df.prob >= opts.threshold_min) & (df.prob <= opts.threshold_max)]

elif ".csv" in catalogPath:
    tab = Table.read(catalogPath, format='csv')

    objids = []
    for tt, row in enumerate(tab):
        lightcurves_all = get_kowalski(row["RA"], row["Dec"], kow,
                                       min_epochs=20)
        nmax, key_to_keep = -1, -1
        for ii, key in enumerate(lightcurves_all.keys()):
            lc = lightcurves_all[key]
            if len(lc["fid"]) > nmax:
                nmax, key_to_keep = len(lc["fid"]), key
        objids.append(int(key_to_keep))
    objids = np.array(objids)
    tab["objid"] = objids
    tab.add_index('objid')

    df = tab.to_pandas()

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

if opts.doIntersection:
    col = df1.columns[0]
    idx = np.array(np.argsort(df1[col])[::-1])
    np.random.shuffle(idx)
    idx = idx.astype(int)[:opts.Nexamples]

    idy = df.index.intersection(df1.iloc[idx].index)
    df = df.loc[idy]
    df1 = df1.iloc[idx]
else:
    idx = np.random.choice(np.arange(len(df)), size=opts.Nexamples)
    idx = idx.astype(int)
    df = df.iloc[idx]

fs = 24
colors = ['g','r','y']
symbols = ['x', 'o', '^']
fids = [1,2,3]
bands = {1: 'g', 2: 'r', 3: 'i'}

if opts.doSubjectSet:
   image_list, metadata_list, subject_set_name = [], [], intersectionType 
   subject_set_name = subject_set_name + "_" + opts.tag
   #subject_set_name = "labeling_guide_2"
   #subject_set_name = "mira_catalog"
   #subject_set_name = "base"
   #subject_set_name = "two"

objfile = os.path.join(plotDir, 'objids.dat')
objfid = open(objfile, 'w')
for ii, (index, row) in enumerate(df.iterrows()): 
    if np.mod(ii,100) == 0:
        print('Loading %d/%d'%(ii,len(df)))

    if index < 0: continue

    if opts.doFakeData:
        nsample = 100
        objid = index
        ra, dec = 10.0, 60.0
        period, amp = 0.5, 1.0
        lightcurves_all = {}
        lightcurves, absmags, bp_rps = [], [], []
        for fid in fids:
            lc = {}
            lc["name"] = bands[fid]
            lc["hjd"] = np.random.uniform(low=0, high=365, size=(nsample,))
            lc["mag"] = np.random.uniform(low=18, high=19, size=(nsample,))
            lc["magerr"] = np.random.uniform(low=0.01, high=0.1, size=(nsample,))           
            lc["fid"] = fid*np.ones(lc["hjd"].shape)
            lc["ra"] = ra*np.ones(lc["hjd"].shape)
            lc["dec"] = dec*np.ones(lc["hjd"].shape)
            lc["absmag"] = [np.nan, np.nan, np.nan]
            lc["bp_rp"] = np.nan
            lc["parallax"] = np.nan

            lightcurves_all[fid] = lc
            lightcurves.append([lc["hjd"], lc["mag"], lc["magerr"]])
            absmags.append(lc["absmag"])
            bp_rps.append(lc["bp_rp"])
 
    else:
        objid, features = get_kowalski_features_objids([index], kow_features,
                                                       dbname="ZTF_source_features_20191101", featuresetname='limited')

        try:
            period = features.period.values[0]
            ra, dec = features.ra.values[0], features.dec.values[0]
        except:
            period = row["p"]
        amp = -1

        lightcurves = get_kowalski(ra, dec, kow_lcs, min_epochs=20, radius=2.0)

    key = list(lightcurves.keys())[0]

    hjd, magnitude, err = lightcurves[key]["hjd"], lightcurves[key]["mag"], lightcurves[key]["magerr"]
    absmag, bp_rp = lightcurves[key]["absmag"], lightcurves[key]["bp_rp"]
    gaia = gaia_query(ra, dec, 5/3600.0)
    d_pc, gofAL = None, None
    if gaia:
        Plx = gaia['Plx'].data.data[0] # mas
        gofAL = gaia["gofAL"].data.data[0]
        # distance in pc
        if Plx > 0 :
            d_pc = 1 / (Plx*1e-3)

    photFile = os.path.join(jsonDir,'%d.json' % index)
    if opts.doLCFile and not os.path.isfile(photFile):
        data_json = {}
        data_json["data"] = {}
        data_json["data"]["scatterPlot"] = {}
        data_json["data"]["scatterPlot"]["data"] = []
        data_json["data"]["scatterPlot"]["chartOptions"] = {"xAxisLabel": "Days", "yAxisLabel": "Brightness"}

        data_json["data"]["barCharts"] = {}
        data_json["data"]["barCharts"]["period"] = {}
        data_json["data"]["barCharts"]["period"]["data"] = []
        data_json["data"]["barCharts"]["period"]["chartOptions"] = {"xAxisLabel": "log Period", "yAxisLabel": "", "yAxisDomain": [-2.5, 3]}
        data_json["data"]["barCharts"]["amplitude"] = {}
        data_json["data"]["barCharts"]["amplitude"]["data"] = []
        data_json["data"]["barCharts"]["amplitude"]["chartOptions"] = {"xAxisLabel": "Amplitude", "yAxisLabel": "", "yAxisDomain": [0, 3]}

        periods, amplitudes = [], []
        for jj, (fid, color, symbol) in enumerate(zip(fids, colors, symbols)):
            if fid == 3: continue

            seriesData = []
            nmax, amp_tmp = -1, amp
            for ii, key in enumerate(lightcurves.keys()):
                lc = lightcurves[key]
                if not lc["fid"][0] == fid: continue
                idx = np.where(lc["fid"][0] == fids)[0]

                tt = lc["hjd"] - Time('2018-01-01T00:00:00',
                                      format='isot', scale='utc').jd
                for x, y, yerr in zip(tt, lc["mag"], lc["magerr"]):
                    data_single = {"x": x, "y": np.median(lc["mag"])-y,
                                   "y_error": yerr}
                    seriesData.append(data_single)
       
                if len(lc["fid"]) > nmax:
                    nmax = len(lc["fid"])
                    amp_tmp = np.diff(np.percentile(lc["mag"], (5,95)))[0]

            if len(seriesData) == 0: continue

            if fid == 1:
                label, color = "g", "#66CDAA"
            elif fid == 2:
                label, color = "r", "#DC143C"
            elif fid == 3:
                label, color = "i", "#DAA520"

            seriesOptions = {"color": color,
                             "label": label,
                             "period": period}
            periodOptions = {"color": color,
                             "label": label,
                             "value": np.log10(period)}
            amplitudeOptions = {"color": color,
                                "label": label,
                                "value": amp_tmp}

            data_json["data"]["scatterPlot"]["data"].append({"seriesData": seriesData, "seriesOptions": seriesOptions})
            data_json["data"]["barCharts"]["period"]["data"].append(periodOptions)
            data_json["data"]["barCharts"]["amplitude"]["data"].append(amplitudeOptions)
 
        with open(photFile, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)

    pngfile = os.path.join(plotDir,'%d.png' % index)
    if opts.doPlots and not os.path.isfile(pngfile):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))
        plt.axes(ax1)
        bands_count = np.zeros((len(fids),1))
        for jj, (fid, color, symbol) in enumerate(zip(fids, colors, symbols)):
            for ii, key in enumerate(lightcurves.keys()):
                lc = lightcurves[key]
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

        xextent = xedges[-1] - xedges[0]
        yextent = yedges[-1] - yedges[0]
        ax2.imshow(counts.T, interpolation='nearest', origin='lower', 
                  cmap = trunc_cmap, norm=LogNorm(), 
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  aspect=xextent/yextent,
                  zorder=-100)        
 
        if not np.isnan(bp_rp[0]) or not np.isnan(absmag[0]):
            for c_num in range(6):
            
                cont_color = color_dict['mustard'] + '{:d}'.format(99 - 10*c_num)
                top_ellipse = arc_patch((bp_rp[0],absmag[0]), 
                                              bp_rp[1]*(c_num+1)/2, 
                                              absmag[2]*(c_num+1)/2, 
                                              color = cont_color, 
                                              zorder=-2*c_num)
                ax2.add_artist(top_ellipse)
                bottom_ellipse = arc_patch((bp_rp[0],absmag[0]), 
                                             bp_rp[1]*(c_num+1)/2, 
                                             absmag[1]*(c_num+1)/2,
                                             theta1 = 180, theta2 = 360, 
                                             color = cont_color, 
                                              zorder=-2*c_num)
                ax2.add_artist(bottom_ellipse)


        ax2.set_xlim([-1,5.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.tick_params(which="both", top=True, right=True)
        ax2.yaxis.set_major_locator(MultipleLocator(4))
        ax2.yaxis.set_minor_locator(MultipleLocator(2))
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        
        ax2.set_ylabel(r'Luminosity $\;\longrightarrow$', fontsize=30)
        ax2.set_xlabel(r'$\;\longleftarrow$ Temperature', fontsize=30)

        plt.tight_layout()
        pngfile = os.path.join(plotDir,'%d.png' % index)
        fig.savefig(pngfile, bbox_inches='tight')
        plt.close()

    pngfile_HR = os.path.join(plotDir,'%d_HR.png' % index)
    if opts.doPlots and not os.path.isfile(pngfile_HR):

        fig = plt.figure(figsize=(10,10))
        ax = plt.gca()

        ax.imshow(counts.T, interpolation='nearest', origin='lower', 
                  cmap = trunc_cmap, norm=LogNorm(), 
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  aspect=xextent/yextent,
                  zorder=-100)        
 
        if not np.isnan(bp_rp[0]) or not np.isnan(absmag[0]):
            for c_num in range(6):
            
                cont_color = color_dict['mustard'] + '{:d}'.format(99 - 10*c_num)
                top_ellipse = arc_patch((bp_rp[0],absmag[0]), 
                                              bp_rp[1]*(c_num+1)/2, 
                                              absmag[2]*(c_num+1)/2, 
                                              color = cont_color, 
                                              zorder=-2*c_num)
                ax.add_artist(top_ellipse)
                bottom_ellipse = arc_patch((bp_rp[0],absmag[0]), 
                                             bp_rp[1]*(c_num+1)/2, 
                                             absmag[1]*(c_num+1)/2,
                                             theta1 = 180, theta2 = 360, 
                                             color = cont_color, 
                                              zorder=-2*c_num)
                ax.add_artist(bottom_ellipse)

        ax.set_xlim([-1,5.0])
        ax.set_ylim([-5,18])
        ax.invert_yaxis()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(which="both", top=True, right=True)
        ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_ylabel(r'Luminosity $\;\longrightarrow$', fontsize=30)
        ax.set_xlabel(r'$\;\longleftarrow$ Temperature', fontsize=30)

        plt.tight_layout()
        fig.savefig(pngfile_HR, bbox_inches='tight')
        plt.close()

    objfid.write('%d %.10f %.10f %.10f\n' % (index, ra, dec, period))
    print('%d %.10f %.10f %.10f' % (index, ra, dec, period))

    if opts.doSubjectSet:
        #image_list.append(pngfile)
        image_list.append({"image_png_1": pngfile, 
                           "application_json": photFile,
                           "image_png_2": pngfile_HR})

        mdict = {'candidate': int(index),
                 'ra': ra, 'dec': dec, 
                 'period': period}
        metadata_list.append(mdict)
objfid.close()

if opts.doSubjectSet:
    #ret = zoo.add_new_subject(image_list,
    #                          metadata_list,
    #                          subject_set_name=subject_set_name)

    if ".csv" in catalogPath: 
        if "class" in tab.colnames and "comments" in tab.colnames:
            class_dict = {"Non-Periodic (eruptive): CV – SU UMa": "SU_UMa", 
                          "Periodic (pulsating): RRLc": "RRLc",
                          "Periodic (pulsating): RR Lyrae DM": "RRLd",
                          "Periodic (pulsating): LSP": "LSP",
                          "Periodic (rotating): RS CVn": "RS_CVn",
                          "Non-Periodic (stochastic): RCB": "R_Cor_Bor",
                          "Periodic (pulsating): SX Phe": "SX_Phe",
                          "Periodic (pulsating): PopII Cepheid": "PopII_Ceph",
                          "Non-periodic (stochastic): ClassT Tauri": "CTTS",
                          "Periodic (eclipsing): Ellipsoidal": "Ellipsoidal",
                          "Periodic (pulsating): RV Tauri": "RV_Tauri",
                          "Periodic (rotating): Weak-line T Tauri": "WTTS",
                          "Non-Periodic (stochastic): Herbig AeBe": "Herbig",
                          "Non-Periodic (eruptive): CV – U Gem": "U_Gem",
                          "Non-Periodic (eruptive): CV – Z Cam": "Z_Cam"}
            for var_type in np.unique(tab["class"]):
                subject_set_name = class_dict[var_type]
                ssn = subject_set_name + "_" + opts.tag
                this_type = np.where(tab["class"] == var_type)
                ret = zoo.add_new_subject_timeseries(image_list[this_type],
                                                     metadata_list[this_type],
                                                     subject_set_name=ssn)
    
    else:
        ret = zoo.add_new_subject_timeseries(image_list,
                                             metadata_list,
                                             subject_set_name=subject_set_name)

