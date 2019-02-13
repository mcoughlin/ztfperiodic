#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:05:27 2017

@author: kburdge
"""

import os, sys
import optparse
import pandas as pd
import numpy as np
import tables
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

import requests
import tqdm
import fdecomp

import ztfperiodic
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query
from ztfperiodic.utils import load_file
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_lightcurve

from gatspy.periodic import LombScargle, LombScargleFast

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--dataDir",default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/")
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-i","--inputDir",default="../input")

    parser.add_option("-r","--ra",default=234.884000,type=float)
    parser.add_option("-d","--declination",default=50.460778,type=float)
    parser.add_option("-f","--filt",default="r")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--doPlots",  action="store_true", default=False)
    parser.add_option("--doJustHR",  action="store_true", default=False)
    parser.add_option("--doOverwrite",  action="store_true", default=False)

    parser.add_option("--doPhase",  action="store_true", default=False)
    parser.add_option("-p","--phase",default=4.736406,type=float)

    parser.add_option("-l","--lightcurve_source",default="matchfiles")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
dataDir = opts.dataDir
outputDir = opts.outputDir
inputDir = opts.inputDir
phase = opts.phase
user = opts.user
pwd = opts.pwd

path_out_dir='%s/%.5f_%.5f'%(outputDir,opts.ra,opts.declination)

if opts.doOverwrite:
    rm_command = "rm -rf %s"%path_out_dir
    os.system(rm_command)

if not os.path.isdir(path_out_dir):
    os.makedirs(path_out_dir)

# Gaia and PS1 
gaia = gaia_query(opts.ra, opts.declination, 5/3600.0)
ps1 = ps1_query(opts.ra, opts.declination, 5/3600.0)

if opts.doPlots:
    gaiaimage = os.path.join(inputDir,'ESA_Gaia_DR2_HRD_Gaia_625.png')
    img=mpimg.imread(gaiaimage)
    img=np.flipud(img)
    plotName = os.path.join(path_out_dir,'gaia.pdf')
    plt.figure(figsize=(12,12))
    plt.imshow(img,origin='lower')

    xval, yval = gaia['BP-RP'], gaia['Gmag'] + 5*(np.log10(gaia['Plx']) - 2)
    xval = 162 + (235-162)*xval/1.0
    yval = 625 + (145-625)*yval/15.0

    plt.plot(xval,yval,'kx')
    plt.savefig(plotName)
    plt.close()

if opts.doJustHR:
    exit(0)

if opts.lightcurve_source == "Kowalski":
    kow = Kowalski(username=opts.user, password=opts.pwd)
    hjd, mag, magerr = get_kowalski(opts.ra, opts.declination, kow)
    if hjd.size == 0:
        print("No data available...")
        exit(0)

elif opts.lightcurve_source == "matchfiles":
    df = get_lightcurve(dataDir, opts.ra, opts.declination, opts.filt, opts.user, opts.pwd)
    mag = df.psfmag.values
    magerr = df.psfmagerr.values
    flux = df.psfflux.values
    fluxerr=df.psffluxerr.values
    hjd = df.hjd.values

    if len(df) == 0:
        print("No data available...")
        exit(0)

ls = LombScargleFast(silence_warnings=True)
#ls = LombScargle()
#ls.optimizer.period_range = (0.001,0.1)
hjddiff = np.max(hjd) - np.min(hjd)
ls.optimizer.period_range = (1,hjddiff)
ls.fit(hjd,mag,magerr)
period = ls.best_period
#phase = period
print("Best period: " + str(period) + " days")

harmonics = np.array([1,2,3,4])*phase
filename = os.path.join(path_out_dir,'harmonics.dat')
fid = open(filename,'w')
for harmonic in harmonics:
    periodogram = ls.periodogram(harmonic)
    fid.write('%.5e %.5e\n'%(harmonic,periodogram))
fid.close()
harmonics = np.loadtxt(filename)

# fit the lightcurve with fourier components, using BIC to decide the optimal number of pars
LCfit = fdecomp.fit_best(np.c_[hjd,mag,magerr],period,5,plotname=False)


if opts.doPlots:
    plotName = os.path.join(path_out_dir,'phot.pdf')
    plt.figure(figsize=(12,8))
    plt.errorbar(hjd-hjd[0],mag,yerr=magerr,fmt='bo')
    fittedmodel = fdecomp.make_f(period)
    plt.plot(hjd-hjd[0],fittedmodel(hjd,*LCfit),'k-')
    plt.xlabel('Time from %.5f [days]'%hjd[0])
    plt.ylabel('Magnitude [ab]')
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()

    plotName = os.path.join(path_out_dir,'periodogram.pdf')
    #periods = np.logspace(-3,-1,10000)
    periods = np.logspace(0,2,10000)
    periodogram = ls.periodogram(periods)
    plt.figure(figsize=(12,8))
    plt.loglog(periods,periodogram)
    if opts.doPhase:
        plt.plot([phase,phase],[0,np.max(periodogram)],'r--')
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.savefig(plotName)
    plt.close()

    plotName = os.path.join(path_out_dir,'harmonics.pdf')
    plt.figure(figsize=(12,8))
    plt.loglog(harmonics[:,0],harmonics[:,1],'bo')
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.savefig(plotName)
    plt.close()

    if opts.doPhase:
        hjd_mod = np.mod(hjd, phase)/phase
        plotName = os.path.join(path_out_dir,'phase.pdf')
        plt.figure(figsize=(12,8))
        plt.errorbar(hjd_mod,mag,yerr=magerr,fmt='bo')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude [ab]')
        plt.title('%.5f'%phase)
        plt.gca().invert_yaxis()
        plt.savefig(plotName)
        plt.close()
