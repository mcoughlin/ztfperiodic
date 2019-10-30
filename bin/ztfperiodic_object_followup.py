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
import h5py
import glob

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

from astropy.io import ascii
from astropy import units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
import astropy.io.fits

import requests

import ztfperiodic
from ztfperiodic import fdecomp
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query
from ztfperiodic.utils import load_file
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_lightcurve
from ztfperiodic.utils import combine_lcs
from ztfperiodic.periodsearch import find_periods
from ztfperiodic.specfunc import correlate_spec, adjust_subplots_band, tick_function

from gatspy.periodic import LombScargleFast

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--doGPU", action="store_true", default=False)
    parser.add_option("--doCPU", action="store_true", default=False)
    parser.add_option("--doSaveMemory", action="store_true", default=False)

    parser.add_option("--dataDir", default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/")
    parser.add_option("-o", "--outputDir", default="../output")
    parser.add_option("-i", "--inputDir", default="../input")

    parser.add_option("-a", "--algorithms", default="LS")

    parser.add_option("-r", "--ra", default=237.3234518, type=float)
    parser.add_option("-d", "--declination", default=39.8249067, type=float)
    parser.add_option("-f", "--filt", default="r")

    parser.add_option("-u", "--user")
    parser.add_option("-w", "--pwd")

    parser.add_option("--doPlots", action="store_true", default=False)
    parser.add_option("--doJustHR", action="store_true", default=False)
    parser.add_option("--doOverwrite", action="store_true", default=False)
    parser.add_option("--doSpectra", action="store_true", default=False)
    parser.add_option("--doPeriodSearch", action="store_true", default=False)

    parser.add_option("--doPhase", action="store_true", default=False)
    parser.add_option("-p", "--phase", default=0.016666, type=float)

    parser.add_option("-l", "--lightcurve_source", default="Kowalski")
 
    parser.add_option("--program_ids", default="1,2,3")
    parser.add_option("--min_epochs", default=1, type=int)

    parser.add_option("-n", "--nstack", default=1, type=int)
    parser.add_option("--objid", type=int)

    parser.add_option("--doRemoveHC", action="store_true", default=False)

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
algorithms = opts.algorithms.split(",")
program_ids = list(map(int,opts.program_ids.split(",")))
min_epochs = opts.min_epochs

scriptpath = os.path.realpath(__file__)
starCatalogDir = os.path.join("/".join(scriptpath.split("/")[:-2]),"catalogs")

WDcat = os.path.join(starCatalogDir,'GaiaHRSet.hdf5') # 993635 targets
with h5py.File(WDcat, 'r') as f:
    gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
    parallax = f['parallax'][:]
absmagWD=gmag+5*(np.log10(np.abs(parallax))-2)

if not opts.objid is None:
    path_out_dir='%s/%.5f_%.5f/%d'%(outputDir, opts.ra, 
                                    opts.declination, opts.objid)
else:
    path_out_dir='%s/%.5f_%.5f'%(outputDir, opts.ra,
                                    opts.declination)

if opts.doOverwrite:
    rm_command = "rm -rf %s"%path_out_dir
    os.system(rm_command)

if not os.path.isdir(path_out_dir):
    os.makedirs(path_out_dir)

if opts.doJustHR:
    gaia = gaia_query(opts.ra, opts.declination, 5/3600.0)

    if opts.doPlots:
        bp_rp, absmag = gaia['BP-RP'], gaia['Gmag'] + 5*(np.log10(gaia['Plx']) - 2)
        plotName = os.path.join(path_out_dir,'gaia.pdf')
        plt.figure(figsize=(12,12))
        hist2 = plt.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
        plt.plot(bp_rp,absmag,'x', c='r',zorder=1,markersize=20)
        plt.xlim([-1,4.0])
        plt.ylim([-5,18])
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(hist2[3])
        cbar.set_label('Object Count')
        plt.xlabel('Gaia BP - RP color')
        plt.ylabel('Gaia G absolute magnitude')
        plt.savefig(plotName)
        plt.close()
    exit(0)

coord = SkyCoord(ra=opts.ra*u.degree, dec=opts.declination*u.degree, frame='icrs')

spectral_data = {}
if opts.doSpectra:
    xid = SDSS.query_region(coord, spectro=True)
    if not xid is None:
        spec = SDSS.get_spectra(matches=xid)[0]
        for ii, sp in enumerate(spec):
            try:
                sp.data["loglam"]
            except:
                continue
            lam = 10**sp.data["loglam"]
            flux = sp.data["flux"]
            key = len(list(spectral_data.keys()))
            spectral_data[key] = {}
            spectral_data[key]["lambda"] = lam
            spectral_data[key]["flux"] = flux

    lamostpage = "http://dr5.lamost.org/spectrum/png/"
    lamostfits = "http://dr5.lamost.org/spectrum/fits/"
   
    LAMOSTcat = os.path.join(starCatalogDir,'lamost.hdf5') # 4,209,894 rows
    with h5py.File(LAMOSTcat, 'r') as f:
        lamost_ra, lamost_dec = f['ra'][:], f['dec'][:]
    LAMOSTidxcat = os.path.join(starCatalogDir,'lamost_indices.hdf5')
    with h5py.File(LAMOSTidxcat, 'r') as f:
        lamost_obsid = f['obsid'][:]
        lamost_inverse = f['inverse'][:]
    lamost = SkyCoord(ra=lamost_ra*u.degree, dec=lamost_dec*u.degree, frame='icrs')
    sep = coord.separation(lamost).deg # cross match a certain object with LAMOST
    idx = np.argmin(sep)
    if sep[idx] < 3.0/3600.0:
        idy = np.where(idx == lamost_inverse)[0]
        obsids = lamost_obsid[idy]
        # for each obsid, download the spectra file (a fits file + a png file)
        for obsid in obsids:
            requestpage = "%s/%d" % (lamostpage, obsid)
            plotName = os.path.join(path_out_dir,'lamost_%d.png' % obsid)
            wget_command = "wget %s -O %s" % (requestpage, plotName)
            os.system(wget_command)
            lamost_im = plt.imread(plotName)
            
            requestpage = "%s/%d" % (lamostfits, obsid)
            fitsName = os.path.join(path_out_dir,'lamost_%d.fits.gz' % obsid)
            wget_command = "wget %s -O %s" % (requestpage, fitsName)
            os.system(wget_command)
    
            hdul = astropy.io.fits.open(fitsName)
            for ii, sp in enumerate(hdul):
                lam = sp.data[2,:]
                flux = sp.data[0,:]
                key = len(list(spectral_data.keys()))
                spectral_data[key] = {}
                spectral_data[key]["lambda"] = lam
                spectral_data[key]["flux"] = flux

if opts.doPlots and len(list(spectral_data.keys()))>0:
    plotName = os.path.join(path_out_dir,'spectra.pdf')
    plt.figure(figsize=(12,12))
    for key in spectral_data:
        plt.plot(spectral_data[key]["lambda"],spectral_data[key]["flux"],'--')
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    plt.savefig(plotName)
    plt.close()

# Gaia and PS1 
ps1 = ps1_query(opts.ra, opts.declination, 5/3600.0)

if opts.lightcurve_source == "Kowalski":
    kow = Kowalski(username=opts.user, password=opts.pwd)
    lightcurves_all = get_kowalski(opts.ra, opts.declination, kow, 
                                   oid=opts.objid,
                                   program_ids=program_ids,
                                   min_epochs=min_epochs)
    lightcurves_combined = combine_lcs(lightcurves_all)

    if len(lightcurves_all.keys()) == 0:
        print("No objects... sorry.")
        exit(0)
    key = list(lightcurves_combined.keys())[0]
    
    hjd, mag, magerr = lightcurves_combined[key]["hjd"], lightcurves_combined[key]["mag"], lightcurves_combined[key]["magerr"]
    fid = lightcurves_combined[key]["fid"]
    ra, dec = lightcurves_combined[key]["ra"], lightcurves_combined[key]["dec"]
    absmag, bp_rp = lightcurves_combined[key]["absmag"], lightcurves_combined[key]["bp_rp"]

    idx = np.argsort(hjd)
    hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
    ra, dec = ra[idx], dec[idx]
    fid = fid[idx]

    if opts.doRemoveHC:
        # remove high cadence observation (30 mins)
        dt = np.diff(hjd)
        idx = np.setdiff1d(np.arange(len(hjd)),
                           np.where(dt < 30.0*60.0/86400.0)[0])
        hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
        ra, dec = ra[idx], dec[idx]
        fid = fid[idx]

    print('RA: %.5f, Dec: %.5f' % (np.median(ra), np.median(dec)))
    filt_str = " ".join([str(x) for x in list(np.unique(fid))])
    print("Filters:  %s" % filt_str)
    print("Number of observations: %d" % len(ra))

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


if opts.doPlots:
    plotName = os.path.join(path_out_dir,'gaia.pdf')
    plt.figure(figsize=(12,12))
    asymmetric_error = [[absmag[1]], [absmag[2]]]
    hist2 = plt.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
    if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
        plt.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                     zorder=1,fmt='.r',markersize=20)
    plt.xlim([-1,4.0])
    plt.ylim([-5,18])
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(hist2[3])
    cbar.set_label('Object Count')
    plt.xlabel('Gaia BP - RP color')
    plt.ylabel('Gaia G absolute magnitude')
    plt.savefig(plotName)
    plt.close()


if opts.doJustHR:
    exit(0)


if not (opts.doCPU or opts.doGPU):
    print("--doCPU or --doGPU required")
    exit(0)

lightcurves = []
lightcurve=(hjd,mag,magerr)
lightcurves.append(lightcurve)

ls = LombScargleFast(silence_warnings=True)
hjddiff = np.max(hjd) - np.min(hjd)
ls.optimizer.period_range = (0.1, hjddiff)
ls.fit(hjd,mag,magerr)

# https://github.com/astroML/gatspy/blob/master/examples/FastLombScargle.ipynb
oversampling = 2
N = len(hjd)
df = 1. / (oversampling * hjddiff) # frequency grid spacing
fmin = 1 / hjddiff
fmax = 10 # minimum period is 0.05 d
Nf = (fmax - fmin) // df
freqs = fmin + df * np.arange(Nf)
periods = 1 / freqs
powers = ls._score_frequency_grid(fmin, df, Nf)
ind_best = np.argsort(powers)[-1]
period = periods[ind_best]
power = powers[ind_best]

# fit the lightcurve with fourier components, using BIC to decide the optimal number of pars
LCfit = fdecomp.fit_best(np.c_[hjd,mag,magerr], period, 5, plotname=False)

if opts.doPlots:
    photFile = os.path.join(path_out_dir,'phot.dat')
    filed = open(photFile,'w')
    for a, b, c in zip(hjd, mag, magerr):
        filed.write('%s %.10f %.10f\n' % (a, b, c))
    filed.close()

    plotName = os.path.join(path_out_dir,'phot.pdf')
    plt.figure(figsize=(12,8))
    plt.errorbar(hjd-hjd[0],mag,yerr=magerr,fmt='ko')
    fittedmodel = fdecomp.make_f(period)
    plt.plot(hjd-hjd[0],fittedmodel(hjd,*LCfit),'k-')
    ymed = np.nanmedian(mag)
    y10, y90 = np.nanpercentile(mag,10), np.nanpercentile(mag,90)
    ystd = np.nanmedian(magerr)
    ymin = y10 - 3*ystd
    ymax = y90 + 3*ystd
    plt.ylim([ymin,ymax])
    plt.xlabel('Time from %.5f [days]'%hjd[0])
    plt.ylabel('Magnitude [ab]')
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()

    colors = ['g','r','k']
    fids = [1,2,3]
    plotName = os.path.join(path_out_dir,'phot_color.pdf')
    plt.figure(figsize=(12,8))
    for myfid, color in zip(fids, colors):
        for ii, key in enumerate(lightcurves_all.keys()):
            lc = lightcurves_all[key]
            if not lc["fid"][0] == myfid: continue
            plt.errorbar(lc["hjd"]-hjd[0],lc["mag"],yerr=lc["magerr"],fmt='%so' % color)
    plt.xlabel('Time from %.5f [days]'%hjd[0])
    plt.ylabel('Magnitude [ab]')
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()

    plotName = os.path.join(path_out_dir,'periodogram.pdf')
    periods = np.logspace(-3,-1,10000)
    #periods = np.logspace(0,2,10000)
    periodogram = ls.periodogram(periods)
    plt.figure(figsize=(12,8))
    plt.loglog(periods,periodogram)
    if opts.doPhase:
        plt.plot([phase,phase],[0,np.max(periodogram)],'r--')
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.savefig(plotName)
    plt.close()
    
    ###### The overall plot ######
    plotName = os.path.join(path_out_dir, 'overall.pdf')
    if len(spectral_data.keys()) > 0:
        fig = plt.figure(figsize=(16,16))
        gs = fig.add_gridspec(nrows=6, ncols=2)
        ax1 = fig.add_subplot(gs[:3, 0])
        ax2 = fig.add_subplot(gs[:3, 1])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    newphase = (hjd-hjd[0])/(period)%2
    ixg = fid==1
    ixr = fid==2
    ixy = fid==3
    ax1.errorbar(newphase[ixg], mag[ixg], yerr=magerr[ixg], fmt='bo')
    ax1.errorbar(newphase[ixr], mag[ixr], yerr=magerr[ixr], fmt='ro')
    ax1.errorbar(newphase[ixy], mag[ixy], yerr=magerr[ixy], fmt='yo')
    # fittedmodel = fdecomp.make_f(period)
    # ax1.plot(hjd-hjd[0],fittedmodel(hjd,*LCfit),'k-')
    ymed = np.nanmedian(mag)
    y10, y90 = np.nanpercentile(mag,5), np.nanpercentile(mag,95)
    ystd = np.nanmedian(magerr)
    ymin = y10 - 5*ystd
    ymax = y90 + 5*ystd
    ax1.set_ylim([ymin,ymax])
    ax1.set_title("P = %.2f day"%period, fontsize=16)
    ax1.set_xlabel("phase")
    ax1.invert_yaxis()
    
    asymmetric_error = np.atleast_2d([absmag[1], absmag[2]]).T
    hist2 = ax2.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
    if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
        ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                     c='r',zorder=1,fmt='o')
    ax2.set_xlim([-1,4.0])
    ax2.set_ylim([-5,18])
    ax2.invert_yaxis()
    fig.colorbar(hist2[3],ax=ax2)
    
    nspec = len(spectral_data.keys())
    if nspec > 1:
        bands = [[4750.0, 4950.0], [6475.0, 6650.0], [8450, 8700]]
        npairs = int(nspec * (nspec-1)/2)
        v_values = np.zeros((len(bands), npairs))
        v_values_unc = np.zeros((len(bands), npairs))
        for jj, band in enumerate(bands):
            ax = fig.add_subplot(gs[jj+3, 0])
            ax_ = fig.add_subplot(gs[jj+3, 1])
            xmin, xmax = band[0], band[1]
            ymin, ymax = np.inf, -np.inf
            
            for key in spectral_data:
                idx = np.where((spectral_data[key]["lambda"] >= xmin) &
                               (spectral_data[key]["lambda"] <= xmax))[0]
                myflux = spectral_data[key]["flux"][idx]
                # quick-and-dirty normalization
                myflux -= np.median(myflux)
                myflux /= max(abs(myflux))
                y1 = np.nanpercentile(myflux,1)
                y99 = np.nanpercentile(myflux,99)
                ydiff = y99 - y1
                ymintmp = y1 - ydiff
                ymaxtmp = y99 + ydiff
                if ymin > ymintmp:
                    ymin = ymintmp
                if ymaxtmp > ymax:
                    ymax = ymaxtmp
                ax.plot(spectral_data[key]["lambda"][idx], myflux, '--')
                correlation_funcs = correlate_spec(spectral_data, band = band)
                # cross correlation
                if correlation_funcs == {}:
                    pass
                else:
                    if len(correlation_funcs) == 1:
                        yheights = [0.5]
                    else:
                        yheights = np.linspace(0.25,0.75,len(correlation_funcs))
                    for kk, key in enumerate(correlation_funcs):
                        vpeak = correlation_funcs[key]['v_peak']
                        vpeak_unc = correlation_funcs[key]['v_peak_unc']
                        Cpeak = correlation_funcs[key]['C_peak']
                        ax_.plot(correlation_funcs[key]["velocity"], correlation_funcs[key]["correlation"])
                        ax_.plot([vpeak, vpeak], [0, Cpeak], 'k--')
                        ax_.text(500, yheights[kk], "v=%.0f +- %.0f"%(vpeak, vpeak_unc))
                        v_values[jj][kk] = vpeak
                        v_values_unc[jj][kk] = vpeak_unc
            ax.set_ylim([ymin,ymax])
            ax.set_xlim([xmin,xmax])
            ax_.set_ylim([0,1])
            ax_.set_xlim([-1000,1000])
            if jj == len(bands)-1:
                ax.set_xlabel('Wavelength [A]')
                ax_.set_xlabel('Velocity [km/s]')
                adjust_subplots_band(ax, ax_)
            else:
                ax_.set_xticklabels([])
            ax.set_ylabel('Flux')
            if jj == 1:
                ax_.set_ylabel('Correlation amplitude')
                new_tick_locations = np.array([-900, -600, -300, 0, 300, 600, 900])
                axmass = ax_.twiny()
                axmass.set_xlim(ax_.get_xlim())
                axmass.set_xticks(new_tick_locations)
                axmass.set_xticklabels(tick_function(new_tick_locations, period))
                axmass.set_xlabel("f($M$) ("+r'$M_\odot$'+')')
            
        # calculate mass functon
        if npairs==1:
            id_pair = 0
        else:
            # select a pair with:
            # (1) reasonable variance among all band measurements
            stds = np.std(v_values, axis=0)
            if np.sum(stds<50)>=1:
                v_values = v_values[:, stds<50]
                v_values_unc = v_values_unc[:, stds<50]
            # (2) largest (absolute) velosity variation
            vsums = np.sum(abs(v_values), axis=0)
            id_pair = np.where(vsums == max(vsums))[0][0]
        v_adopt = np.median(v_values[:,id_pair])
        id_band = np.where(v_values[:,id_pair]==v_adopt)[0][0]
        v_adopt_unc = v_values_unc[id_band,id_pair]
        K = abs(v_adopt/2.) # [km/s] assuming that the velocity variation is max and min in rv curve
        K_unc = abs(v_adopt_unc/2.) # [km/s]
        P = 2*period # [day] if ellipsodial modulation, amplitude are roughly the same, 
                    # then the photometric period is probably half of the orbital period
        fmass = (K * 100000)**3 * (P*86400) / (2*np.pi*const.G.cgs.value) / const.M_sun.cgs.value
        fmass_unc = 3 * fmass / K * K_unc
        ax2.set_title("f($M$) = %.2f +- %.2f ("%(fmass, fmass_unc)+r'$M_\odot$'+')', fontsize=16)
    plt.savefig(plotName)
    plt.close()

    fig.savefig(plotName, bbox_inches='tight')
    plotName = os.path.join(path_out_dir,'overall.png')
    fig.savefig(plotName, bbox_inches='tight')
    plt.close()

    if opts.doPhase:
        hjd_mod = np.mod(hjd, 2.0*phase)/(2.0*phase)
        idx = np.argsort(hjd_mod)
        hjd_mod = hjd_mod[idx]
        mag_mod = mag[idx]
        magerr_mod = magerr[idx]
        
        if opts.nstack > 1:
            idxs = np.array_split(np.arange(len(hjd_mod)),int(float(len(hjd_mod))/opts.nstack))
            hjd_new, mag_new, magerr_new = [], [], []
            for idx in idxs:
                hjd_new.append(np.mean(hjd_mod[idx]))
                mag_new.append(np.average(mag_mod[idx], weights=magerr_mod[idx]**2))
                magerr_new.append(1/np.sqrt(np.sum(1.0/magerr_mod[idx]**2)))
            hjd_mod, mag_mod, magerr_mod = np.array(hjd_new), np.array(mag_new), np.array(magerr_new)

        plotName = os.path.join(path_out_dir,'phase.pdf')
        plt.figure(figsize=(12,8))
        plt.errorbar(hjd_mod,mag_mod,yerr=magerr_mod,fmt='ko')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude [ab]')
        if not opts.objid is None:
            if opts.objid == 10798192012899:
                plt.ylim([18.1,18.5])
            #elif opts.objid == 10798191008264:
            #    plt.ylim([18.0,17.7])
        plt.gca().invert_yaxis()
        plt.savefig(plotName)
        plt.close()

        plotName = os.path.join(path_out_dir,'phase_color.pdf')
        plt.figure(figsize=(12,8))
        for fid, color in zip(fids, colors):
            for ii, key in enumerate(lightcurves_all.keys()):
                lc = lightcurves_all[key]
                if not lc["fid"][0] == fid: continue
                plt.errorbar(np.mod(lc["hjd"]-hjd[0], 2.0*phase)/(2.0*phase), lc["mag"],yerr=lc["magerr"],fmt='%so' % color)
        plt.xlabel('Time from %.5f [days]'%hjd[0])
        plt.ylabel('Magnitude [ab]')
        plt.gca().invert_yaxis()
        plt.savefig(plotName)
        plt.close()

        plotName = os.path.join(path_out_dir,'overall_color.pdf')
        if len(spectral_data.keys()) > 0:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

        for fid, color in zip(fids, colors):
            for ii, key in enumerate(lightcurves_all.keys()):
                lc = lightcurves_all[key]
                if not lc["fid"][0] == fid: continue
                ax1.errorbar(np.mod(lc["hjd"]-hjd[0], 2.0*phase)/(2.0*phase), lc["mag"],yerr=lc["magerr"],fmt='%so' % color)
        #period2=period
        #ymed = np.nanmedian(magnitude)
        #y10, y90 = np.nanpercentile(magnitude,10), np.nanpercentile(magnitude,90)
        #ystd = np.nanmedian(err)
        #ymin = y10 - 7*ystd
        #ymax = y90 + 7*ystd
        #ax1.set_ylim([ymin,ymax])
        ax1.invert_yaxis()
        asymmetric_error = np.atleast_2d([absmag[1], absmag[2]]).T
        hist2 = ax2.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
        if not np.isnan(bp_rp) or not np.isnan(absmag[0]):
            ax2.errorbar(bp_rp,absmag[0],yerr=asymmetric_error,
                         c='r',zorder=1,fmt='o')
        ax2.set_xlim([-1,4.0])
        ax2.set_ylim([-5,18])
        ax2.invert_yaxis()
        fig.colorbar(hist2[3],ax=ax2)
        if len(spectral_data.keys()) > 0:
            xmin, xmax = 6475.0, 6650.0
            ymin, ymax = -np.inf, np.inf
            for key in spectral_data:
                idx = np.where( (spectral_data[key]["lambda"] >= xmin) &
                                (spectral_data[key]["lambda"] <= xmax))[0]
                y1 = np.nanpercentile(spectral_data[key]["flux"][idx],1)
                y99 = np.nanpercentile(spectral_data[key]["flux"][idx],99)
                ydiff = y99 - y1
                ymintmp = y1 - ydiff
                ymaxtmp = y99 + ydiff
                if ymin < ymintmp:
                    ymin = ymintmp
                if ymaxtmp < ymax:
                    ymax = ymaxtmp
                ax3.plot(spectral_data[key]["lambda"],spectral_data[key]["flux"],'--')
            ax3.set_ylim([ymin,ymax])
            ax3.set_xlim([xmin,xmax])
            ax3.set_xlabel('Wavelength [A]')
            ax3.set_ylabel('Flux')

        fig.savefig(plotName, bbox_inches='tight')
        plotName = os.path.join(path_out_dir,'overall_color.png')
        fig.savefig(plotName, bbox_inches='tight')
        plt.close()

if opts.doPeriodSearch:
    baseline = max(hjd)-min(hjd)
    if baseline<10:
        fmin, fmax = 18, 1440
    else:
        fmin, fmax = 2/baseline, 480
    
    samples_per_peak = 10
    
    df = 1./(samples_per_peak * baseline)
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)
    
    print('Cataloging lightcurves...')
    catalogFile = os.path.join(path_out_dir,'catalog')
    fid = open(catalogFile,'w')
    for algorithm in algorithms:
        periods_best, significances, pdots = find_periods(algorithm, lightcurves, freqs, doGPU=opts.doGPU, doCPU=opts.doCPU)
        period, significance = periods_best[0], significances[0]
        stat = calc_stats(hjd, mag, magerr, period)
    
        fid.write('%s %.10f %.10f ' % (algorithm, period, significance))
        fid.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n"%(stat[0], stat[1], stat[2], stat[3], stat[4], stat[5], stat[6], stat[7], stat[8], stat[9], stat[10], stat[11], stat[12], stat[13], stat[14], stat[15], stat[16], stat[17], stat[18], stat[19], stat[20], stat[21], stat[22], stat[23], stat[24], stat[25], stat[26], stat[27], stat[28], stat[29], stat[30], stat[31], stat[32], stat[33], stat[34], stat[35]))
    fid.close()
