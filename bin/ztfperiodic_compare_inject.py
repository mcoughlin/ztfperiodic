#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py

import numpy as np

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.patches as patches

import astropy
from astropy.table import Table, vstack, hstack
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

from ztfperiodic.utils import angular_distance
from ztfperiodic.utils import convert_to_hex

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("--doField",  action="store_true", default=False)
    parser.add_option("-f","--field",default=853,type=int)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/inject_quadrants_Primary_DR3/catalog/compare")
    parser.add_option("-c","--catalog",default="/home/michael.coughlin/ZTF/inject_quadrants_Primary_DR3/catalog/ECE_ELS_EAOV")

    parser.add_option("--algorithm",default="EAOV")
    parser.add_option("--sig",default=7.0,type=float)

    parser.add_option("-i","--injtype",default="wdb")

    opts, args = parser.parse_args()

    return opts

def read_catalog(catalog_file):

    amaj, amin, phi = None, None, None
    default_err, default_mag = 5.0, np.nan

    if ".dat" in catalog_file:
        lines = [line.rstrip('\n') for line in open(catalog_file)]
        names, ras, decs, errs = [], [], [], []
        periods, classifications, mags = [], [], []
        if ("fermi" in catalog_file):
            amaj, amin, phi = [], [], []
        for line in lines:
            lineSplit = list(filter(None,line.split(" ")))
            if ("blue" in catalog_file) or ("uvex" in catalog_file) or ("xraybinary" in catalog_file):
                ra_hex, dec_hex = convert_to_hex(float(lineSplit[0])*24/360.0,delimiter=''), convert_to_hex(float(lineSplit[1]),delimiter='')
                if dec_hex[0] == "-":
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
                else:
                    objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
                names.append(objname)
                ras.append(float(lineSplit[0]))
                decs.append(float(lineSplit[1]))
            elif ("CRTS" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                errs.append(default_err)
                periods.append(float(lineSplit[3]))
                classifications.append(lineSplit[4])
                mags.append(default_mag)
            elif ("vlss" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                errs.append(err)
            elif ("fermi" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = np.sqrt(float(lineSplit[3])**2 + float(lineSplit[4])**2)*3600.0
                errs.append(err)
                amaj.append(float(lineSplit[3]))
                amin.append(float(lineSplit[4]))
                phi.append(float(lineSplit[5]))
            elif ("swift" in catalog_file) or ("xmm" in catalog_file):
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                err = float(lineSplit[3])
                errs.append(err)
            else:
                names.append(lineSplit[0])
                ras.append(float(lineSplit[1]))
                decs.append(float(lineSplit[2]))
                errs.append(default_err)
        names = np.array(names)
        ras, decs, errs = np.array(ras), np.array(decs), np.array(errs)
        if ("fermi" in catalog_file):
            amaj, amin, phi = np.array(amaj), np.array(amin), np.array(phi)

        columns = ["name", "ra", "dec", "amaj", "amin", "phi"]
        tab = Table([names, ras, decs, amaj, amin, phi],
                    names=columns)

    elif ".hdf5" in catalog_file:
        with h5py.File(catalog_file, 'r') as f:
            ras, decs = f['ra'][:], f['dec'][:]
        catalog_file_mag = catalog_file.replace(".hdf5",
                                                "_mag.hdf5")
        if os.path.isfile(catalog_file_mag):
            with h5py.File(catalog_file_mag, 'r') as f:
                mags = f['mag'][:]

        errs = default_err*np.ones(ras.shape)
        periods = np.zeros(ras.shape)
        classifications = np.zeros(ras.shape)

        names = []
        for ra, dec in zip(ras, decs):
            ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)
        names = np.array(names)

        columns = ["name", "ra", "dec", "period", "classification",
                   "mag"]
        tab = Table([names, ras, decs, periods, classifications,
                     mags],
                     names=columns)

    return tab

def load_catalog(catalog,doFermi=False,doSimbad=False,
                         doField=False,field=-1,
                         algorithm='ECE',
                         injtype="wdb"):

    customSimbad=Simbad() 
    customSimbad.add_votable_fields("otype(V)")
    customSimbad.add_votable_fields("otype(3)")
    customSimbad.add_votable_fields("otype(N)")
    customSimbad.add_votable_fields("otype(S)")

    if doFermi:
        filenames = sorted(glob.glob(os.path.join(catalog,"*/*.dat")))[::-1] + \
                    sorted(glob.glob(os.path.join(catalog,"*/*.h5")))[::-1]
    elif doField:
        filenames = sorted(glob.glob(os.path.join(catalog,"%d_*.dat" % field)))[::-1] + \
                    sorted(glob.glob(os.path.join(catalog,"%d_*.h5" % field)))[::-1]
    else:
        filenames = sorted(glob.glob(os.path.join(catalog,"*.dat")))[::-1] + \
                    sorted(glob.glob(os.path.join(catalog,"*.h5")))[::-1]
                     
    h5names = ["objid", "ra", "dec",
               "stats0", "stats1", "stats2", "stats3", "stats4",
               "stats5", "stats6", "stats7", "stats8", "stats9",
               "stats10", "stats11", "stats12", "stats13", "stats14",
               "stats15", "stats16", "stats17", "stats18", "stats19",
               "stats20", "stats21"] 

    h5periodnames = ["objid", "period", "sig", "pdot", 
                     "periodicstats0", "periodicstats1", "periodicstats2",
                     "periodicstats3", "periodicstats4", "periodicstats5",
                     "periodicstats6", "periodicstats7", "periodicstats8",
                     "periodicstats9", "periodicstats10", "periodicstats11",
                     "periodicstats12", "periodicstats13"]

    if injtype == "wdb":
        h5injnames = ["inj_period", "inj_inclination", "inj_sbratio"]
    elif injtype == "gaussian":
        h5injnames = ["inj_period", "inj_amplitude", "inj_phase"]

    cnt = 0
    #filenames = filenames[:500]
    for ii, filename in enumerate(filenames):
        if np.mod(ii,100) == 0:
            print(filename)
            print('Loading file %d/%d' % (ii, len(filenames)))

        filenameSplit = filename.split("/")
        catnum = filenameSplit[-1].replace(".dat","").replace(".h5","").split("_")[-1]

        try:
            with h5py.File(filename, 'r') as f:
                name = f['names'].value
                filters = f['filters'].value
                stats = f['stats'].value 
                periodic_stats = f['stats_%s' % algorithm].value
                injections = f['injections'].value
        except:
            continue
        data_tmp = Table(rows=stats, names=h5names)
        data_tmp['name'] = name
        data_tmp['filt'] = filters
        data_period_tmp = Table(rows=periodic_stats, names=h5periodnames)
        data_period_tmp.remove_column('objid')
        data_inj_tmp = Table(rows=injections, names=h5injnames)

        data_tmp = hstack([data_tmp, data_period_tmp, data_inj_tmp], join_type='inner')
        if len(data_tmp) == 0: continue

        data_tmp['name'] = data_tmp['name'].astype(str)
        data_tmp['filt'] = data_tmp['filt'].astype(str)
        data_tmp['catnum'] = int(catnum) * np.ones(data_tmp["ra"].shape)

        coord = SkyCoord(data_tmp["ra"], data_tmp["dec"], unit=u.degree)
        simbad = ["N/A"] * len(coord)
        if doSimbad:
            print('Querying simbad: %d/%d' %(ii,len(filenames)))
            doQuery = True
            result_table = None
            nquery = 1
            while doQuery and (not ii==1078):
                try:
                    result_table = customSimbad.query_region(coord,
                                                             radius=2*u.arcsecond)
                    doQuery = False
                    nquery = nquery + 1
                except:
                    nquery = nquery + 1
                    time.sleep(10)
                    continue
                if nquery >= 3:
                    break

            if not result_table is None:
                ra = result_table['RA'].filled().tolist()
                dec = result_table['DEC'].filled().tolist()
    
                ra  = Angle(ra, unit=u.hour)
                dec = Angle(dec, unit=u.deg)
    
                coords2 = SkyCoord(ra=ra,
                                   dec=dec, frame='icrs')
                idx,sep,_ = coords2.match_to_catalog_sky(coord)
                for jj, ii in enumerate(idx):
                    simbad[ii] = result_table[jj]["OTYPE_S"]
        data_tmp['simbad'] = simbad
        data_tmp['simbad'] = data_tmp['simbad'].astype(str)

        if cnt == 0:
            data = copy.copy(data_tmp)
        else:
            data = vstack([data,data_tmp])
        cnt = cnt + 1
 
    if len(data) == 0:
        print('No data in %s available...' % catalog)
        return []

    sig = data["sig"]
    idx = np.arange(len(sig))/len(sig)
    sigsort = idx[np.argsort(sig)]
    data["sigsort"] = sigsort

    return data

# Parse command line
opts = parse_commandline()
outputDir = opts.outputDir
catalog = opts.catalog
injtype = opts.injtype

if opts.doField:
    outputDir = os.path.join(outputDir,str(opts.field))

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

name = opts.algorithm

catfile = os.path.join(outputDir,'catalog_%s.fits' % name)

if not os.path.isfile(catfile):
    cat = load_catalog(opts.catalog,
                       doField=opts.doField,field=opts.field,
                       algorithm=opts.algorithm,
                       injtype=injtype)
    cat.write(catfile, format='fits')
else:
    cat = Table.read(catfile, format='fits')
idx1 = np.where(cat["sig"] >= opts.sig)[0]
print('Keeping %.5f %% of objects in catalog 1' % (100*len(idx1)/len(cat)))
catalog = SkyCoord(ra=cat["ra"]*u.degree, dec=cat["dec"]*u.degree, frame='icrs')

outputDir = os.path.join(outputDir, name)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

if opts.doPlots:

    if injtype == "wdb":
        period_min, period_max = 4.0*60.0/86400.0, 60.0*60.0/86400.0
    elif injtype == "gaussian":
        period_min, period_max = 0.1, 100.0 

    xedges = np.logspace(np.log10(period_min),np.log10(period_max),100)
    yedges = np.logspace(np.log10(period_min),np.log10(period_max),100)

    H, xedges, yedges = np.histogram2d(cat['inj_period'],cat['period'], bins=(xedges, yedges))
    H = H.T  # Let each row list bins with common y range.
    X, Y = np.meshgrid(xedges, yedges)
    #H[H==0] = np.nan

    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',0)
 
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(1,1,1)

    c = plt.pcolormesh(X, Y, H, vmin=1.0,vmax=np.max(H),norm=LogNorm(),
                       cmap=cmap)
    cbar = plt.colorbar(c)
    cbar.set_label('Counts')
    ax.set_xscale('log')
    ax.set_yscale('log')

    vals = np.linspace(np.min(xedges), np.max(xedges), 100)
    plt.plot(vals, vals, 'k--')

    plt.xlim([period_min, period_max])
    plt.ylim([period_min, period_max])
    plt.xlabel('Period [Injected]')
    plt.ylabel('Period [Recovered]')
  
    pdffile = os.path.join(outputDir,'periods.pdf')
    fig.savefig(pdffile, bbox_inches='tight')
    plt.close()

    inj_diff = np.abs(cat['inj_period']-cat['period'])/cat['period']
    print(inj_diff)
    idx1 = np.where( (np.abs(inj_diff)<1e-2) | (np.abs(inj_diff-1)<1e-2) | (np.abs(inj_diff-0.5)<1e-2) )[0]
    idx2 = np.setdiff1d(np.arange(len(inj_diff)), idx1)

    idx3 = np.where( (np.abs(inj_diff)<1e-2) )[0]
    idx4 = np.where( (np.abs(inj_diff-1)<1e-2) )[0]

    period_found = cat['inj_period'][idx1]
    period_missed = cat['inj_period'][idx2]

    bins = np.linspace(0,5,50)
    hist, bin_edges = np.histogram(inj_diff, bins=bins)
    bins = (bin_edges[1:] + bin_edges[:-1])/2.0

    plotName = os.path.join(outputDir, "inj_diff.pdf")
    plt.figure(figsize=(12,8))
    ax = plt.gca()
    plt.plot(bins, hist, color = 'k', linestyle='-', drawstyle='steps')
    plt.xlabel('Relative period [injected vs. recovered]')
    plt.ylabel('Count')
    #ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

    if injtype == "wdb":
        bins = np.logspace(-3,-1,50)
    elif injtype == "gaussian":
        bins = np.logspace(-1,2,50)

    hist1, bin_edges = np.histogram(period_found, bins=bins)
    hist2, bin_edges = np.histogram(period_missed, bins=bins)
    bins = (bin_edges[1:] + bin_edges[:-1])/2.0

    plotName = os.path.join(outputDir, "periods_mf.pdf")
    plt.figure(figsize=(12,8))
    ax = plt.gca()
    plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
             label='Found')
    plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
             label='Missed')
    plt.legend()
    plt.xlabel('Periods [days]')
    plt.ylabel('Count')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

    if injtype == "wdb":
        bins = np.linspace(80,90,50)
        hist1, bin_edges = np.histogram(incl_found, bins=bins)
        hist2, bin_edges = np.histogram(incl_missed, bins=bins)
        bins = (bin_edges[1:] + bin_edges[:-1])/2.0
    
        incl_found = cat['inj_inclination'][idx1]
        incl_missed = cat['inj_inclination'][idx2]
    
        sbratio_found = cat['inj_sbratio'][idx1]
        sbratio_missed = cat['inj_sbratio'][idx2]
        sbratio_correct = cat['inj_sbratio'][idx3]
        sbratio_double = cat['inj_sbratio'][idx4]
    
        plotName = os.path.join(outputDir, "incl_mf.pdf")
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
                 label='Found')
        plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
                 label='Missed')
        plt.legend()
        plt.xlabel('Inclination [deg]')
        plt.ylabel('Count')
        ax.set_yscale('log')
        plt.savefig(plotName, bbox_inches='tight')
        plt.close()
    
        bins = np.linspace(0,1,50)
        hist1, bin_edges = np.histogram(sbratio_found, bins=bins)
        hist2, bin_edges = np.histogram(sbratio_missed, bins=bins)
        bins = (bin_edges[1:] + bin_edges[:-1])/2.0
        
        plotName = os.path.join(outputDir, "sbratio_mf.pdf")
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
                 label='Found')
        plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
                 label='Missed')
        plt.legend()
        plt.xlabel('Surface Brightness Ratio')
        plt.ylabel('Count')
        ax.set_yscale('log')
        plt.savefig(plotName, bbox_inches='tight')
        plt.close()
    
        bins = np.linspace(0,1,50)
        hist1, bin_edges = np.histogram(sbratio_correct, bins=bins)
        hist2, bin_edges = np.histogram(sbratio_double, bins=bins)
        bins = (bin_edges[1:] + bin_edges[:-1])/2.0
    
        plotName = os.path.join(outputDir, "sbratio_correct_double.pdf")
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
                 label='Correct')
        plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
                 label='Double')
        plt.legend()
        plt.xlabel('Surface Brightness Ratio')
        plt.ylabel('Count')
        ax.set_yscale('log')
        plt.savefig(plotName, bbox_inches='tight')
        plt.close()

    elif injtype == "gaussian":

        amplitude_found = cat['inj_amplitude'][idx1]
        amplitude_missed = cat['inj_amplitude'][idx2]
 
        phase_found = cat['inj_phase'][idx1]
        phase_missed = cat['inj_phase'][idx2]
 
        bins = np.logspace(-2,0,50)
        hist1, bin_edges = np.histogram(amplitude_found, bins=bins)
        hist2, bin_edges = np.histogram(amplitude_missed, bins=bins)
        bins = (bin_edges[1:] + bin_edges[:-1])/2.0
 
        plotName = os.path.join(outputDir, "amplitude_mf.pdf")
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
                 label='Found')
        plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
                 label='Missed')
        plt.legend()
        plt.xlabel('Amplitude')
        plt.ylabel('Count')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(plotName, bbox_inches='tight')
        plt.close()

        bins = np.linspace(0,2*np.pi,50)
        hist1, bin_edges = np.histogram(phase_found, bins=bins)
        hist2, bin_edges = np.histogram(phase_missed, bins=bins)
        bins = (bin_edges[1:] + bin_edges[:-1])/2.0
 
        plotName = os.path.join(outputDir, "phase_mf.pdf")
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        plt.plot(bins, hist1, color = 'k', linestyle='-', drawstyle='steps',
                 label='Found')
        plt.plot(bins, hist2, color = 'r', linestyle='--', drawstyle='steps',
                 label='Missed')
        plt.legend()
        plt.xlabel('Phase')
        plt.ylabel('Count')
        ax.set_yscale('log')
        plt.savefig(plotName, bbox_inches='tight')
        plt.close()
