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

from astroquery.vizier import Vizier

from gatspy.periodic import LombScargle, LombScargleFast

from penquins import Kowalski

LOGIN_URL = "https://irsa.ipac.caltech.edu/account/signon/login.do"
meta_baseurl="https://irsa.ipac.caltech.edu/ibe/search/ztf/products/"
data_baseurl="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/"

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

def gaia_query(ra_deg, dec_deg, rad_deg, maxmag=25,
               maxsources=1):
    """
    Query Gaia DR1 @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS',
                             'phot_g_mean_mag','phot_r_mean_mag',
                             'Plx', 'e_Plx', 'BP-RP'],
                    column_filters={"phot_g_mean_mag":
                                    ("<%f" % maxmag),
                                   "phot_r_mean_mag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    return vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="I/345/gaia2")[0]

def ps1_query(ra_deg, dec_deg, rad_deg, maxmag=25,
               maxsources=1):
    """
    Query Pan-STARRS @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RAJ2000', 'DEJ2000',
                             'gmag','rmag','imag','zmag','ymag'],
                    column_filters={"gmag":
                                    ("<%f" % maxmag),
                                   "imag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    return vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="II/349/ps1")[0]

def get_cookie(username, password):
    """Get a cookie from the IPAC login service
    Parameters
    ----------
    username: `str`
        The IPAC account username
    password: `str`
        The IPAC account password
    """
    url = "%s?josso_cmd=login&josso_username=%s&josso_password=%s" % (LOGIN_URL, username, password)
    response = requests.get(url)
    cookies = response.cookies
    return cookies

def load_file(url, localdir = "/tmp", auth = None, chunks=1024, outf=None, showpbar=False):
    """Load a file from the specified URL and save it locally.

    Parameters
    ----------
    url : `str`
        The URL from which the file is to be downloaded
    localdir : `str`
        The local directory to which the file is to be saved
    auth : tuple
        A tuple of (username, password) to access the ZTF archive
    chunks: `int`
        size of chunks (in Bytes) used to write file to disk.
    outf : `str` or None
        if not None, the downloaded file will be saved to fname,
        overwriting the localdir option.
    showpbar : `bool`
        if True, use tqdm to display the progress bar for the current download.
    """
    cookies = get_cookie(auth[0], auth[1])
    response = requests.get(url, stream = True, cookies = cookies)
    response.raise_for_status()
    file = '%s/%s' % (localdir, url[url.rindex('/') + 1:])
    if not outf is None:
        file = outf
    with open(file, 'wb') as handle:
        iterator = response.iter_content(chunks)
        if showpbar:
            size = int(response.headers['Content-length'])
            iterator = tqdm.tqdm(
                iterator, total=int(size/chunks), unit='KB', unit_scale=False, mininterval=0.2)
        for block in iterator:
            handle.write(block)
    return os.stat(file).st_size

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    return c

def get_kowalski(ra, dec, user, pwd, radius = 5.0):

    k = Kowalski(username=user, password=pwd)

    qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(ra,dec), "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20181220": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1}" } } }  
    r = k.query(query=qu)

    hjd, mag, magerr = [], [], []
    key = list(r["result_data"].keys())[0]
    data = r["result_data"][key]
    key = list(data.keys())[0]
    data = data[key][0]["data"]
    print(data)
    for dic in data:
        hjd.append(dic["hjd"])
        mag.append(dic["mag"])
        magerr.append(dic["magerr"])

    return np.array(hjd), np.array(mag), np.array(magerr)

def get_lightcurve(dataDir, ra, dec, filt):

    directory="%s/*/*/*"%dataDir
    lightcurve = [] 

    querystr="?POS=%.4f,%.4f"%(ra, dec)
    querystr+="&ct=csv"
    url=os.path.join(meta_baseurl, 'sci')
    tmpfile="tmp.tbl"
    load_file(url+querystr, outf=tmpfile, auth=(user, pwd), showpbar=True)
    
    data = ascii.read(tmpfile)  
    print(tmpfile)

    for f in glob.iglob(directory):
        fsplit = f.split("/")[-1].split("_")
        fieldid = int(fsplit[1])
        filtim = fsplit[2][1]
        ccdid = int(fsplit[3][1:])
        qid = int(fsplit[4][1])

        if not filt == filtim: continue
        idx = np.where((data["field"]==fieldid) &
                       (data["ccdid"]==ccdid) &
                       (data["qid"]==qid))[0]
        if len(idx) == 0: continue
        print('Checking %s ...'%f)

        with tables.open_file(f) as store:
            for tbl in store.walk_nodes("/", "Table"):
                if tbl.name in ["sourcedata", "transientdata"]:
                    group = tbl._v_parent
                    continue
            srcdata = pd.DataFrame.from_records(group.sourcedata[:])
            srcdata.sort_values('matchid', axis=0, inplace=True)
            exposures = pd.DataFrame.from_records(group.exposures[:])
            merged = srcdata.merge(exposures, on="expid")

            dist = haversine_np(merged["ra"], merged["dec"], ra, dec)
            idx = np.argmin(dist)
            row = merged.iloc[[idx]]
            df = merged[merged['matchid'] == row["matchid"].values[0]]
            return df

    return lightcurve    

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
    hjd, mag, magerr = get_kowalski(opts.ra, opts.declination, opts.user, opts.pwd)
    # KEVIN ADD hjd to mjd function
    if mjd.size == 0:
        print("No data available...")
        exit(0)

elif opts.lightcurve_source == "matchfiles":
    df = get_lightcurve(dataDir, opts.ra, opts.declination, opts.filt)
    mag = df.psfmag.values
    magerr = df.psfmagerr.values
    flux = df.psfflux.values
    fluxerr=df.psffluxerr.values
    mjd = df.mjd.values

    if len(df) == 0:
        print("No data available...")
        exit(0)

ls = LombScargleFast(silence_warnings=True)
#ls = LombScargle()
#ls.optimizer.period_range = (0.001,0.1)
mjddiff = np.max(mjd) - np.min(mjd)
ls.optimizer.period_range = (1,mjddiff)
ls.fit(mjd,mag,magerr)
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
LCfit = fdecomp.fit_best(np.c_[mjd,mag,magerr],period,5,plotname=False)


if opts.doPlots:
    plotName = os.path.join(path_out_dir,'phot.pdf')
    plt.figure(figsize=(12,8))
    plt.errorbar(mjd-mjd[0],mag,yerr=magerr,fmt='bo')
    fittedmodel = fdecomp.make_f(period)
    plt.plot(mjd-mjd[0],fittedmodel(mjd,*LCfit),'k-')
    plt.xlabel('Time from %.5f [days]'%mjd[0])
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
        mjd_mod = np.mod(mjd, phase)/phase
        plotName = os.path.join(path_out_dir,'phase.pdf')
        plt.figure(figsize=(12,8))
        plt.errorbar(mjd_mod,mag,yerr=magerr,fmt='bo')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude [ab]')
        plt.title('%.5f'%phase)
        plt.gca().invert_yaxis()
        plt.savefig(plotName)
        plt.close()
