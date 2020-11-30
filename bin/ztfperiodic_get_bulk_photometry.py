#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import pickle
from functools import reduce

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

from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = -1
Simbad.TIMEOUT = 300000

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_external
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.lcstats import sawtooth_decomposition, make_s
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query
from ztfperiodic.utils import galex_query
from ztfperiodic.utils import sdss_query
from ztfperiodic.utils import database_query

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

    parser.add_option("--inputDir", default="../input")

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/wuma_period_minimum")
    #parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2_ids_DR2/catalog/compare/rrlyr/")
    parser.add_option("-c","--catalogPath",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/wuma_period_minimum")

    parser.add_option("--doDifference",  action="store_true", default=False)
    parser.add_option("-d","--differencePath",default="/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.dscu.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.rrlyr.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ea.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.eb.f.h5,/home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/compare/slices/d11.ew.f.h5")

    parser.add_option("--doIntersection",  action="store_true", default=False)
    parser.add_option("-i","--intersectionPath",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/slices_merged/wuma.d12_dnn_v2_20200921.h5.h5")

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    opts, args = parser.parse_args()

    return opts


# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
inputDir = opts.inputDir
catalogPath = opts.catalogPath
differencePath = opts.differencePath   
intersectionPath = opts.intersectionPath

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

catalogDir = os.path.join(outputDir,'catalog')
if not os.path.isdir(catalogDir):
    os.makedirs(catalogDir)

condorDir = os.path.join(outputDir,'condor')
if not os.path.isdir(condorDir):
    os.makedirs(condorDir)

logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

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

if ".h5" in catalogPath:
    try:
        df = pd.read_hdf(catalogPath, 'df')
    except:
        df = pd.read_hdf(catalogPath, 'df_merged')
elif ".fits" in catalogPath:
    tab = Table.read(catalogPath, format='fits')
    df = tab.to_pandas()
    df.set_index('objid',inplace=True)
else:
    filenames = glob.glob(os.path.join(plotDir,'*.png'))
    objids, periods = [], []
    if len(filenames) == 0:
        filedirs = glob.glob(os.path.join(outputDir,'*_*','*-*'))
        for ii, filedir in enumerate(filedirs):
            filenames = glob.glob(os.path.join(filedir,'*.png'))
            for jj, filename in enumerate(filenames):
                if np.mod(jj, 10) == 0:
                    print('Dir %d/%d File %d/%d' % (ii+1,len(filedirs),
                                                    jj+1,len(filenames)))
                filenameSplit = filename.split("/")[-1].split(".png")[0].split("_")
                sig, ra, dec, period, filt = np.array(filenameSplit,
                                                      dtype=float)

                if not opts.sigthresh is None:
                    if sig < opts.sigthresh: continue

                lcs = get_kowalski(ra, dec, kow, radius = 1.0)
                for objid in lcs.keys():
                    objids.append(objid)
                    periods.append(period)

    else:
        for filename in filenames:
            filenameSplit = filename.split("/")[-1].split(".png")[0]
            objids.append(int(filenameSplit))
            periods.append(-1)

    df = pd.DataFrame(data=objids)
    df.columns = ['objid']
    df.set_index('objid',inplace=True)

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

mag = ['AllWISE__w1mpro', 'AllWISE__w2mpro',
       'AllWISE__w3mpro', 'AllWISE__w4mpro',
       'PS1_DR1__gMeanPSFMag',
       'PS1_DR1__rMeanPSFMag',
       'PS1_DR1__iMeanPSFMag',
       'PS1_DR1__zMeanPSFMag',
       'PS1_DR1__yMeanPSFMag']
       
magerr = ['AllWISE__w1sigmpro', 'AllWISE__w2sigmpro', 
          'AllWISE__w3sigmpro', 'AllWISE__w4sigmpro',
          'PS1_DR1__gMeanPSFMagErr',
          'PS1_DR1__rMeanPSFMagErr',
          'PS1_DR1__iMeanPSFMagErr',
          'PS1_DR1__zMeanPSFMagErr',
          'PS1_DR1__yMeanPSFMagErr']

parallax = ['Gaia_DR2__parallax', 'Gaia_DR2__parallax_error']
gaia_mags = ['Gaia_DR2__phot_g_mean_mag', 'Gaia_DR2__phot_bp_mean_mag',
             'Gaia_DR2__phot_rp_mean_mag']

catalogFile = os.path.join(outputDir,"catalog.h5")
if not os.path.isfile(catalogFile):
    objids = np.array(df.index)
    #objids = objids[:1000]
    
    objids, features = get_kowalski_features_objids(objids, kow,
                                                 featuresetname='all')

    snr = features["Gaia_DR2__parallax"]/features["Gaia_DR2__parallax_error"]
    idx = np.where((snr >= 3) & ~np.isnan(snr))[0]
    
    objids = np.array(objids.iloc[idx])
    features = features.iloc[idx]
    
    ra = np.array(features['ra'])
    dec = np.array(features['dec'])
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    galcoords = np.vstack([coord.galactic.l.deg, coord.galactic.b.deg]).T
    
    mag_tmp = np.array(features[mag])
    magerr_tmp = np.array(features[magerr])
    parallax = np.array(features[parallax])
    gaia = np.array(features[gaia_mags]) 

    nobs, ncols = mag_tmp.shape
    mag = np.zeros((nobs, ncols + 2 + 5))
    magerr = np.zeros((nobs, ncols + 2 + 5))     

    mag[:, :ncols] = mag_tmp
    magerr[:, :ncols] = magerr_tmp

    radius = 5
    for kk, (r, d) in enumerate(zip(ra, dec)):

        if np.mod(kk,100) == 0:
            print('Loading %d/%d'%(kk,len(ra)))

        external = get_kowalski_external(r, d, kow, radius = radius)
        FUVmag, NUVmag = external["mag"][9], external["mag"][10]
        e_FUVmag, e_NUVmag = external["magerr"][9], external["magerr"][10]

        if not np.isnan(FUVmag):
            mag[kk, ncols] = FUVmag
        if not np.isnan(NUVmag):
            mag[kk, ncols+1] = NUVmag
        if not np.isnan(e_FUVmag):
            magerr[kk, ncols] = e_FUVmag
        if not np.isnan(e_NUVmag):
            magerr[kk, ncols+1] = e_NUVmag

        sdss = sdss_query(r, d, 5/3600.0)
        if sdss:
            umag = sdss["umag"].data.data[0]
            gmag = sdss["gmag"].data.data[0]
            rmag = sdss["rmag"].data.data[0]
            imag = sdss["imag"].data.data[0] 
            zmag = sdss["zmag"].data.data[0]
            e_umag = sdss["e_umag"].data.data[0]
            e_gmag = sdss["e_gmag"].data.data[0]
            e_rmag = sdss["e_rmag"].data.data[0]
            e_imag = sdss["e_imag"].data.data[0]
            e_zmag = sdss["e_zmag"].data.data[0]

            mag[kk, ncols+2] = umag
            mag[kk, ncols+3] = gmag
            mag[kk, ncols+4] = rmag
            mag[kk, ncols+5] = imag
            mag[kk, ncols+6] = zmag
            magerr[kk, ncols+2] = e_umag
            magerr[kk, ncols+3] = e_gmag
            magerr[kk, ncols+4] = e_rmag
            magerr[kk, ncols+5] = e_imag
            magerr[kk, ncols+6] = e_zmag
 
    with h5py.File(catalogFile, 'w') as hf:
        hf.create_dataset("objids",  data=objids)
        hf.create_dataset("mag",  data=mag)
        hf.create_dataset("magerr",  data=magerr)
        hf.create_dataset("parallax",  data=parallax)
        hf.create_dataset("galcoords", data=galcoords)
        hf.create_dataset("ra", data=ra)
        hf.create_dataset("dec", data=dec)    
        hf.create_dataset("gaia", data=gaia)

with h5py.File(catalogFile, 'r') as f:
    mag, magerr, parallax = f['mag'][:], f['magerr'][:], f['parallax'][:]
    objids, galcoords = f['objids'][:], f['galcoords'][:]
    ra, dec = f['ra'][:], f['dec'][:]   
    gaia = f['gaia'][:]


radius = 5
for r, d in zip(ra, dec):
    #qu = { "query_type": "cone_search", "query": {"object_coordinates": {"radec": {'test': [r,d]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "GALEX": { "filter": "{}", "projection": "{}"} } } }

    #r = database_query(kow, qu, nquery = 10)

    data = get_kowalski_external(r, d, kow, radius = 5.0)
    print(data)

    #qu = { "query_type": "info", "query": {"command": "catalog_names"}}
    #r = database_query(kow, qu, nquery = 1)

print(stop)

if opts.doPlots:

    w1, w2, w3, w4 = mag[:,0], mag[:,1], mag[:,2], mag[:,3]
    idx = np.where(np.not_equal(w1,0) & np.not_equal(w2,0) &
                   np.not_equal(w3,0) & np.not_equal(w4,0))[0]
    w1, w2, w3, w4 = w1[idx], w2[idx], w3[idx], w4[idx]

    plotName = os.path.join(outputDir,'WISE.pdf')
    plt.figure(figsize=(7,5))
    hist2 = plt.hist2d(w1-w2, w3-w4, bins=100,zorder=0,norm=LogNorm())
    plt.xlabel('WISE $w_1-w_2$')
    plt.ylabel('WISE $w_3-w_4$')
    plt.tight_layout()
    plt.savefig(plotName)
    plt.close()

    FUV, NUV = mag[:,9], mag[:,10]
    idx = np.where(np.not_equal(FUV,0) & np.not_equal(NUV,0))[0]
    FUV, NUV = FUV[idx], NUV[idx]

    plotName = os.path.join(outputDir,'galex.pdf')
    plt.figure(figsize=(7,5))
    plt.scatter(FUV, NUV, s=20, c='k')
    plt.xlabel('Galex FUV')
    plt.ylabel('Galex NUV')
    plt.tight_layout()
    plt.savefig(plotName)
    plt.close()

    WDcat = os.path.join(inputDir,'GaiaHRSet.hdf5')
    with h5py.File(WDcat, 'r') as f:
        gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
        parallaxWD = f['parallax'][:]
    absmagWD=gmag+5*(np.log10(np.abs(parallaxWD))-2)

    gp, bp, rp = gaia[:,0], gaia[:,1], gaia[:,2]
    par, parerr = parallax[:,0], parallax[:,1]
    bp_rp = bp-rp
    absmag = gp+5*(np.log10(np.abs(par))-2)
    absmag_high = gp+5*(np.log10(np.abs(par+parerr))-2)-(gp+5*(np.log10(np.abs(par))-2))
    absmag_low = gp+5*(np.log10(np.abs(par))-2)-(gp+5*(np.log10(np.abs(par-parerr))-2))
    asymmetric_error = np.vstack([absmag_high,absmag_low])

    plotName = os.path.join(outputDir,'HR.pdf')
    plt.figure(figsize=(8,6))
    hist2 = plt.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
    plt.errorbar(bp_rp,absmag,yerr=asymmetric_error,
                 c='r',zorder=1,fmt='o',alpha=0.01)
    plt.xlabel('Gaia BP - RP')
    plt.ylabel('Gaia $M_G$')
    plt.xlim([-1,4.0])
    plt.ylim([-5,18])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

