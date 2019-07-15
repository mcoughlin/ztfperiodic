
import os, sys
import optparse
import pandas as pd
import numpy as np
import tables
import glob
import time
import scipy.constants as ct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from astropy.io import ascii
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, EarthLocation

import requests
import tqdm

from astroquery.vizier import Vizier

from gatspy.periodic import LombScargle, LombScargleFast

LOGIN_URL = "https://irsa.ipac.caltech.edu/account/signon/login.do"
meta_baseurl="https://irsa.ipac.caltech.edu/ibe/search/ztf/products/"
data_baseurl="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/"

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
                             'e_Gmag',
                             'Plx', 'e_Plx', 'BP-RP', 'e_BPmag', 'e_RPmag',
                             'Teff', 'Rad', 'Lum'],
                    column_filters={"phot_g_mean_mag":
                                    ("<%f" % maxmag),
                                   "phot_r_mean_mag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    try:
        source = vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="I/345/gaia2")
        return source[0]
    except:
        return []

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

    try:
        source = vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="II/349/ps1")
        return source[0]
    except:
        return []
 
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

def get_catalog(data):

    ras, decs = [], []
    for dat in data:
        if 'candidate' in dat:
            ras.append(dat["candidate"]["ra"])
            decs.append(dat["candidate"]["dec"])
        else:
            ras.append(dat["ra"])
            decs.append(dat["dec"])

    return SkyCoord(ra=np.array(ras)*u.degree, dec=np.array(decs)*u.degree, frame='icrs')

def get_kowalski(ra, dec, kow, radius = 5.0, oid = None,
                 program_ids = [1, 2,3], min_epochs = 1, name = None):

    tmax = Time('2019-01-01T00:00:00', format='isot', scale='utc').jd

    #qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(ra,dec), "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20190614": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'filter': 1}" } } }
    qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(ra,dec), "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20190614": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'data.catflags': 1, 'filter': 1}" }, "Gaia_DR2": { "filter": "{}", "projection": "{'parallax': 1, 'parallax_error': 1, 'phot_g_mean_mag': 1, 'phot_bp_mean_mag': 1, 'phot_rp_mean_mag': 1, 'ra': 1, 'dec': 1}"}, "ZTF_alerts": { "filter": "{}", "projection": "{'candidate.jd': 1,'candidate.fid': 1, 'candidate.magpsf': 1, 'candidate.sigmapsf': 1, 'candidate.magnr': 1, 'candidate.sigmagnr': 1, 'candidate.distnr': 1, 'candidate.fid': 1, 'candidate.programid': 1, 'candidate.maglim': 1, 'candidate.isdiffpos': 1, 'candidate.ra': 1, 'candidate.dec': 1}" } } }

    r = database_query(kow, qu, nquery = 10)

    if not "result_data" in r:
        print("Query for RA: %.5f, Dec: %.5f failed... returning."%(ra,dec)) 
        return {}

    key1, key2, key3 = 'ZTF_sources_20190614', 'Gaia_DR2', 'ZTF_alerts'
    data1, data2, data3 = r["result_data"][key1], r["result_data"][key2], r["result_data"][key3]
    key = list(data1.keys())[0]
    data = data1[key]
    key = list(data2.keys())[0]
    data2 = data2[key]
    key = list(data3.keys())[0]
    data3 = data3[key]

    cat2 = get_catalog(data2)
    cat3 = get_catalog(data3)

    lightcurves = {}
    for datlist in data:
        objid = str(datlist["_id"])
        if not oid is None:
            if not objid == str(oid):
                continue
        dat = datlist["data"]
        filt = datlist["filter"]
        hjd, mag, magerr, fid = [], [], [], []
        ra, dec = [], []

        for dic in dat:
            if not dic["programid"] in program_ids: continue
            if (dic["programid"]==1) and (dic["hjd"] > tmax): continue
            if not dic["catflags"] == 0: continue

            hjd.append(dic["hjd"])
            mag.append(dic["mag"])
            magerr.append(dic["magerr"])
            ra.append(dic["ra"])
            dec.append(dic["dec"])
            fid.append(filt)
        if len(hjd) < min_epochs: continue

        lightcurves[objid] = {}
        lightcurves[objid]["hjd"] = np.array(hjd)
        lightcurves[objid]["mag"] = np.array(mag)
        lightcurves[objid]["magerr"] = np.array(magerr)
        lightcurves[objid]["ra"] = np.array(ra)
        lightcurves[objid]["dec"] = np.array(dec)
        lightcurves[objid]["fid"] = np.array(fid)

        if name is None:
            ra_hex, dec_hex = convert_to_hex(np.median(ra)*24/360.0,delimiter=''), convert_to_hex(np.median(dec),delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            lightcurves[objid]["name"] = objname
        else:
            lightcurves[objid]["name"] = name

    objids = []
    ras, decs, fids = [], [], []
    for objid in lightcurves.keys():
        objids.append(objid)
        ras.append(np.median(lightcurves[objid]["ra"]))
        decs.append(np.median(lightcurves[objid]["dec"]))
        fids.append(np.median(lightcurves[objid]["fid"]))
    ras, decs, fids = np.array(ras), np.array(decs), np.array(fids).astype(int)

    coords = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs')
    if (len(coords) > 0) and (len(cat2) > 0):
        idx,sep,_ = coords.match_to_catalog_sky(cat2)
    
        for objid, ii, s in zip(objids, idx, sep):
            if s.arcsec > 1:
                lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
                lightcurves[objid]["bp_rp"] = np.nan
            else:     
                dat2 = data2[ii]
                parallax, parallaxerr = dat2["parallax"], dat2["parallax_error"]
                g_mag, bp_mag, rp_mag = dat2["phot_g_mean_mag"], dat2["phot_bp_mean_mag"], dat2["phot_rp_mean_mag"]
                if not ((parallax is None) or (g_mag is None) or (bp_mag is None) or (rp_mag is None)):
                    lightcurves[objid]["absmag"] = [g_mag+5*(np.log10(np.abs(parallax))-2),g_mag+5*(np.log10(np.abs(parallax+parallaxerr))-2)-(g_mag+5*(np.log10(np.abs(parallax))-2)),g_mag+5*(np.log10(np.abs(parallax))-2)-(g_mag+5*(np.log10(np.abs(parallax-parallaxerr))-2))]
                    lightcurves[objid]["bp_rp"] = bp_mag-rp_mag
    
            if not "absmag" in lightcurves[objid]:
                lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
                lightcurves[objid]["bp_rp"] = np.nan
    else:
        for objid in objids:
            lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
            lightcurves[objid]["bp_rp"] = np.nan

    # storage for outputdata
    magnr,sigmagnr,fid = [],[],[]
    jd, mag, magerr, pos = [], [], [], []
    ra, dec, fid, pid = [], [], [], []

    # posneg dict
    idp = dict()
    idp['t'] = 1
    idp['1'] = 1
    idp['f'] = 0
    idp['0'] = 0

    # loop over ids to get the data 
    for datlist in data3:
        objid = str(datlist["_id"])
        dat = datlist["candidate"]

        if idp[dat["isdiffpos"]] == 1:
            continue

        jd.append(dat["jd"])
        mag.append(dat["magpsf"])
        magerr.append(dat["sigmapsf"])
        magnr.append(dat["magnr"])
        sigmagnr.append(dat["sigmagnr"])
        pos.append(idp[dat["isdiffpos"]])
        ra.append(dat["ra"])
        dec.append(dat["dec"])
        fid.append(dat["fid"])
        pid.append(dat["programid"])

    if len(jd) == 0: 
        return lightcurves

    hjds_alert = JD2HJD(jd, ra, dec)
    mags_alert = np.array(mag)
    magerrs_alert = np.array(magerr)
    magnrs_alert = np.array(magnr)
    sigmagnrs_alert = np.array(sigmagnr)
    ras_alert, decs_alert = np.array(ra), np.array(dec)
    fids_alert = np.array(fid)
    pos_alert = np.array(pos,dtype=float)

    fluxrefs_alert,fluxrefs_err_alert,_ = mag2flux(magnrs_alert,sigmagnrs_alert)
    fluxdiffs_alert,fluxdiffs_err_alert,_ = mag2flux(mags_alert,magerrs_alert)

    flux = fluxrefs_alert + (-1)**(1.-pos_alert)*fluxdiffs_alert
    fluxerr = np.sqrt(fluxdiffs_err_alert**2 + fluxrefs_err_alert**2)

    mags_alert, magerrs_alert = flux2mag(flux, fluxerr)

    for hjd, mag, magerr, ra, dec, fid in zip(hjds_alert, mags_alert, magerrs_alert, ras_alert, decs_alert, fids_alert):

        if np.isnan(mag):
            continue
        
        idx = np.where(fid == fids)[0]
        if len(idx) == 0:
            continue

        dist = angular_distance(ra, dec, ras[idx], decs[idx])
        dist = dist * 3600.0

        if np.min(dist) > 1.0:
            continue
 
        idy = np.argmin(dist)       
        objid = objids[idx[idy]]

        lightcurves[objid]["hjd"] = np.append(lightcurves[objid]["hjd"], hjd)
        lightcurves[objid]["mag"] = np.append(lightcurves[objid]["mag"], mag)
        lightcurves[objid]["magerr"] = np.append(lightcurves[objid]["magerr"], magerr)
        lightcurves[objid]["ra"] = np.append(lightcurves[objid]["ra"], ra)
        lightcurves[objid]["dec"] = np.append(lightcurves[objid]["dec"], dec)
        lightcurves[objid]["fid"] = np.append(lightcurves[objid]["fid"], fid)
  
    return lightcurves

def get_kowalski_list(ras, decs, kow, program_ids = [1,2,3], min_epochs = 1,
                      max_error = 2.0, errs = None, names = None,
                      amaj=None, amin=None, phi=None,
                      doCombineFilt=False,
                      doRemoveHC=False):

    baseline=0
    cnt=0
    lnames = []
    lightcurves, filters, ids, coordinates = [], [], [], []
    absmags, bp_rps = [], []   
 
    if errs is None:
        errs = 5.0*np.ones(ras.shape)
    if names is None:
        names = []
        for ra, dec in zip(ras, decs):
            ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)

    for name, ra, dec, err in zip(names, ras, decs, errs):
        if amaj is not None:
            ellipse = patches.Ellipse((ra, dec), amaj[cnt], amin[cnt],
                                      angle=phi[cnt]) 

        if np.mod(cnt,100) == 0:
            print('%d/%d'%(cnt,len(ras)))       
        ls = get_kowalski(ra, dec, kow, radius = err, oid = None,
                          program_ids = program_ids, name = name)
        if len(ls.keys()) == 0: continue

        if doCombineFilt:
            ls = combine_lcs(ls)

        for ii, lkey in enumerate(ls.keys()):
            l = ls[lkey]
            raobj, decobj = l["ra"], l["dec"]
            if len(raobj) == 0:
                continue

            if amaj is not None:
                if not ellipse.contains_point((np.median(raobj),np.median(decobj))): continue

            hjd, mag, magerr, fid = l["hjd"], l["mag"], l["magerr"], l["fid"]
            hjd, mag, magerr = np.array(hjd),np.array(mag),np.array(magerr)
            fid = np.array(fid)

            idx = np.where(~np.isnan(mag) & ~np.isnan(magerr))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            raobj, decobj = raobj[idx], decobj[idx]
            fid = fid[idx]

            idx = np.where(magerr<=max_error)[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            raobj, decobj = raobj[idx], decobj[idx]
            fid = fid[idx]

            idx = np.argsort(hjd)
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            raobj, decobj = raobj[idx], decobj[idx]
            fid = fid[idx]

            if doRemoveHC:
                dt = np.diff(hjd)
                idx = np.setdiff1d(np.arange(len(hjd)),
                                   np.where(dt < 30.0*60.0/86400.0)[0])
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                raobj, decobj = raobj[idx], decobj[idx]
                fid = fid[idx]

            if len(hjd) < min_epochs: continue

            lightcurve=(hjd,mag,magerr)
            lightcurves.append(lightcurve)

            coordinate=(np.median(raobj),np.median(decobj))
            coordinates.append(coordinate)

            filters.append(np.unique(fid).tolist())
            ids.append(int(lkey))

            absmags.append(l["absmag"])
            bp_rps.append(l["bp_rp"])

            lnames.append(l["name"])
 
            newbaseline = max(hjd)-min(hjd)
            if newbaseline>baseline:
                baseline=newbaseline
        cnt = cnt + 1

    return lightcurves, coordinates, filters, ids, absmags, bp_rps, lnames, baseline

def get_simulated(ra, dec, min_epochs = 1, name = None, doUsePDot = False):

    filters = [1,2]
    num_lcs = len(filters)

    max_pdot = 1e-10
    min_pdot = 1e-12

    min_period = 3 * 60.0/86400.0  # 3 minutes
    max_period = 50.0*24*3600.0/86400.0  # 50 days
    max_period = 1.0/2.0

    min_freq = 1./max_period
    max_freq = 1./min_period

    if doUsePDot:
        pdots = 10**np.random.uniform(low=np.log10(min_pdot), high=np.log10(max_pdot), size=num_lcs)
    else:
        pdots = np.zeros((num_lcs,))

    baseline = 366.0
    #baseline = 1.0 
    # 1*ct.Julian_year # 30 days

    actual_freqs = np.random.uniform(low=min_freq, high=max_freq, size=num_lcs)
    number_of_pts = np.random.random_integers(80, 97, size=num_lcs)

    max_mag_factor = 10.0
    min_mag_factor = 1.0
    min_mag_factor = 9.0
    mag_factors = np.random.uniform(low=min_mag_factor, high=max_mag_factor, size=num_lcs)

    lightcurves = {}
    lcs = []
    ce_checks = []
    for i, (num_pts, freq, mag_fac) in enumerate(zip(number_of_pts, actual_freqs, mag_factors)):

        objid = '%10d' % int(np.random.uniform(low=0.0, high=1e10))

        hjd = np.random.uniform(low=0.0, high=baseline, size=num_pts)
        initial_phase = np.random.uniform(low=0.0, high=2*np.pi)
        vert_shift = np.random.uniform(low=mag_fac, high=3*mag_fac)
        pdot = -pdots[i]
        time_vals = hjd - np.min(hjd)
        mag = mag_fac*np.sin(2*np.pi*freq*(time_vals - (1./2.)*pdot*freq*time_vals**2) + initial_phase) + vert_shift
        magerr = 0.05*np.ones(mag.shape)

        #filename = "/home/mcoughlin/ZTF/ztfperiodic/data/lightcurves/ZTFJ1539+5027PTFData.txt"
        #data_out = np.loadtxt(filename)
        #n = 100
        #idx = np.random.randint(len(data_out[:,0]), size=(n,))
        #idx = np.unique(np.sort(idx))
        #idx = np.arange(len(data_out[:,0])).astype(int)
        #idx = np.argsort(data_out[:,2])[:n]
        #idx = np.sort(idx)
        #hjd, mag, magerr = data_out[:,0], data_out[:,1], data_out[:,2]
        #hjd = BJDConvert(hjd, 234.884000, 50.460778).value
        #time_vals = hjd - np.min(hjd)

        #P = (414.79153768 + 9*(0.75/1000))/86400.0
        #freq = 1/P
        #pdot = -2.365e-11
        #phases=np.mod((time_vals-(1.0/2.0)*pdot*freq*(time_vals)**2), P)/P

        if len(hjd) < min_epochs: continue

        lightcurves[objid] = {}
        lightcurves[objid]["hjd"] = hjd
        lightcurves[objid]["mag"] = mag
        lightcurves[objid]["magerr"] = magerr
        lightcurves[objid]["ra"] = ra*np.ones(mag.shape)
        lightcurves[objid]["dec"] = dec*np.ones(mag.shape)
        lightcurves[objid]["fid"] = filters[i]*np.ones(mag.shape)

        if name is None:
            ra_hex, dec_hex = convert_to_hex(np.median(ra)*24/360.0,delimiter=''), convert_to_hex(np.median(dec),delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            lightcurves[objid]["name"] = objname
        else:
            lightcurves[objid]["name"] = name

        lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
        lightcurves[objid]["bp_rp"] = np.nan

    return lightcurves

def get_simulated_list(ras, decs, min_epochs = 1, names = None,
                      doCombineFilt=False,
                      doRemoveHC=False,
                      doUsePDot=False):

    baseline=0
    cnt=0
    lnames = []
    lightcurves, filters, ids, coordinates = [], [], [], []
    absmags, bp_rps = [], []

    if names is None:
        names = []
        for ra, dec in zip(ras, decs):
            ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)

    for name, ra, dec in zip(names, ras, decs):

        if np.mod(cnt,100) == 0:
            print('%d/%d'%(cnt,len(ras)))
        ls = get_simulated(ra, dec, min_epochs = min_epochs, name = name,
                           doUsePDot=doUsePDot)
        if len(ls.keys()) == 0: continue

        if doCombineFilt:
            ls = combine_lcs(ls)

        for ii, lkey in enumerate(ls.keys()):
            l = ls[lkey]
            raobj, decobj = l["ra"], l["dec"]
            if len(raobj) == 0:
                continue

            hjd, mag, magerr, fid = l["hjd"], l["mag"], l["magerr"], l["fid"]
            hjd, mag, magerr = np.array(hjd),np.array(mag),np.array(magerr)
            fid = np.array(fid)

            idx = np.where(~np.isnan(mag) & ~np.isnan(magerr))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            raobj, decobj = raobj[idx], decobj[idx]
            fid = fid[idx]

            idx = np.argsort(hjd)
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            raobj, decobj = raobj[idx], decobj[idx]
            fid = fid[idx]

            if doRemoveHC:
                dt = np.diff(hjd)
                idx = np.setdiff1d(np.arange(len(hjd)),
                                   np.where(dt < 30.0*60.0/86400.0)[0])
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                raobj, decobj = raobj[idx], decobj[idx]
                fid = fid[idx]

            if len(hjd) < min_epochs: continue

            lightcurve=(hjd,mag,magerr)
            lightcurves.append(lightcurve)

            coordinate=(np.median(raobj),np.median(decobj))
            coordinates.append(coordinate)

            filters.append(np.unique(fid).tolist())
            ids.append(int(lkey))

            absmags.append(l["absmag"])
            bp_rps.append(l["bp_rp"])

            lnames.append(l["name"])

            newbaseline = max(hjd)-min(hjd)
            if newbaseline>baseline:
                baseline=newbaseline
        cnt = cnt + 1

    return lightcurves, coordinates, filters, ids, absmags, bp_rps, lnames, baseline

def combine_lcs(ls):

    ras, decs = [], []
    hjds, mags, magerrs = [], [], []
    for ii, lkey in enumerate(ls.keys()):
        l = ls[lkey]
        if ii == 0:
            name = l["name"]
            raobj, decobj = l["ra"], l["dec"]
            hjd, mag, magerr = l["hjd"], l["mag"]-np.median(l["mag"]), l["magerr"]
            fid = l["fid"]
            absmag, bp_rp = l["absmag"], l["bp_rp"]
        else:
            raobj = np.hstack((raobj,l["ra"]))
            decobj = np.hstack((decobj,l["dec"]))
            hjd = np.hstack((hjd,l["hjd"]))
            mag = np.hstack((mag,l["mag"]-np.median(l["mag"])))
            magerr = np.hstack((magerr,l["magerr"]))
            fid = np.hstack((fid,l["fid"]))

    data = {}
    data["ra"] = raobj
    data["dec"] = decobj
    data["hjd"] = hjd
    data["mag"] = mag
    data["magerr"] = magerr
    data["fid"] = fid
    data["absmag"] = absmag
    data["bp_rp"] = bp_rp
    data["name"] = name

    return {lkey: data}

def get_kowalski_bulk(field, ccd, quadrant, kow,
                      program_ids = [2,3], min_epochs = 1, max_error = 2.0,
                      num_batches=1, nb=0):

    tmax = Time('2019-01-01T00:00:00', format='isot', scale='utc').jd

    qu = {"query_type":"general_search","query":"db['ZTF_sources_20190614'].count_documents({'field':%d,'ccd':%d,'quad':%d})"%(field,ccd,quadrant)}
    r = database_query(kow, qu, nquery = 10)

    if not "result_data" in r:
        print("Query for field: %d, CCD: %d, quadrant %d failed... returning."%(field, ccd, quadrant))
        return [], [], [], []

    nlightcurves = r['result_data']['query_result']
    batch_size = np.ceil(nlightcurves/num_batches).astype(int)

    baseline=0
    cnt=0
    names = []
    lightcurves, coordinates, filters, ids = [], [], [], []
    absmags, bp_rps = [], []

    objdata = {}
    #for nb in range(num_batches):
    for nb in [nb]:
        print("Querying batch number %d/%d..."%(nb, num_batches))

        qu = {"query_type":"general_search","query":"db['ZTF_sources_20190614'].find({'field':%d,'ccd':%d,'quad':%d},{'_id':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1,'data.catflags':1}).skip(%d).limit(%d)"%(field,ccd,quadrant,int(nb*batch_size),int(batch_size))}
        r = database_query(kow, qu, nquery = 10)

        if not "result_data" in r:
            print("Query for batch number %d/%d failed... continuing."%(nb, num_batches))
            continue

        #qu = {"query_type":"general_search","query":"db['ZTF_sources_20190614'].find_one({})"}
        #r = kow.query(query=qu)

        datas = r["result_data"]["query_result"]
        for data in datas:
            hjd, mag, magerr, ra, dec, fid = [], [], [], [], [], []
            objid = data["_id"]
            filt = data["filter"]
            data = data["data"]
            for dic in data:
                if not dic["programid"] in program_ids: continue
                if (dic["programid"]==1) and (dic["hjd"] > tmax): continue
                if not dic["catflags"] == 0: continue

                hjd.append(dic["hjd"])
                mag.append(dic["mag"])
                magerr.append(dic["magerr"])
                ra.append(dic["ra"])
                dec.append(dic["dec"])
                fid.append(filt)

            hjd, mag, magerr = np.array(hjd),np.array(mag),np.array(magerr)
            ra, dec = np.array(ra), np.array(dec)

            fid = np.array(fid)
            idx = np.where(~np.isnan(mag) & ~np.isnan(magerr))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

            idx = np.where(magerr<=max_error)[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

            if len(hjd) < min_epochs: continue

            lightcurve=(hjd,mag,magerr)
            lightcurves.append(lightcurve)

            coordinate=(np.median(ra),np.median(dec))
            coordinates.append(coordinate)

            filters.append(np.unique(fid).tolist())
            ids.append(objid)

            absmags.append([np.nan, np.nan, np.nan])
            bp_rps.append(np.nan)

            ra_hex, dec_hex = convert_to_hex(np.median(ra)*24/360.0,delimiter=''), convert_to_hex(np.median(dec),delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)

            newbaseline = max(hjd)-min(hjd)
            if newbaseline>baseline:
                baseline=newbaseline
            cnt = cnt + 1

    return lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline

def get_lightcurve(dataDir, ra, dec, filt, user, pwd):

    directory="%s/*/*/*"%dataDir
    lightcurve = []

    querystr="?POS=%.4f,%.4f"%(ra, dec)
    querystr+="&ct=csv"
    url=os.path.join(meta_baseurl, 'sci')
    tmpfile="tmp.tbl"
    load_file(url+querystr, outf=tmpfile, auth=(user, pwd), showpbar=True)

    data = ascii.read(tmpfile)

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

def get_matchfile(f):

    lightcurves, coordinates = [], []
    baseline = 0
    with tables.open_file(f) as store:
        for tbl in store.walk_nodes("/", "Table"):
            if tbl.name in ["sourcedata", "transientdata"]:
                group = tbl._v_parent
                break
        srcdata = pd.DataFrame.from_records(store.root.matches.sourcedata.read_where('programid > 1'))
        srcdata.sort_values('matchid', axis=0, inplace=True)
        exposures = pd.DataFrame.from_records(store.root.matches.exposures.read_where('programid > 1'))
        merged = srcdata.merge(exposures, on="expid")

        if len(merged.matchid.unique()) == 0:
            return [], [], baseline

        matchids = np.array(merged.matchid)
        values, indices, counts = np.unique(matchids, return_counts=True,return_inverse=True)
        idx = np.where(counts>50)[0]

        matchids, idx2 = np.unique(matchids[idx],return_index=True)
        ncounts = counts[idx][idx2]
        nmatchids = len(idx)

        cnt = 0
        for k in matchids:
            if np.mod(cnt,100) == 0:
                print('%d/%d'%(cnt,len(matchids)))
            df = merged[merged['matchid'] == k]
            RA, Dec, x, err = df.ra, df.dec, df.psfmag, df.psfmagerr
            obsHJD = df.hjd

            if len(x) < 50: continue

            lightcurve=(obsHJD,x,err)
            lightcurves.append(lightcurve)

            coordinate=(RA.values[0],Dec.values[0])
            coordinates.append(coordinate)
 
            newbaseline = max(obsHJD)-min(obsHJD)
            if newbaseline>baseline:
                baseline=newbaseline
            cnt = cnt + 1

    return lightcurves, coordinates, baseline

def database_query(kow, qu, nquery = 5):

    r = {}
    cnt = 0
    while cnt < nquery:
        r = kow.query(query=qu)
        if "result_data" in r:
            break
        time.sleep(5)        
        cnt = cnt + 1
    return r

def BJDConvert(mjd, RA, Dec):
        times=mjd
        t = Time(times,format='mjd',scale='utc')
        t2=t.tdb
        c = SkyCoord(RA,Dec, unit="deg")
        d=c.transform_to(BarycentricTrueEcliptic)
        Palomar=EarthLocation.of_site('Palomar')
        delta=t2.light_travel_time(c,kind='barycentric',location=Palomar)
        BJD_TDB=t2+delta

        return BJD_TDB

def JD2HJD(jd,ra,dec):
    objectcoords = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    palomar = EarthLocation.of_site('palomar')
    times = Time(jd, format='jd',scale='utc', location=palomar)

    ltt_helio = times.light_travel_time(objectcoords, 'heliocentric')

    times_heliocentre = times.utc + ltt_helio

    return times_heliocentre.jd

def angular_distance(ra1, dec1, ra2, dec2):

    delt_lon = (ra1 - ra2)*np.pi/180.
    delt_lat = (dec1 - dec2)*np.pi/180.
    dist = 2.0*np.arcsin( np.sqrt( np.sin(delt_lat/2.0)**2 + \
         np.cos(dec1*np.pi/180.)*np.cos(dec2*np.pi/180.)*np.sin(delt_lon/2.0)**2 ) )

    return dist/np.pi*180.

def convert_to_hex(val, delimiter=':', force_sign=False):
    """
    Converts a numerical value into a hexidecimal string

    Parameters:
    ===========
    - val:           float
                     The decimal number to convert to hex.

    - delimiter:     string
                     The delimiter between hours, minutes, and seconds
                     in the output hex string.

    - force_sign:    boolean
                     Include the sign of the string on the output,
                     even if positive? Usually, you will set this to
                     False for RA values and True for DEC

    Returns:
    ========
    A hexadecimal representation of the input value.
    """
    s = np.sign(val)
    s_factor = 1 if s > 0 else -1
    val = np.abs(val)
    degree = int(val)
    minute = int((val  - degree)*60)
    second = (val - degree - minute/60.0)*3600.
    if degree == 0 and s_factor < 0:
        return '-00{2:s}{0:02d}{2:s}{1:.2f}'.format(minute, second, delimiter)
    elif force_sign or s_factor < 0:
        deg_str = '{:+03d}'.format(degree * s_factor)
    else:
        deg_str = '{:02d}'.format(degree * s_factor)
    return '{0:s}{3:s}{1:02d}{3:s}{2:.2f}'.format(deg_str, minute, second, delimiter)

def overlapping_histogram(a, bins): 
    a =  a.ravel() 
    n = np.zeros(len(bins), int) 

    block = 65536 
    for i in np.arange(0, len(a), block): 
        sa = np.sort(a[i:i+block]) 
        n += np.r_[sa.searchsorted(bins[:-1,1], 'left'), sa.searchsorted(bins[-1,1], 'right')] - np.r_[sa.searchsorted(bins[:-1,0], 'left'), sa.searchsorted(bins[-1,0], 'right')] 
    return n, (bins[:,0]+bins[:,1])/2. 

def mag2flux(mag,dmag=[],flux_0 = 3631.0):
    # converts magnitude to flux in Jy
    flux = flux_0 * 10**(-0.4*mag)

    if dmag==[]:
        return flux
    else:
        dflux_p = (flux_0 * 10**(-0.4*(mag-dmag)) - flux)
        dflux_n = (flux_0 * 10**(-0.4*(mag+dmag)) - flux)
        return flux, dflux_p, dflux_n

def flux2mag(flux, fluxerr, flux_0 = 3631.0):
    mag = -2.5*np.log10(flux/flux_0)
    magerr = np.abs(fluxerr)/flux
    return mag, magerr
