
import os, sys
import pandas as pd
import numpy as np
import h5py
import tables
import glob
import time

from scipy.interpolate import interpolate as interp

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches

import astropy.io.ascii as asci
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, EarthLocation

import requests
import tqdm

from astroquery.vizier import Vizier

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

LOGIN_URL = "https://irsa.ipac.caltech.edu/account/signon/login.do"
meta_baseurl="https://irsa.ipac.caltech.edu/ibe/search/ztf/products/"
data_baseurl="https://irsa.ipac.caltech.edu/ibe/data/ztf/products/"

DEFAULT_TIMEOUT = 5  # seconds

def gaia_query(ra_deg, dec_deg, rad_deg, maxmag=25,
               maxsources=1):
    """
    Query Gaia DR1 @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
    
    See below for explanation:
    https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    
    """
    vquery = Vizier(columns=['all'], column_filters={"phot_g_mean_mag":
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
                             'gmag','rmag','imag','zmag','ymag',
                             'e_gmag','e_rmag','e_imag','e_zmag','e_ymag'],
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

def galex_query(ra_deg, dec_deg, rad_deg, maxmag=25,
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
                             'FUVmag', 'NUVmag',
                             'e_FUVmag', 'e_NUVmag'],
                    column_filters={"FUVmag":
                                    ("<%f" % maxmag),
                                   "NUVmag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')

    try:
        source = vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="II/335/galex_ais")
        return source[0]
    except:
        return []


def sdss_query(ra_deg, dec_deg, rad_deg, maxmag=25,
               maxsources=1):
    """
    Query Pan-STARRS @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS',
                             'umag', 'gmag', 'rmag', 'imag', 'zmag',
                             'e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag'],
                    column_filters={"gmag":
                                    ("<%f" % maxmag),
                                   "rmag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')

    try:
        source = vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="V/147/sdss12")
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

def get_kowalski_external(ra, dec, kow, radius = 5.0):

    qu = { "query_type": "cone_search", "query": {"object_coordinates": {"radec": {'test': [ra,dec]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "Gaia_EDR3": { "filter": "{}", "projection": "{}"}, "AllWISE": { "filter": "{}", "projection": "{}" }, "PS1_DR1": { "filter": "{}", "projection": "{}"}, "GALEX": { "filter": "{}", "projection": "{}"} } } }

    start = time.time()
    r = database_query(kow, qu, nquery = 10)
    end = time.time()
    loadtime = end - start

    if not "data" in r:
        print("Query for RA: %.5f, Dec: %.5f failed... returning."%(ra,dec))
        return {}

    key1, key2, key3, key4 = 'PS1_DR1', 'Gaia_EDR3', 'AllWISE', 'GALEX'
    data1, data2 = r["data"][key1], r["data"][key2]
    data3, data4 = r["data"][key3], r["data"][key4]
    key = list(data1.keys())[0]
    data1 = data1[key]
    key = list(data2.keys())[0]
    data2 = data2[key]
    key = list(data3.keys())[0]
    data3 = data3[key]
    key = list(data4.keys())[0]
    data4 = data4[key]

    w1mpro, w2mpro, w3mpro, w4mpro = np.nan, np.nan, np.nan, np.nan
    w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro = np.nan, np.nan, np.nan, np.nan

    if len(data3) > 0:
        if "w1mpro" in data3:
            w1mpro = data3["w1mpro"]
        if "w2mpro" in data3:
            w2mpro = data3["w2mpro"]
        if "w3mpro" in data3:
            w3mpro = data3["w3mpro"]
        if "w4mpro" in data3:
            w4mpro = data3["w4mpro"]
        if "w1sigmpro" in data3:
            w1sigmpro = data3["w1sigmpro"]
        if "w2sigmpro" in data3:
            w2sigmpro = data3["w2sigmpro"]
        if "w3sigmpro" in data3:
            w3sigmpro = data3["w3sigmpro"]
        if "w4sigmpro" in data3:
            w4sigmpro = data3["w4sigmpro"]

    gMeanPSFMag, rMeanPSFMag, iMeanPSFMag, zMeanPSFMag, yMeanPSFMag = np.nan, np.nan, np.nan, np.nan, np.nan
    gMeanPSFMagErr, rMeanPSFMagErr, iMeanPSFMagErr, zMeanPSFMagErr, yMeanPSFMagErr = np.nan, np.nan, np.nan, np.nan, np.nan

    if len(data1) > 0:
        data1 = data1[0]
        if "gMeanPSFMag" in data1:
            gMeanPSFMag = data1["gMeanPSFMag"]
        if "rMeanPSFMag" in data1:
            rMeanPSFMag = data1["rMeanPSFMag"]
        if "iMeanPSFMag" in data1:
            iMeanPSFMag = data1["iMeanPSFMag"]
        if "zMeanPSFMag" in data1:
            zMeanPSFMag = data1["zMeanPSFMag"]
        if "yMeanPSFMag" in data1:
            yMeanPSFMag = data1["yMeanPSFMag"]
        if "gMeanPSFMagErr" in data1:
            gMeanPSFMagErr = data1["gMeanPSFMagErr"]
        if "rMeanPSFMagErr" in data1:
            rMeanPSFMagErr = data1["rMeanPSFMagErr"]
        if "iMeanPSFMagErr" in data1:
            iMeanPSFMagErr = data1["iMeanPSFMagErr"]
        if "zMeanPSFMagErr" in data1:
            zMeanPSFMagErr = data1["zMeanPSFMagErr"]
        if "yMeanPSFMagErr" in data1:
            yMeanPSFMagErr = data1["yMeanPSFMagErr"]

    parallax, parallax_error = np.nan, np.nan

    if len(data2) > 0:
        data2 = data2[0]
        if "parallax" in data2:
            parallax = data2["parallax"]
        if "parallax_error" in data2:
            parallax_error = data2["parallax_error"]

    FUVmag, NUVmag = np.nan, np.nan
    e_FUVmag, e_NUVmag = np.nan, np.nan
    if len(data4) > 0:
        data4 = data4[0]
        if "NUVmag" in data4:
            NUVmag = data4["NUVmag"]
        if "FUVmag" in data4:
            FUVmag = data4["FUVmag"]
        if "NUVmag" in data4:
            e_NUVmag = data4["e_NUVmag"]
        if "e_FUVmag" in data4:
            e_FUVmag = data4["e_FUVmag"]
        print(FUVmag, NUVmag, e_FUVmag, e_NUVmag)

    external = {}
    external["mag"] = [w1mpro, w2mpro, w3mpro, w4mpro,
                       gMeanPSFMag, rMeanPSFMag,
                       iMeanPSFMag, zMeanPSFMag, yMeanPSFMag,
                       FUVmag, NUVmag]
    external["magerr"] = [w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro,
                          gMeanPSFMagErr, rMeanPSFMagErr,
                          iMeanPSFMagErr, zMeanPSFMagErr,
                          yMeanPSFMagErr,
                          e_FUVmag, e_NUVmag]
    external["parallax"] = [parallax, parallax_error]

    return external

def get_kowalski(ra, dec, kow, radius = 5.0, oid = None,
                 program_ids = [1, 2,3], min_epochs = 1, name = None):

    tmax = Time('2020-06-30T00:00:00', format='isot', scale='utc').jd

    #qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(ra,dec), "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20191101": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'filter': 1}" } } }
    qu = { "query_type": "cone_search", "query": {"object_coordinates": {"radec": {'test': [ra,dec]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20210401": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'data.catflags': 1, 'filter': 1}" }, "Gaia_EDR3": { "filter": "{}", "projection": "{'parallax': 1, 'parallax_error': 1, 'phot_g_mean_mag': 1, 'phot_bp_mean_mag': 1, 'phot_rp_mean_mag': 1, 'phot_g_mean_mag_err': 1, 'phot_bp_mean_flux_over_error': 1, 'phot_rp_mean_flux_over_error': 1, 'ra': 1, 'dec': 1}"}, "ZTF_alerts": { "filter": "{}", "projection": "{'candidate.jd': 1,'candidate.fid': 1, 'candidate.magpsf': 1, 'candidate.sigmapsf': 1, 'candidate.magnr': 1, 'candidate.sigmagnr': 1, 'candidate.distnr': 1, 'candidate.fid': 1, 'candidate.programid': 1, 'candidate.maglim': 1, 'candidate.isdiffpos': 1, 'candidate.ra': 1, 'candidate.dec': 1}" } } } }

    start = time.time()
    r = database_query(kow, qu, nquery = 10)
    end = time.time()
    loadtime = end - start

    if not "data" in r:
        print("Query for RA: %.5f, Dec: %.5f failed... returning."%(ra,dec)) 
        return {}

    key1, key2, key3 = 'ZTF_sources_20210401', 'Gaia_EDR3', 'ZTF_alerts'
    data1, data2, data3 = r["data"][key1], r["data"][key2], r["data"][key3]
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

    print('Loaded %d lightcurves in %.5f seconds' % (len(data), loadtime))

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
                lightcurves[objid]["bp_rp"] = [np.nan, np.nan]
                lightcurves[objid]["parallax"] = np.nan
            else:     
                dat2 = data2[ii]

                if not "parallax" in dat2:
                    parallax, parallaxerr = None, None
                else:
                    parallax, parallaxerr = dat2["parallax"], dat2["parallax_error"]
                if not "phot_g_mean_mag" in dat2:
                    g_mag = None
                else:
                    g_mag = dat2["phot_g_mean_mag"]

                if not "phot_bp_mean_mag" in dat2:
                    bp_mag = None
                else:
                    bp_mag = dat2["phot_bp_mean_mag"]

                if not "phot_rp_mean_mag" in dat2:
                    rp_mag = None
                else:
                    rp_mag = dat2["phot_rp_mean_mag"]

                if not "phot_bp_mean_flux_over_error" in dat2:
                    bp_mean_flux_over_error = None
                else:
                    bp_mean_flux_over_error = dat2["phot_bp_mean_flux_over_error"]

                if not "phot_rp_mean_flux_over_error" in dat2:
                    rp_mean_flux_over_error = None
                else:
                    rp_mean_flux_over_error = dat2["phot_rp_mean_flux_over_error"]

                if not ((parallax is None) or (g_mag is None) or (bp_mag is None) or (rp_mag is None) or (rp_mean_flux_over_error is None) or (rp_mean_flux_over_error is None)):
                    lightcurves[objid]["absmag"] = [g_mag+5*(np.log10(np.abs(parallax))-2),g_mag+5*(np.log10(np.abs(parallax+parallaxerr))-2)-(g_mag+5*(np.log10(np.abs(parallax))-2)),g_mag+5*(np.log10(np.abs(parallax))-2)-(g_mag+5*(np.log10(np.abs(parallax-parallaxerr))-2))]
                    lightcurves[objid]["bp_rp"] = [bp_mag-rp_mag, 2.5/np.log(10) * np.hypot(1/bp_mean_flux_over_error, 1/rp_mean_flux_over_error)] 
                    lightcurves[objid]["parallax"] = parallax    

            if not "absmag" in lightcurves[objid]:
                lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
                lightcurves[objid]["bp_rp"] = [np.nan, np.nan]
                lightcurves[objid]["parallax"] = np.nan
    else:
        for objid in objids:
            lightcurves[objid]["absmag"] = [np.nan, np.nan, np.nan]
            lightcurves[objid]["bp_rp"] = [np.nan, np.nan]
            lightcurves[objid]["parallax"] = np.nan

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

def get_kowalski_objids(objids, kow, program_ids = [1,2,3], min_epochs = 1,
                        doRemoveHC=False, doExtinction=False, max_error = 2.0,
                        doHCOnly=False,
                        doSigmaClipping=False,
                        sigmathresh=5.0,
                        doOutbursting=False,
                        doPercentile=False,
                        percmin = 10.0, percmax = 90.0,
                        doParallel = False,
                        Ncore = 8):

    baseline=0
    cnt=0
    names = []
    lightcurves, coordinates, filters, ids = [], [], [], []
    absmags, bp_rps = [], []

    start = time.time()

    Nmax = 100
    Ncatalog = int(np.ceil(float(len(objids))/Nmax))
    objids_split = np.array_split(objids, Ncatalog)

    data_out = []
    if False: # parallel not working currently...
    #if doParallel:
        from joblib import Parallel, delayed
        data_out = Parallel(n_jobs=Ncore)(delayed(get_kowalski_objid)(objids_tmp,kow,program_ids=program_ids,min_epochs=min_epochs,doRemoveHC=doRemoveHC,doExtinction=doExtinction,doSigmaClipping=doSigmaClipping,sigmathresh=sigmathresh,doOutbursting=doOutbursting,doPercentile=doPercentile,percmin = percmin, percmax = percmax, doHCOnly=doHCOnly) for objids_tmp in objids_split)
    else:
        for oo in range(Ncatalog):
            if np.mod(oo, 10) == 0:
                print('Loading object set %d/%d' % (oo, Ncatalog))
    
            data = get_kowalski_objid(objids_split[oo], kow,
                                      program_ids=program_ids,
                                      min_epochs=min_epochs,
                                      doRemoveHC=doRemoveHC,
                                      doHCOnly=doHCOnly,
                                      doExtinction=doExtinction,
                                      doSigmaClipping=doSigmaClipping,
                                      sigmathresh=sigmathresh,
                                      doOutbursting=doOutbursting,
                                      doPercentile=doPercentile,
                                      percmin = percmin, percmax = percmax)
            data_out.append(data)

    for oo, data in enumerate(data_out):
        lightcurves = lightcurves + data[0]
        coordinates = coordinates + data[1]
        filters = filters + data[2]
        ids = ids + data[3]
        absmags = absmags + data[4]
        bp_rps = bp_rps + data[5]
        names = names + data[6]
        baseline = np.max([baseline,data[7]])

    end = time.time()
    loadtime = end - start

    if len(lightcurves) > 1:
        print('Loaded %d lightcurves in %.5f seconds' % (len(lightcurves), loadtime))

    return lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline


def get_kowalski_objid(objids, kow, program_ids = [1,2,3], min_epochs = 1,
                       doRemoveHC=False, doExtinction=False, max_error = 2.0,
                       doHCOnly=False,
                       doSigmaClipping=False,
                       sigmathresh=5.0,
                       doOutbursting=False,
                       doPercentile=False,
                       percmin = 10.0, percmax = 90.0):

    baseline=0
    cnt=0
    names = []
    lightcurves, coordinates, filters, ids = [], [], [], []
    absmags, bp_rps = [], []

    tmax = Time('2020-06-30T00:00:00', format='isot', scale='utc').jd

    #qu = {"query_type":"find",
    #      "query": {"catalog": 'ZTF_sources_20210401',
    #                "filter": {'_id': {'$eq': int(objid)}},
    #                "projection": "{'_id':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1,'data.catflags':1}"
    #                }
    #     }

    qu = {"query_type":"find",
          "query": {"catalog": 'ZTF_sources_20210401',
                    "filter": {'_id': {'$in': objids.tolist()}}, 
                    "projection": "{'_id':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1,'data.catflags':1}"
                    },
          "kwargs": {'max_time_ms': 10000}
         }
    r = database_query(kow, qu, nquery = 10)

    if not "data" in r:
        print("Query for objid %d failed... continuing."%(objids))
        return []

    datas = r["data"]

    for ii, data in enumerate(datas):
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

        idx = np.argsort(hjd)
        hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
        fid = fid[idx]
        ra, dec = ra[idx], dec[idx]

        if doRemoveHC:
            idx = []
            for ii, t in enumerate(hjd):
                if ii == 0:
                    idx.append(ii)
                else:
                    dt = hjd[ii] - hjd[idx[-1]]
                    if dt >= 30.0*60.0/86400.0:
                        idx.append(ii)
            if len(idx) == 0: continue 
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            ra, decobj = ra[idx], dec[idx]
            fid = fid[idx]

        elif doHCOnly:
            idx = []
            for ii, t in enumerate(hjd):
                if ii == 0:
                    idx.append(ii)
                else:
                    dt = hjd[ii] - hjd[idx[-1]]
                    if dt >= 30.0*60.0/86400.0:
                        idx.append(ii)
            idx = np.setdiff1d(np.arange(len(hjd)), idx)
            if len(idx) == 0: continue
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

            bins = np.arange(np.floor(np.min(hjd)),
                             np.ceil(np.max(hjd)))
            hist, bin_edges = np.histogram(hjd, bins=bins)
            bins = (bin_edges[1:] + bin_edges[:-1])/2.0

            if len(hist) == 0: continue

            idx3 = np.argmax(hist)
            bin_start, bin_end = bin_edges[idx3], bin_edges[idx3+1]              
            idx = np.where((hjd >= bin_start) & (hjd <= bin_end))[0]

            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            ra, decobj = ra[idx], dec[idx]
            fid = fid[idx]

        if doSigmaClipping or doOutbursting:
            iqr = np.diff(np.percentile(mag,q=[25,75]))[0]
            idx = np.where(mag >= np.median(mag)-sigmathresh*iqr)[0]
            if doOutbursting and (len(idx) == len(mag)):
                continue
            if doSigmaClipping:
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                ra, dec = ra[idx], dec[idx]
                fid = fid[idx]

        if doPercentile:
            if len(hjd) == 0: continue
            magmin, magmax = np.percentile(mag, percmin), np.percentile(mag, percmax)
            idx = np.where((mag >= magmin) & (mag <= magmax))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

        if len(hjd) < min_epochs: continue

        lightcurve=(hjd,mag,magerr)
        lightcurves.append(lightcurve)
        filters.append(np.unique(fid).tolist())
        nlightcurves = 1

        radius = 5
        qu = { "query_type": "cone_search", "query": {"object_coordinates": { "radec": {'test': [np.median(ra),np.median(dec)]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "Gaia_EDR3": { "filter": "{}", "projection": "{'parallax': 1, 'parallax_error': 1, 'phot_g_mean_mag': 1, 'phot_bp_mean_mag': 1, 'phot_rp_mean_mag': 1, 'phot_bp_mean_flux_over_error': 1, 'phot_rp_mean_flux_over_error': 1, 'ra': 1, 'dec': 1}"} } }}

        r = database_query(kow, qu, nquery = 10)

        coords = SkyCoord(ra=np.median(ra)*u.degree, 
                          dec=np.median(dec)*u.degree, frame='icrs')

        absmag, bp_rp = [np.nan, np.nan, np.nan], [np.nan, np.nan]
        if "data" in r:
            key2 = 'Gaia_EDR3'
            data2 = r["data"][key2]
            key = list(data2.keys())[0]
            data2 = data2[key]
            cat2 = get_catalog(data2)
            if len(cat2) > 0:
                idx,sep,_ = coords.match_to_catalog_sky(cat2)
                dat2 = data2[idx]
            else:
                dat2 = {}

            if not "parallax" in dat2:
                parallax, parallaxerr = None, None
            else:
                parallax, parallaxerr = dat2["parallax"], dat2["parallax_error"]
            if not "phot_g_mean_mag" in dat2:
                g_mag = None
            else:
                g_mag = dat2["phot_g_mean_mag"]

            if not "phot_bp_mean_mag" in dat2:
                bp_mag = None
            else:
                bp_mag = dat2["phot_bp_mean_mag"]

            if not "phot_rp_mean_mag" in dat2:
                rp_mag = None
            else:
                rp_mag = dat2["phot_rp_mean_mag"]

            if not "phot_bp_mean_flux_over_error" in dat2:
                bp_mean_flux_over_error = None
            else:
                bp_mean_flux_over_error = dat2["phot_bp_mean_flux_over_error"]

            if not "phot_rp_mean_flux_over_error" in dat2:
                rp_mean_flux_over_error = None
            else:
                rp_mean_flux_over_error = dat2["phot_rp_mean_flux_over_error"]

            if not ((parallax is None) or (g_mag is None) or (bp_mag is None) or (rp_mag is None) or (rp_mean_flux_over_error is None) or (rp_mean_flux_over_error is None)):
                absmag = [g_mag+5*(np.log10(np.abs(parallax))-2),g_mag+5*(np.log10(np.abs(parallax+parallaxerr))-2)-(g_mag+5*(np.log10(np.abs(parallax))-2)),g_mag+5*(np.log10(np.abs(parallax))-2)-(g_mag+5*(np.log10(np.abs(parallax-parallaxerr))-2))]
                bp_rp = [bp_mag-rp_mag, 2.5/np.log(10) * np.hypot(1/bp_mean_flux_over_error, 1/rp_mean_flux_over_error)]

        for jj in range(nlightcurves):
            coordinate=(np.median(ra),np.median(dec))
            coordinates.append(coordinate)
            ids.append(objid)
            absmags.append(absmag)
            bp_rps.append(bp_rp)

            ra_hex, dec_hex = convert_to_hex(np.median(ra)*24/360.0,delimiter=''), convert_to_hex(np.median(dec),delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)

        newbaseline = max(hjd)-min(hjd)
        if newbaseline>baseline:
            baseline=newbaseline

    return [lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline]


def get_kowalski_features_list(ras, decs, kow,
                               errs = None, names = None,
                               amaj=None, amin=None, phi=None,
                               featuresetname='f',
                               dbname='ZTF_source_features_DR5'):

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

    cnt = 0
    ztf_ids, df_features = [], []
    for name, ra, dec, err in zip(names, ras, decs, errs):
        if amaj is not None:
            ellipse = patches.Ellipse((ra, dec), amaj[cnt], amin[cnt],
                                      angle=phi[cnt])

        ids, features = get_kowalski_features_ind(ra, dec, kow,
                                                  radius = err, oid = None,
                                                  featuresetname=featuresetname,
                                                  dbname=dbname)
        if len(ids) == 0:
            continue

        if cnt == 0:
            ztf_ids = ids
            df_features = features
        else:
            ztf_ids = ztf_ids.append(ids)
            df_features = df_features.append(features)

        cnt = cnt + 1

    return ztf_ids, df_features


def get_kowalski_list(ras, decs, kow, program_ids = [1,2,3], min_epochs = 1,
                      max_error = 2.0, errs = None, names = None,
                      amaj=None, amin=None, phi=None,
                      doCombineFilt=False,
                      doRemoveHC=False, doExtinction=False,
                      doSigmaClipping=False,
                      sigmathresh=5.0,
                      doOutbursting=False,
                      doCrossMatch=False,
                      crossmatch_radius=3.0):

    baseline=0
    cnt=0
    lnames = []
    lightcurves, filters, ids, coordinates = [], [], [], []
    absmags, bp_rps = [], []   
 
    if doExtinction:
        from dustmaps.config import config
        from dustmaps.bayestar import BayestarQuery
        fullpath = "/".join(sys.argv[0].split("/")[:-2])
        dustmapspath = os.path.join(fullpath, 'dustmaps')
        config['data_dir'] = dustmapspath
        bayestar = BayestarQuery()

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

        if doCrossMatch:
            ls = crossmatch_lcs(ls, crossmatch_radius=crossmatch_radius)

        for ii, lkey in enumerate(ls.keys()):
            l = ls[lkey]
            raobj, decobj = l["ra"], l["dec"]
            if len(raobj) == 0:
                continue

            if doExtinction:
                parallax = l["parallax"]
                dist = 1/(parallax*1e-3)
                if np.isnan(dist) or (dist < 0):
                    dist = 100*1e3
                
                coord = SkyCoord(np.median(raobj)*u.deg,
                                 np.median(decobj)*u.deg, 
                                 distance=dist*u.pc, frame='icrs')
                ebv = bayestar(coord, mode='median')
                extinction_coeff = np.array([3.518, 2.617, 1.971, 1.549, 
                                             1.263, 0.7927, 0.4690, 0.3026])
                extinction = extinction_coeff * ebv

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
                idx = []
                for ii, t in enumerate(hjd):
                    if ii == 0:
                        idx.append(ii)
                    else:
                        dt = hjd[ii] - hjd[idx[-1]]
                        if dt >= 30.0*60.0/86400.0:
                            idx.append(ii)
                idx = np.array(idx)
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                raobj, decobj = raobj[idx], decobj[idx]
                fid = fid[idx]

            if doSigmaClipping or doOutbursting:
                iqr = np.diff(np.percentile(mag,q=[25,75]))[0] 
                idx = np.where(mag >= np.median(mag)-sigmathresh*iqr)[0]
                if doOutbursting and (len(idx) == len(mag)):
                    continue
                if doSigmaClipping: 
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

    import ztfperiodic.simulate

    filters = [1,2]
    num_lcs = len(filters)

    max_pdot = 1e-10
    min_pdot = 1e-12

    min_period = 10 * 60.0/86400.0  # 10 minutes
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

        hjd = ztfperiodic.simulate.time()
        flux, p, errs = ztfperiodic.simulate.pdot_lc(t_obs=hjd, Pdot=pdots[i],
                                               period=1/freq)
        mag = -2.5*np.log10(flux)
        magerr = 0.05*np.ones(mag.shape)

        #hjd = np.random.uniform(low=0.0, high=baseline, size=num_pts)
        #initial_phase = np.random.uniform(low=0.0, high=2*np.pi)
        #vert_shift = np.random.uniform(low=mag_fac, high=3*mag_fac)
        #pdot = -pdots[i]
        #time_vals = hjd - np.min(hjd)
        #mag = mag_fac*np.sin(2*np.pi*freq*(time_vals - (1./2.)*pdot*freq*time_vals**2) + initial_phase) + vert_shift
        #magerr = 0.05*np.ones(mag.shape)

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
        lightcurves[objid]["bp_rp"] = [np.nan, np.nan]
        lightcurves[objid]["parallax"] = np.nan

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
            absmag, bp_rp, parallax = l["absmag"], l["bp_rp"], l["parallax"]
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
    data["parallax"] = parallax
    data["name"] = name

    return {lkey: data}

def crossmatch_lcs(ls, crossmatch_radius=3.0):

    # determine ras and decs
    ras, decs = [], []
    for ii, lkey in enumerate(ls.keys()):
        l = ls[lkey]
        ras.append(np.mean(l["ra"]))
        decs.append(np.mean(l["dec"]))
    keys, ras, decs = np.array(ls.keys()), np.array(ras), np.array(decs)

    groups = []
    while len(ras) > 0:
        ra, dec = ras[0], decs[0]

        coords = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs')
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        sep = coord.separation(coords).arcsec
       
        idx = np.where(sep <= crossmatch_radius)[0]
        groups.append(keys[idx])
        idy = np.where(sep > crossmatch_radius)[0]
        keys, ras, decs = keys[idy], ras[idy], decs[idy]

    data = {}
    for keys in groups:
        for ii, lkey in enumerate(keys):
            l = ls[lkey]
            if ii == 0:
                name = l["name"]
                raobj, decobj = l["ra"], l["dec"]
                hjd, mag, magerr = l["hjd"], l["mag"]-np.median(l["mag"]), l["magerr"]
                fid = l["fid"]
                absmag, bp_rp, parallax = l["absmag"], l["bp_rp"], l["parallax"]
            else:
                raobj = np.hstack((raobj,l["ra"]))
                decobj = np.hstack((decobj,l["dec"]))
                hjd = np.hstack((hjd,l["hjd"]))
                mag = np.hstack((mag,l["mag"]-np.median(l["mag"])))
                magerr = np.hstack((magerr,l["magerr"]))
                fid = np.hstack((fid,l["fid"]))

        data[lkey] = {}
        data[lkey]["ra"] = raobj
        data[lkey]["dec"] = decobj
        data[lkey]["hjd"] = hjd
        data[lkey]["mag"] = mag
        data[lkey]["magerr"] = magerr
        data[lkey]["fid"] = fid
        data[lkey]["absmag"] = absmag
        data[lkey]["bp_rp"] = bp_rp
        data[lkey]["parallax"] = parallax
        data[lkey]["name"] = name

    return data

def get_kowalski_bulk(field, ccd, quadrant, kow,
                      program_ids = [2,3], min_epochs = 1, max_error = 2.0,
                      num_batches=1, nb=0,
                      doRemoveHC=False, doHCOnly=False,
                      doSigmaClipping=False,
                      sigmathresh=5.0,
                      doOutbursting=False,
                      doAlias=False,
                      doPercentile=False,
                      percmin = 10.0, percmax = 90.0):

    tmax = Time('2020-06-30T00:00:00', format='isot', scale='utc').jd

    qu = {"query_type":"general_search","query":"db['ZTF_sources_20210401'].count_documents({'field':%d,'ccd':%d,'quad':%d})"%(field,ccd,quadrant)}

    start = time.time()
    r = database_query(kow, qu, nquery = 10)
    end = time.time()
    loadtime = end - start

    if not "data" in r:
        print("Query for field: %d, CCD: %d, quadrant %d failed... returning."%(field, ccd, quadrant))
        return [], [], [], [], [], [], [], []

    if doAlias:
        magerrdir = "/home/michael.coughlin/ZTF/ztfperiodic/input"
        gmagerr = os.path.join(magerrdir,'gmagerr.txt')
        rmagerr = os.path.join(magerrdir,'Rmagerr.txt')

        gerr = pd.read_csv(gmagerr,sep=' ',names=['Mag','Err'])
        rerr = pd.read_csv(rmagerr,sep=' ',names=['Mag','Err'])
        gmags, gerrs = gerr['Mag'], gerr['Err']
        rmags, rerrs = rerr['Mag'], rerr['Err']

    nlightcurves = r['data']
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

        qu = {"query_type":"general_search","query":"db['ZTF_sources_20210401'].find({'field':%d,'ccd':%d,'quad':%d},{'_id':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1,'data.catflags':1}).skip(%d).limit(%d)"%(field,ccd,quadrant,int(nb*batch_size),int(batch_size))}
        r = database_query(kow, qu, nquery = 10)

        if not "data" in r:
            print("Query for batch number %d/%d failed... continuing."%(nb, num_batches))
            continue

        #qu = {"query_type":"general_search","query":"db['ZTF_sources_20210401'].find_one({})"}
        #r = kow.query(query=qu)

        datas = r["data"]

        if doHCOnly:
            tt = np.empty((0,1))
            for data in datas:
                hjd = []
                data = data["data"]
                for dic in data:
                    if not dic["programid"] in program_ids: continue
                    if (dic["programid"]==1) and (dic["hjd"] > tmax): continue
                    if not dic["catflags"] == 0: continue

                    hjd.append(dic["hjd"])
                hjd = np.array(hjd)
                tt = np.unique(np.append(tt,hjd))
            magmat = np.nan*np.ones((len(tt),len(datas))) # (nepoch x nsources)

        for ii, data in enumerate(datas):
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

            idx = np.argsort(hjd)
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

            if doAlias:
                basemag = np.random.uniform(low=15.0, high=21.0)
                errors = np.interp(basemag,gmags,gerrs)
                mag = errors*np.random.randn(len(mag))+basemag+0.05*np.sin(2*np.pi*1.0*hjd)
                magerr[:] = errors

            if doRemoveHC:
                idx = []
                for ii, t in enumerate(hjd):
                    if ii == 0:
                        idx.append(ii)
                    else:
                        dt = hjd[ii] - hjd[idx[-1]]
                        if dt >= 30.0*60.0/86400.0:
                            idx.append(ii)
                idx = np.array(idx).astype(int)
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                ra, dec = ra[idx], dec[idx]
                fid = fid[idx]
            elif doHCOnly:
                dt = np.diff(hjd)
                idx = np.setdiff1d(np.arange(len(hjd)),
                                   np.where(dt >= 30.0*60.0/86400.0)[0])
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                fid = fid[idx]
                ra, dec = ra[idx], dec[idx]

            if doSigmaClipping or doOutbursting:
                iqr = np.diff(np.percentile(mag,q=[25,75]))[0]
                idx = np.where(mag >= np.median(mag)-sigmathresh*iqr)[0]
                if doOutbursting and (len(idx) == len(mag)):
                    continue
                if doSigmaClipping:
                    hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                    ra, dec = ra[idx], dec[idx]
                    fid = fid[idx]

            if len(hjd) < min_epochs: continue

            if doPercentile:
                magmin, magmax = np.percentile(mag, percmin), np.percentile(mag, percmax)
                idx = np.where((mag >= magmin) & (mag <= magmax))[0]
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                fid = fid[idx]
                ra, dec = ra[idx], dec[idx]

            if doHCOnly:
                f = interp.interp1d(hjd, mag, fill_value=np.nan, bounds_error=False)
                yinterp = f(tt)
                magmat[:, ii] = yinterp - np.nanmedian(yinterp)

                lightcurves_tmp, filters_tmp = split_lightcurve(hjd, mag,
                                                                magerr, fid,
                                                                min_epochs)
                if len(lightcurves_tmp) == 0:
                    continue
                nlightcurves = len(lightcurves_tmp)
                for lightcurve, filt in zip(lightcurves_tmp, filters_tmp):
                    lightcurves.append(lightcurve)
                    filters.append(filt)
            else:
                lightcurve=(hjd,mag,magerr)
                lightcurves.append(lightcurve)
                filters.append(np.unique(fid).tolist())
                nlightcurves = 1

            for jj in range(nlightcurves): 
                coordinate=(np.median(ra),np.median(dec))
                coordinates.append(coordinate)
                ids.append(objid)
                absmags.append([np.nan, np.nan, np.nan])
                bp_rps.append([np.nan, np.nan])

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

    if doHCOnly:
        magmat_median = np.nanmedian(magmat, axis=1)
        f = interp.interp1d(tt, magmat_median, fill_value='extrapolate')
        lightcurves2 = []
        for ii in range(len(lightcurves)):
            magmat_median_array = f(lightcurves[ii][0])
            magmat_median_array[np.isnan(magmat_median_array)] = 0.0
            lightcurve2 = (lightcurves[ii][0],
                           lightcurves[ii][1] - magmat_median_array,
                           lightcurves[ii][2])
            lightcurves2.append(lightcurve2)
        lightcurves = lightcurves2

    print('Loaded %d lightcurves in %.5f seconds' % (len(lightcurves), loadtime))

    return lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline

def get_featuresetnames(featuresetname):

    feature_all = ['ra', 'dec', 'period', 'significance', 'pdot', 'n', 'median', 'wmean', 'chi2red', 'roms', 'wstd', 'norm_peak_to_peak_amp', 'norm_excess_var', 'median_abs_dev', 'iqr', 'i60r', 'i70r', 'i80r', 'i90r', 'skew', 'smallkurt', 'inv_vonneumannratio', 'welch_i', 'stetson_j', 'stetson_k', 'ad', 'sw', 'f1_power', 'f1_BIC', 'f1_a', 'f1_b', 'f1_amp', 'f1_phi0', 'f1_relamp1', 'f1_relphi1', 'f1_relamp2', 'f1_relphi2', 'f1_relamp3', 'f1_relphi3', 'f1_relamp4', 'f1_relphi4', 'n_ztf_alerts', 'mean_ztf_alert_braai', 'dmdt', 'AllWISE___id', 'AllWISE__w1mpro', 'AllWISE__w1sigmpro', 'AllWISE__w2mpro', 'AllWISE__w2sigmpro', 'AllWISE__w3mpro', 'AllWISE__w3sigmpro', 'AllWISE__w4mpro', 'AllWISE__w4sigmpro', 'AllWISE__ph_qual', 'Gaia_EDR3___id', 'Gaia_EDR3__phot_g_mean_mag', 'Gaia_EDR3__phot_bp_mean_mag', 'Gaia_EDR3__phot_rp_mean_mag', 'Gaia_EDR3__parallax', 'Gaia_EDR3__parallax_error', 'Gaia_EDR3__pmra', 'Gaia_EDR3__pmra_error', 'Gaia_EDR3__pmdec', 'Gaia_EDR3__pmdec_error', 'Gaia_EDR3__astrometric_excess_noise', 'Gaia_EDR3__phot_bp_rp_excess_factor', 'PS1_DR1___id', 'PS1_DR1__gMeanPSFMag', 'PS1_DR1__gMeanPSFMagErr', 'PS1_DR1__rMeanPSFMag', 'PS1_DR1__rMeanPSFMagErr', 'PS1_DR1__iMeanPSFMag', 'PS1_DR1__iMeanPSFMagErr', 'PS1_DR1__zMeanPSFMag', 'PS1_DR1__zMeanPSFMagErr', 'PS1_DR1__yMeanPSFMag', 'PS1_DR1__yMeanPSFMagErr', 'PS1_DR1__qualityFlag']

    feature_set11 = ['median', 'wmean', 'chi2red', 'roms', 'wstd', 'norm_peak_to_peak_amp',
           'norm_excess_var', 'median_abs_dev', 'iqr', 'i60r', 'i70r', 'i80r', 'i90r',
           'skew', 'smallkurt', 'inv_vonneumannratio', 'welch_i', 'stetson_j',
           'stetson_k', 'ad', 'sw']
    
    feature_set12 = ['f1_power', 'f1_BIC', 'f1_a', 'f1_b', 'f1_amp',
           'f1_phi0', 'f1_relamp1', 'f1_relphi1', 'f1_relamp2', 'f1_relphi2',
           'f1_relamp3', 'f1_relphi3', 'f1_relamp4', 'f1_relphi4']
    
    feature_set21 = ['period', 'significance', 'pdot']
    feature_set22 = [ # removing n from featureset. It does not really belong there.
    #    'n', 
        'n_ztf_alerts', 'mean_ztf_alert_braai']
    
    feature_set31 = [
    #    'AllWISE___id', 
            'AllWISE__w1mpro', 'AllWISE__w1sigmpro',
           'AllWISE__w2mpro', 'AllWISE__w2sigmpro', 'AllWISE__w3mpro',
           'AllWISE__w3sigmpro', 'AllWISE__w4mpro', 'AllWISE__w4sigmpro',
    #       'AllWISE__ph_qual', 
    #        'Gaia_EDR3___id', 
            'Gaia_EDR3__phot_g_mean_mag',
           'Gaia_EDR3__phot_bp_mean_mag', 'Gaia_EDR3__phot_rp_mean_mag',
           'Gaia_EDR3__parallax',
    #    'Gaia_EDR3__parallax_error', 'Gaia_EDR3__pmra',
    #       'Gaia_EDR3__pmra_error', 'Gaia_EDR3__pmdec', 'Gaia_EDR3__pmdec_error',
    #       'Gaia_EDR3__astrometric_excess_noise',
           'Gaia_EDR3__phot_bp_rp_excess_factor',
    #    'PS1_DR1___id',
           'PS1_DR1__gMeanPSFMag', 'PS1_DR1__gMeanPSFMagErr',
           'PS1_DR1__rMeanPSFMag', 'PS1_DR1__rMeanPSFMagErr',
           'PS1_DR1__iMeanPSFMag', 'PS1_DR1__iMeanPSFMagErr',
           'PS1_DR1__zMeanPSFMag', 'PS1_DR1__zMeanPSFMagErr',
           'PS1_DR1__yMeanPSFMag', 'PS1_DR1__yMeanPSFMagErr',
    #       'PS1_DR1__qualityFlag'
            ]
    feature_set32 = ['Gaia_EDR3__parallax_error', 'Gaia_EDR3__pmra',
           'Gaia_EDR3__pmra_error', 'Gaia_EDR3__pmdec', 'Gaia_EDR3__pmdec_error',
           'Gaia_EDR3__astrometric_excess_noise']
   
    phenomenological = ['dmdt', 'ad', 'chi2red', 'f1_a', 'f1_amp' ,'f1_b', 
                        'f1_BIC', 'f1_phi0', 'f1_power', 'f1_relamp1',
                        'f1_relamp2', 'f1_relamp3', 'f1_relamp4',
                        'f1_relphi1', 'f1_relphi2', 'f1_relphi3',
                        'f1_relphi4', 'i60r', 'i70r', 'i80r', 'i90r',
                        'inv_vonneumannratio', 'iqr', 'median',
                        'median_abs_dev', 'norm_excess_var',
                        'norm_peak_to_peak_amp', 'pdot', 'period',
                        'roms', 'significance', 'skew', 'smallkurt',
                        'stetson_j', 'stetson_k', 'sw', 'welch_i', 
                        'wmean', 'wstd', 'n_ztf_alerts',
                        'mean_ztf_alert_braai']
    ontological = phenomenological + ['AllWISE__w1mpro',
                                      'AllWISE__w1sigmpro',
                                      'AllWISE__w2mpro',
                                      'AllWISE__w2sigmpro',
                                      'AllWISE__w3mpro',
                                      'AllWISE__w3sigmpro',
                                      'AllWISE__w4mpro',
                                      'AllWISE__w4sigmpro',
                                      'Gaia_EDR3__phot_g_mean_mag',
                                      'Gaia_EDR3__phot_bp_mean_mag',
                                      'Gaia_EDR3__phot_rp_mean_mag',
                                      'Gaia_EDR3__parallax',
                                      'Gaia_EDR3__parallax_error',
                                      'Gaia_EDR3__pmra',
                                      'Gaia_EDR3__pmra_error',
                                      'Gaia_EDR3__pmdec',
                                      'Gaia_EDR3__pmdec_error',
                                      'Gaia_EDR3__astrometric_excess_noise',
                                      'Gaia_EDR3__phot_bp_rp_excess_factor',
                                      'PS1_DR1__gMeanPSFMag',
                                      'PS1_DR1__gMeanPSFMagErr',
                                      'PS1_DR1__rMeanPSFMag',
                                      'PS1_DR1__rMeanPSFMagErr',
                                      'PS1_DR1__iMeanPSFMag',
                                      'PS1_DR1__iMeanPSFMagErr',
                                      'PS1_DR1__zMeanPSFMag',
                                      'PS1_DR1__zMeanPSFMagErr',
                                      'PS1_DR1__yMeanPSFMag',
                                      'PS1_DR1__yMeanPSFMagErr'] 

    feature_limited = ['ra', 'dec', 'period', 'significance']

    # Do b, d, e, f in that order 
    feature_set_b = feature_set11
    feature_set_c =  feature_set_b + feature_set12
    feature_set_d =  feature_set_c + feature_set21 + feature_set22
    feature_set_e =  feature_set_d + feature_set31
    feature_set_f =  feature_set_e + feature_set32
    
    feature_set_nonztf = feature_set31 + feature_set32  
 
    featuresetnames = {'b': feature_set_b,   # 21 features
                       'c': feature_set_c,  # 35 features
                       'd': feature_set_d,  # 41 features - 1 (n)
                       'e': feature_set_e,  # 64 features - 1 (n)
                       'f': feature_set_f,  # 70 features - 1 (n)
                       'phenomenological': phenomenological,
                       'ontological': ontological,
                       'all': feature_all,
                       'nonztf': feature_set_nonztf,
                       'limited': feature_limited}

    return featuresetnames[featuresetname]

def get_kowalski_objids_from_radec(ra, dec, kow, radius = 5.0,
                                   dbname='ZTF_source_features_DR5'):

    qu = { "query_type": "cone_search", "query": {"object_coordinates": { "radec": { "test": [ra,dec]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { dbname: { "filter": "{}", "projection": "{'_id':1}" } } } }

    r = database_query(kow, qu, nquery = 10)

    data = r["data"][dbname]
    key = data.keys()[0]
    data = data[key]

    objids = []
    for datlist in data:
        objid = str(datlist["_id"])
        objids.append(int(objid))

    return objids

def get_kowalski_features_ind(ra, dec, kow, radius = 5.0, oid = None,
                              featuresetname='f',
                              dbname='ZTF_source_features_DR5'):

    start = time.time()

    featuresetnames = get_featuresetnames(featuresetname)

    qu = { "query_type": "cone_search", "query": {"object_coordinates": { "radec": { "test": [ra,dec]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { dbname: { "filter": "{}", "projection": "{}" } } } }

    r = database_query(kow, qu, nquery = 10)

    data = r["data"][dbname]
    key = data.keys()[0]
    data = data[key]

    features = {}
    for datlist in data:
        objid = str(datlist["_id"])
        if not oid is None:
            if not objid == str(oid):
                continue
        datlist["ztf_id"] = datlist["_id"]
        del datlist["_id"]
        df_series = pd.Series(datlist).fillna(0)
        features[objid] = df_series

    if not features:
        return [], []

    df_features = pd.DataFrame.from_dict(features, orient='index', 
                                         columns=df_series.index)

    end = time.time()
    loadtime = end - start

    return df_features["ztf_id"], df_features[featuresetnames]

def get_kowalski_features_objids(objids, kow, featuresetname='f',
                                 dbname='ZTF_source_features_DR5'):

    start = time.time()

    featuresetnames = get_featuresetnames(featuresetname)

    features = {}    
    for ii, objid in enumerate(objids):
        if (ii > 0) and (np.mod(ii,100) == 0):
            print('Loading %d/%d...' % (ii, len(objids)))

        qu = {"query_type":"find",
              "query": {"catalog": dbname,
                        "filter": {"_id": int(objid)},
                        "projection": {}},
             }
        r = database_query(kow, qu, nquery = 10)

        if not "data" in r:
            print("Query for objid %d failed... continuing."%(objid))
            continue
 
        if len(r["data"]) == 0: continue

        datlist = r["data"][0]
        datlist["ztf_id"] = datlist["_id"]
        del datlist["_id"]

        df_series = pd.Series(datlist).fillna(0)
        features[objid] = df_series

    if not features:
        return [], []

    df_features = pd.DataFrame.from_dict(features, orient='index',
                                         columns=df_series.index)

    end = time.time()
    loadtime = end - start

    if len(objids) > 1:
        print('Loaded %d features in %.5f seconds' % (len(objids), loadtime))

    return df_features["ztf_id"], df_features[featuresetnames]


def get_kowalski_classifications_objids(objids, kow,
                                        dbname='ZTF_source_classifications_20191101',
                                        version='d11_dnn_v2_20200627'):

    classifications = {}
    for ii, objid in enumerate(objids):
        if (ii > 0) and (np.mod(ii,100) == 0):
            print('Loading %d/%d...' % (ii, len(objids)))

        qu = {"query_type":"find",
              "query": {"catalog": dbname,
                        "filter": {"_id": int(objid)},
                        "projection": {}},
             }
        r = database_query(kow, qu, nquery = 10)

        if not "data" in r:
            print("Query for objid %d failed... continuing."%(objid))
            continue

        if len(r["data"]) == 0: continue

        datlist = r["data"][0]

        datas = {}
        for datkey in datlist.keys():
            if datkey == "_id":
                datas["_id"] = datlist["_id"]
                continue
            for dat in datlist[datkey]:
                if "version" in dat.keys() and dat["version"] == version:
                    datas[datkey] = dat["value"]

        datas["ztf_id"] = datas["_id"]
        del datas["_id"]
        df_series = pd.Series(datas).fillna(0)
        classifications[objid] = df_series

    if not classifications:
        return []

    df_classifications = pd.DataFrame.from_dict(classifications, orient='index',
                                                columns=df_series.index)

    return df_classifications

def get_kowalski_features(kow, num_batches=1, nb=0, featuresetname='f',
                          dbname='ZTF_source_features_20191101'):

    featuresetnames = get_featuresetnames(featuresetname)

    #qu = {"query_type":"general_search","query":"db['ZTF_source_features_20191101'].count_documents()"}
    qu = {"query_type":"general_search","query":"db['%s'].count_documents()" % dbname}

    #start = time.time()
    #r = database_query(kow, qu, nquery = 10)
    #end = time.time()
    #loadtime = end - start

    #if not "data" in r:
    #    print("Query for batch %d failed... returning."%(nb))
    #    return [], [], [], []

    #nlightcurves = r['data']['query_result']

    if dbname == 'ZTF_source_features_20191101_20_fields':
        nlightcurves = 34681547
    elif dbname == 'ZTF_source_features_20191101':
        nlightcurves = 578676249
    else:
        print('dbname %s now known... exiting.')
        exit(0)

    #nlightcurves = 1000000

    batch_size = np.ceil(nlightcurves/num_batches).astype(int)

    objdata = {}
    #for nb in range(num_batches):
    for nb in [nb]:
        print("Querying batch number %d/%d..."%(nb, num_batches))

        start = time.time()
        #qu = {"query_type":"general_search","query":"db['%s'].find({}).skip(%d).limit(%d)"%(dbname, int(nb*batch_size),int(batch_size))}
        qu = {"query_type":"find",
              "query": {"catalog": dbname,
                        "filter": {},
                        "projection": {}},
              "kwargs": {"skip": int(nb*batch_size),
                        "limit": int(batch_size)}
             }
                       
        r = database_query(kow, qu, nquery = 10)
        end = time.time()
        loadtime = end - start
        print("Feature query: %.5f seconds" % loadtime)

        if not "data" in r:
            print("Query for batch number %d/%d failed... continuing."%(nb, num_batches))
            continue

        #qu = {"query_type":"general_search","query":"db['ZTF_sources_20210401'].find_one({})"}
        #r = kow.query(query=qu)

        datas = r["data"]
        start = time.time()
        df_features = pd.DataFrame(datas).fillna(0)
        df_features.rename(columns={"_id": "ztf_id"}, inplace=True)
        end = time.time()
        loadtime = end - start
        print("Dataframe: %.5f seconds" % loadtime)

    return df_features["ztf_id"], df_features[featuresetnames]


def split_lightcurve(hjd, mag, magerr, fid, min_epochs):

    dt = np.diff(hjd)
    idy = np.where(dt > 1e-2)[0]
    ddy = np.diff(idy)
    idz = np.where(ddy >= min_epochs)[0]

    lightcurves_tmp, filters_tmp = [], [] 
    if len(idy) > 0:
        for idpeak in idz:
            idx = np.arange(idy[idpeak]+1, idy[idpeak+1]-1)
            if len(idx) >= min_epochs: continue

            lightcurve=(hjd[idx],mag[idx],magerr[idx])
            lightcurves_tmp.append(lightcurve)
            filters_tmp.append(np.unique(fid[idx]).tolist())
    elif (len(idy) == 1) and idy[0] > min_epochs:
        idx = np.arange(idy[0]-1)
        lightcurve=(hjd[idx],mag[idx],magerr[idx])
        lightcurves_tmp.append(lightcurve)
        filters_tmp.append(np.unique(fid[idx]).tolist())

    return lightcurves_tmp, filters_tmp 

def get_lightcurve(dataDir, ra, dec, filt, user, pwd):
    """
    Get the light curve from ipac database website
    user & pwd to ipac database
    """
    directory="%s/*/*/*"%dataDir
    lightcurve = []

    querystr="?POS=%.4f,%.4f"%(ra, dec)
    querystr+="&ct=csv"
    url = os.path.join(meta_baseurl, 'sci')
    tmpfile="tmp.tbl"
    load_file(url+querystr, outf=tmpfile, auth=(user, pwd), showpbar=True)

    data = asci.read(tmpfile)

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


def find_matchfile(matchfileDir, objid = 10593142036566):
    """
    Find the filepath based on an objid 
    On Schoty: matchfileDir = "/gdata/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch"
    """
    fs=str(objid)
    fieldID=fs[1:5]
    rcID=fs[5:7]
    filterID=fs[7]
    if int(filterID)==1:
        filterid='g'
    elif int(filterID)==2:
        filterid='r'
    elif int(filterID)==3:
        filterid='i'
    ccdid=int((np.ceil((float(rcID)+1) / 4)))
    if ccdid<10:
        ccdid=str(0)+str(ccdid)
    else:
        ccdid=str(ccdid)
    quadrant= str((int(rcID) % 4) +1)
    rounded=int(50 * np.ceil(float(fieldID[0:5])/50))
    if (rounded>999):
        filename=matchfileDir+'/rc'+rcID+'/fr00'+str(rounded-49)+'-00'+str(rounded)+\
            '/ztf_00'+fieldID+'_z'+filterid+'_c'+ccdid+'_q'+quadrant+'_match.pytable'
    else:
        filename=matchfileDir+'/rc'+rcID+'/fr000'+str(rounded-49)+'-000'+str(rounded)+\
            '/ztf_00'+fieldID+'_z'+filterid+'_c'+ccdid+'_q'+quadrant+'_match.pytable'
    return filename


def get_matchfile(kow, filename, min_epochs = 1,
                  program_ids=[1,2,3],
                  doRemoveHC=False, doHCOnly=False,
                  Ncatalog = 1, Ncatindex = 0, max_error = np.inf,
                  doSigmaClipping=False,
                  sigmathresh=5.0,
                  doOutbursting=False,
                  doPercentile=False,
                  percmin = 10.0, percmax = 90.0,
                  matchfileType = 'forced'):
    """
    Read matchfile (hdf file) light curves given the filename
    e.g.: f = '/path/to/fr000551-000600/ztf_000593_zr_c04_q3_match.pytable'
    """

    start = time.time()

    tmax = Time('2020-06-30T00:00:00', format='isot', scale='utc').jd

    bands = {'g': 1, 'r': 2, 'i': 3, 'z': 4, 'J': 5}

    if matchfileType == 'forced':
        filenameSplit = filename.split("/")[-1].split("_")
        field_id, ccd_id, q_id = int(filenameSplit[1]), int(filenameSplit[2]), int(filenameSplit[3])
        filt = filenameSplit[4][1]
        filt = bands[filt]
    
        f = h5py.File(filename, 'r')
        sourcedata = np.array(f['data']['sourcedata'][:])
        exposures = f['data']['exposures'][:]
        sources = f['data']['sources'][:]
        nlightcurves = len(sources)
        n_exp = len(exposures)
    elif matchfileType == 'kevin':
        f = h5py.File(filename, 'r')
        sources = [str(WD) for WD in f['Objects']]
    else:
        print('I do not know that match file type... exiting.')
        raise Exception

    baseline = 0
    names = []
    lightcurves, coordinates, filters, ids = [], [], [], []
    absmags, bp_rps = [], []
    hjds = []

    kks_split = np.array_split(np.arange(len(sources)),Ncatalog)
    kks = kks_split[Ncatindex]

    for kk, source in enumerate(sources):
        if not np.isin(kk, kks, assume_unique=True): continue
        if np.mod(kk, 100) == 0:
            print('Reading source: %d' % kk)

        if matchfileType == 'forced':
            hjd, mag, magerr, ra, dec, fid = [], [], [], [], [], []

            idx = kk*n_exp + np.arange(0,n_exp)
            lc = [sourcedata[jj] for jj in idx]
            for jj, dic in enumerate(lc):
                if not exposures[jj][2] in program_ids: continue
                if (exposures[jj][2]==1) and (exposures[jj][1] > tmax): continue
                if not dic[2] == 0: continue
    
                hjd.append(exposures[jj][1])
                mag.append(dic[0])
                magerr.append(dic[1])
                ra.append(source[1])
                dec.append(source[2])
                fid.append(filt)

            hjd, mag, magerr = np.array(hjd),np.array(mag),np.array(magerr)
            ra, dec = np.array(ra), np.array(dec)
            fid = np.array(fid)

        elif matchfileType == 'kevin':
            LC=f['Objects'][source]
            hjd=LC[:,0]
            mag=LC[:,1]
            magerr=LC[:,2]
            RA=f['Objects'][source].attrs['RA']
            Dec=f['Objects'][source].attrs['Dec']
            ref_flux=f['Objects'][source].attrs['ref_r_flux']
            parallax=f['Objects'][source].attrs['parallax']
            parallaxerr=f['Objects'][source].attrs['parallax_error']
            bprp=f['Objects'][source].attrs['bp_rp']
            G=f['Objects'][source].attrs['G']
            pm=f['Objects'][source].attrs['pm']
 
            ra = RA*np.ones(hjd.shape)
            dec = Dec*np.ones(hjd.shape)
            fid = np.ones(hjd.shape)

        idx = np.where(~np.isnan(mag) & ~np.isnan(magerr))[0]
        hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
        fid = fid[idx]
        ra, dec = ra[idx], dec[idx]

        idx = np.where(magerr<=max_error)[0]
        hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
        fid = fid[idx]
        ra, dec = ra[idx], dec[idx]

        idx = np.argsort(hjd)
        hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
        fid = fid[idx]
        ra, dec = ra[idx], dec[idx]

        if doRemoveHC:
            idx = []
            for ii, t in enumerate(hjd):
                if ii == 0:
                    idx.append(ii)
                else:
                    dt = hjd[ii] - hjd[idx[-1]]
                    if dt >= 30.0*60.0/86400.0:
                        idx.append(ii)
            if len(idx) == 0: continue 
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            ra, decobj = ra[idx], dec[idx]
            fid = fid[idx]

        elif doHCOnly:
            idx = []
            for ii, t in enumerate(hjd):
                if ii == 0:
                    idx.append(ii)
                else:
                    dt = hjd[ii] - hjd[idx[-1]]
                    if dt >= 30.0*60.0/86400.0:
                        idx.append(ii)
            idx = np.setdiff1d(np.arange(len(hjd)), idx)
            if len(idx) == 0: continue
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

            bins = np.arange(np.floor(np.min(hjd)),
                             np.ceil(np.max(hjd)))
            hist, bin_edges = np.histogram(hjd, bins=bins)
            bins = (bin_edges[1:] + bin_edges[:-1])/2.0

            if len(hist) == 0: continue

            idx3 = np.argmax(hist)
            bin_start, bin_end = bin_edges[idx3], bin_edges[idx3+1]              
            idx = np.where((hjd >= bin_start) & (hjd <= bin_end))[0]

            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            ra, decobj = ra[idx], dec[idx]
            fid = fid[idx]

        if doSigmaClipping or doOutbursting:
            iqr = np.diff(np.percentile(mag,q=[25,75]))[0]
            idx = np.where(mag >= np.median(mag)-sigmathresh*iqr)[0]
            if doOutbursting and (len(idx) == len(mag)):
                continue
            if doSigmaClipping:
                hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
                ra, dec = ra[idx], dec[idx]
                fid = fid[idx]

        if doPercentile:
            if len(hjd) == 0: continue
            magmin, magmax = np.percentile(mag, percmin), np.percentile(mag, percmax)
            idx = np.where((mag >= magmin) & (mag <= magmax))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]
            ra, dec = ra[idx], dec[idx]

        if len(hjd) < min_epochs: continue

        lightcurve=(hjd,mag,magerr)
        lightcurves.append(lightcurve)
        filters.append(np.unique(fid).tolist())
        nlightcurves = 1

        if not kow is None:
            radius = 5
            qu = { "query_type": "cone_search", "query": {"object_coordinates": { "radec": {'test': [np.float64(np.median(ra)),np.float64(np.median(dec))]}, "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "Gaia_EDR3": { "filter": "{}", "projection": "{'parallax': 1, 'parallax_error': 1, 'phot_g_mean_mag': 1, 'phot_bp_mean_mag': 1, 'phot_rp_mean_mag': 1, 'phot_bp_mean_flux_over_error': 1, 'phot_rp_mean_flux_over_error': 1, 'ra': 1, 'dec': 1}"} } }}

            r = database_query(kow, qu, nquery = 10)

        coords = SkyCoord(ra=np.median(ra)*u.degree, 
                          dec=np.median(dec)*u.degree, frame='icrs')

        absmag, bp_rp = [np.nan, np.nan, np.nan], [np.nan, np.nan]

        if matchfileType == 'kevin':
            absmag = [G+5*(np.log10(np.abs(parallax))-2),G+5*(np.log10(np.abs(parallax+parallaxerr))-2)-(G+5*(np.log10(np.abs(parallax))-2)),G+5*(np.log10(np.abs(parallax))-2)-(G+5*(np.log10(np.abs(parallax-parallaxerr))-2))]
            bp_rp = [bprp, np.abs(bprp*0.01)]

        if not kow is None:
            if "data" in r:
                key2 = 'Gaia_EDR3'
                data2 = r["data"][key2]
                key = list(data2.keys())[0]
                data2 = data2[key]
                cat2 = get_catalog(data2)
                if len(cat2) > 0:
                    idx,sep,_ = coords.match_to_catalog_sky(cat2)
                    dat2 = data2[idx]
                else:
                    dat2 = {}
    
                if not "parallax" in dat2:
                    parallax, parallaxerr = None, None
                else:
                    parallax, parallaxerr = dat2["parallax"], dat2["parallax_error"]
                if not "phot_g_mean_mag" in dat2:
                    g_mag = None
                else:
                    g_mag = dat2["phot_g_mean_mag"]
    
                if not "phot_bp_mean_mag" in dat2:
                    bp_mag = None
                else:
                    bp_mag = dat2["phot_bp_mean_mag"]
    
                if not "phot_rp_mean_mag" in dat2:
                    rp_mag = None
                else:
                    rp_mag = dat2["phot_rp_mean_mag"]
    
                if not "phot_bp_mean_flux_over_error" in dat2:
                    bp_mean_flux_over_error = None
                else:
                    bp_mean_flux_over_error = dat2["phot_bp_mean_flux_over_error"]
    
                if not "phot_rp_mean_flux_over_error" in dat2:
                    rp_mean_flux_over_error = None
                else:
                    rp_mean_flux_over_error = dat2["phot_rp_mean_flux_over_error"]
    
                if not ((parallax is None) or (g_mag is None) or (bp_mag is None) or (rp_mag is None) or (rp_mean_flux_over_error is None) or (rp_mean_flux_over_error is None)):
                    absmag = [g_mag+5*(np.log10(np.abs(parallax))-2),g_mag+5*(np.log10(np.abs(parallax+parallaxerr))-2)-(g_mag+5*(np.log10(np.abs(parallax))-2)),g_mag+5*(np.log10(np.abs(parallax))-2)-(g_mag+5*(np.log10(np.abs(parallax-parallaxerr))-2))]
                    bp_rp = [bp_mag-rp_mag, 2.5/np.log(10) * np.hypot(1/bp_mean_flux_over_error, 1/rp_mean_flux_over_error)]
    
        for jj in range(nlightcurves):
            coordinate=(np.median(ra),np.median(dec))
            coordinates.append(coordinate)

            if matchfileType == 'forced':
                ids.append(source[0])
            elif matchfileType == 'kevin':
                ids.append(int(source[1:-1]))

            absmags.append(absmag)
            bp_rps.append(bp_rp)
    
            ra_hex, dec_hex = convert_to_hex(np.median(ra)*24/360.0,delimiter=''), convert_to_hex(np.median(dec),delimiter='')
            if dec_hex[0] == "-":
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
            else:
                objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
            names.append(objname)
    
        newbaseline = max(hjd)-min(hjd)
        if newbaseline>baseline:
            baseline=newbaseline

    end = time.time()
    loadtime = end - start
    print('Loaded %d lightcurves in %.5f seconds' % (len(lightcurves), loadtime))

    return [lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline]


def get_matchfile_original(f):
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
    if not "timeout" in qu:
        qu["timeout"] = DEFAULT_TIMEOUT
    cnt = 0
    while cnt < nquery:
        r = kow.query(query=qu)
        if (r is not None) and ("data" in r):
            break
        time.sleep(5)        
        cnt = cnt + 1
    return r


def BJDConvert(mjd, RA, Dec):
    times=mjd
    t = Time(times,format='mjd',scale='utc')
    t2=t.tdb
    c = SkyCoord(RA,Dec, unit="deg")
    Palomar=EarthLocation.of_site('palomar')
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

def sigma_model (mag, A, B, C, D):

    """ Calculating the corrected ZTF photometric uncertainties:
    According to the ZTF Data Explanatory Supplement (Section 6.8.1),
    photometric uncertainties in the matchfiles are estimated using a
    simple model:
    magerr = A + B*x + C*10^(0.4*x) + D*10^(0.8*x) for x <= 21.0
    magerr = b*x + c for x > 21.0
    According to Frank Masci (email from 17 Nov 2020), the "b" and "c"
    coefficients are not stored anywhere. So for objects fainter than 21
    I am assuming photometric uncertainties equal to those of a 21-mag
    object. Note that according to Frank, the model applies to the
    magnitude range 13 <= mag <= 21.
    """

    if isinstance(mag,(list,np.ndarray)):
        aux = np.clip(mag,None,21.0)
    else:
        aux = min((mag,21.0))
    tmp = pow(10.0,0.4*aux)
    magerr = A + B*aux + C*tmp + D*tmp**2
    return magerr
