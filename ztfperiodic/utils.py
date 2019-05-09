
import os, sys
import optparse
import pandas as pd
import numpy as np
import tables
import glob
import time

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

def get_kowalski(ra, dec, kow, radius = 5.0, oid = None, program_ids = [2,3], min_epochs = 1):

    tmax = Time('2019-01-01T00:00:00', format='isot', scale='utc').jd

    qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(ra,dec), "cone_search_radius": "%.2f"%radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20190412": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'filter': 1}" } } }
    r = database_query(kow, qu, nquery = 10)

    if not "result_data" in r:
        print("Query for RA: %.5f, Dec: %.5f failed... returning."%(ra,dec)) 
        return {}

    key = list(r["result_data"].keys())[0]
    data = r["result_data"][key]
    key = list(data.keys())[0]
    data = data[key]

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

    return lightcurves

def get_kowalski_list(ras, decs, kow, program_ids = [2,3], min_epochs = 1,
                      max_error = 2.0, errs = None,
                      amaj=None, amin=None, phi=None,
                      doCombineFilt=False,
                      doRemoveHC=False):

    baseline=0
    cnt=0
    lightcurves, filters, coordinates = [], [], []
    
    if errs is None:
        errs = 5.0*np.ones(ras.shape)

    for ra, dec, err in zip(ras, decs, errs):
        if amaj is not None:
            ellipse = patches.Ellipse((ra, dec), amaj[cnt], amin[cnt],
                                      angle=phi[cnt]) 

        if np.mod(cnt,100) == 0:
            print('%d/%d'%(cnt,len(ras)))       
        ls = get_kowalski(ra, dec, kow, radius = err, oid = None,
                          program_ids = program_ids)
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

            newbaseline = max(hjd)-min(hjd)
            if newbaseline>baseline:
                baseline=newbaseline
        cnt = cnt + 1

    return lightcurves, coordinates, filters, baseline

def combine_lcs(ls):

    ras, decs = [], []
    hjds, mags, magerrs = [], [], []
    for ii, lkey in enumerate(ls.keys()):
        l = ls[lkey]
        if ii == 0:
            raobj, decobj = l["ra"], l["dec"]
            hjd, mag, magerr = l["hjd"], l["mag"]-np.median(l["mag"]), l["magerr"]
            fid = l["fid"]
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

    return {'combined': data}

def get_kowalski_bulk(field, ccd, quadrant, kow,
                      program_ids = [2,3], min_epochs = 1, max_error = 2.0,
                      num_batches=1, nb=0):

    tmax = Time('2019-01-01T00:00:00', format='isot', scale='utc').jd

    qu = {"query_type":"general_search","query":"db['ZTF_sources_20190412'].count_documents({'field':%d,'ccd':%d,'quad':%d})"%(field,ccd,quadrant)}
    r = database_query(kow, qu, nquery = 10)

    if not "result_data" in r:
        print("Query for field: %d, CCD: %d, quadrant %d failed... returning."%(field, ccd, quadrant))
        return [], [], [], []

    nlightcurves = r['result_data']['query_result']
    batch_size = np.ceil(nlightcurves/num_batches).astype(int)

    baseline=0
    cnt=0
    lightcurves, coordinates, filters = [], [], []

    objdata = {}
    #for nb in range(num_batches):
    for nb in [nb]:
        print("Querying batch number %d/%d..."%(nb, num_batches))

        qu = {"query_type":"general_search","query":"db['ZTF_sources_20190412'].find({'field':%d,'ccd':%d,'quad':%d},{'_id':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1}).skip(%d).limit(%d)"%(field,ccd,quadrant,int(nb*batch_size),int(batch_size))}
        r = database_query(kow, qu, nquery = 10)

        if not "result_data" in r:
            print("Query for batch number %d/%d failed... continuing."%(nb, num_batches))
            continue

        #qu = {"query_type":"general_search","query":"db['ZTF_sources_20190412'].find_one({})"}
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

                hjd.append(dic["hjd"])
                mag.append(dic["mag"])
                magerr.append(dic["magerr"])
                ra.append(dic["ra"])
                dec.append(dic["dec"])
                fid.append(filt)

            hjd, mag, magerr = np.array(hjd),np.array(mag),np.array(magerr)
            fid = np.array(fid)
            idx = np.where(~np.isnan(mag) & ~np.isnan(magerr))[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]

            idx = np.where(magerr<=max_error)[0]
            hjd, mag, magerr = hjd[idx], mag[idx], magerr[idx]
            fid = fid[idx]

            if len(hjd) < min_epochs: continue

            lightcurve=(hjd,mag,magerr)
            lightcurves.append(lightcurve)

            coordinate=(np.median(ra),np.median(dec))
            coordinates.append(coordinate)

            filters.append(np.unique(fid).tolist())

            newbaseline = max(hjd)-min(hjd)
            if newbaseline>baseline:
                baseline=newbaseline
            cnt = cnt + 1

    return lightcurves, coordinates, filters, baseline

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
