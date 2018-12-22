import glob
import sys
import os
import argparse
import math
import numpy as np
import pandas as pd
import tables
from concurrent.futures import as_completed
from astropy.io import fits
import matplotlib.pyplot as plt
import pymongo



def connect_to_db(_config):
    """ setup the connection to the matchID database
    
    Parameters
    ----------

    _config : dict
        configuration

    Returns
    -------
    _client : ???
        ???
    _db : ???
        object to interface with db
    

    """

    _client = pymongo.MongoClient(host=_config['database']['host'], port=_config['database']['port'])
    # grab main database:
    _db = _client[_config['database']['db']]

    # authenticate
    _db.authenticate(_config['database']['user'], _config['database']['pwd'])

    return _client, _db



def query_db(coords,r=3.,catname='ZTF_20181219'):
    """ given a set of coordinates, get the matchIDs from the database
        Note that everything that is in the aperture is returned.
    
    Parameters
    ----------

    coords : 2d-array
        2d-array with Ra,dec in degrees
    r : float
        matching radius
    coords : str
        catalog name

    Returns
    -------
    matchIDs : 1d-array
        an array with matchIDs (int64)
    

    """

    # db config
    config = {'database':
                  {
                      'host': 'kowalski.caltech.edu',
                      'port': 27017,
                      'db': 'ztf',
                      'user': 'ztf_reader',
                      'pwd': 'VerySecretPa$$word'
                  }
    }
    _, db = connect_to_db(config)

    # cone search radius must be in radians:
    cone_search_radius = r * np.pi / 180.0 / 3600.

    # construct query: RA and Dec's must be in degrees; RA must c (-180, 180]
    query = {'$or': []}
    for ra,dec in coords:
        # convert
        #_ra, _dec = radec_str2geojson(*obj_crd)
        obj_query = {'coordinates.radec_geojson': {
            '$geoWithin': {'$centerSphere': [[ra-180., dec], cone_search_radius]}}}
        query['$or'].append(obj_query)

    # execute query: [return only id's as an example]
    cursor = db[catname].find(query, {'_id': 1})
    
    # put the matchIDs in an array
    matchIDs = np.array([i['_id'] for i in list(cursor)],dtype='str')

    return(matchIDs)



def split_objID(ID):
    """ get the long matchID and split it into fieldID, CCDid, filter, matchid
    """
    ID = str(ID)
    fID = int(ID[1:5]) # get fieldID
    rcID = int(ID[5:7]) # get ccdid
    filt = int(ID[7])
    return np.r_[fID,rcID,filt]



def make_filename(fID,rcID,filt,filedir="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/"):
    """for a fieldID, CCDid, and filter make the filename"""
    
    filtdict = {1:'g',2:'r',3:'i',}
    floor50 = lambda fID: int(np.floor((fID-1)*1./50)*50)+1
    ceil50 = lambda fID: int(np.ceil(fID*1./50)*50)
    quadrant = lambda rcID: rcID%4+1
    ccdid = lambda rcID: rcID//4+1 #int((math.ceil((int(rcID)+1) / 4)))

    filename = filedir
    filename += 'rc%02d/' %rcID
    filename += 'fr%06d-' %floor50(fID)
    filename += '%06d/' %ceil50(fID)
    filename += 'ztf_%06d_' %fID
    filename += 'z%s_' %filtdict[filt]
    filename += 'c%02d_' %ccdid(rcID)
    filename += 'q%d_' %quadrant(rcID)
    filename += 'match.pytable'

    return filename



def query_matchfile_by_matchID(f,matchIDs,stats=False,flags=False):
    """ Get lightcurves and stats from a matchfile for a set of matchIDs

    Parameters
    ----------
    f : str
        matchfile name
    matchIDs : 1D-array
        list of matchIDs to get
    return_flags : bool
        return the quality flags for the lightcurves
    return_stats : bool
        return the stats for the matchID

    Returns
    -------
    lightcurves : list
        a list of array with (t,y,dy)
    flags : list
        a list of tuples with the quality flags for the epochs
    stats : 1D-array
        the stats for the selected stars

    """

    with tables.open_file(f) as store:
        # get only data for the requested matchIDs
        conditions = ('|'.join(['(matchid=={0})'.format(value) for value in matchIDs]))

        # get the lightcurve data
        rows = store.root.matches.sourcedata.read_where(conditions)
        srcdata = pd.DataFrame.from_records(rows)
        srcdata.sort_values('matchid', axis=0, inplace=True)

        # got epoch info
        exposures = pd.DataFrame.from_records(store.root.matches.exposures.read_where('programid>1'))
    
        # merge
        merged = srcdata.merge(exposures, on="expid")

        # get coords if requested        
        coords = np.array(store.root.matches.sources.read_where(conditions)[['ra','dec']])
        coords = pd.DataFrame.from_records(coords)

        # lightcurves
        lightcurves = [merged[merged['matchid'] == mID][['obshjd','psfmag','psfmagerr']].values for mID in matchIDs]


    return coords.values,lightcurves



def query_matchfile(f,conditions="(nobs > 100)",photquality='good',
        t0=2458119.5,
        calibration_variance = 0.00,
        return_flags=False):
    """ Get lightcurves and stats from a matchfile

    Parameters
    ----------
    f : str
        matchfile name
    conditions : float
        selection criteria on the ZTF stats table
    photquality : bool
        return only good photometry; see code for what is good
    t0 : float
        subtract this value from all HJD values in the LCs
        default value is JD for 2018-01-01 00:00:00
    calibration_variance: float
        additional variance which is added to all lightcurves
        new_sigma = np.sqrt(sigma**2+calibration_variance**2)
    return_flags : bool
        return the quality flags for the lightcurves

    Returns
    -------
    sources : 1D-array
        the stats for the selected stars
    lightcurves : list
        a list of tuples with (t,y,dy)
    flags : list
        a list of tuples with the quality flags for the epochs

    """

    with tables.open_file(f) as store:
        # select only objects with N>100 from stats table
        sources = store.root.matches.sources.read_where(conditions)
        print(np.size(np.unique(sources['matchid'])))

        # get the matchIDs
        allids = store.root.matches.sourcedata.cols.matchid[:]
        # get the indices for targets of interest
        idx = np.where(np.in1d(allids, sources['matchid']))

        # get the lightcurves
        rows = store.root.matches.sourcedata.read_coordinates(idx) 
        rows = rows[rows['programid'] > 1] # drop MSIP data
        print('WARNING: programID==1 is dropped')

        # do quality cuts on the data
        if str.lower(photquality)=='good':
            rows = rows[rows['relphotflags'] < 4] # drop bad photometry
        else:
            pass


        # load into dataframe
        srcdata = pd.DataFrame.from_records(rows)
        srcdata.sort_values('matchid', axis=0, inplace=True)

        # load exposures
        exposures = pd.DataFrame.from_records(
            store.root.matches.exposures.read_where('programid > 1'))
        # merge the lightcurves and exposures
        merged = srcdata.merge(exposures, on="expid")

        # get matchids 
        matchids = np.array(merged.matchid)
        stats = sources[np.isin(sources['matchid'],matchids)]

        # count number of targets etc
        values, indices, counts = np.unique(merged.matchid, 
                                            return_counts=True,
                                            return_inverse=True)


        # TODO: THIS PART IS SLOW
        # print number of stars
        print('Loading %d lightcurves' %(np.size(np.unique(indices))))

        if return_flags:
            # transform the lightcurves to a list of tuples [(t,y,dy),...]
            lightcurves = []
            flags = []
            for k,mID in enumerate(np.unique(matchids)):
                if not k%100:
                    print(k)
                df = merged[merged['matchid'] == mID]
                if calibration_variance > 0:
                    lightcurve=(df.hjd.values-t0,
                                df.psfmag.values,
                                np.sqrt(df.psfmagerr.values**2+calibration_variance**2))
                else:
                    lightcurve=(df.hjd.values-t0,
                                df.psfmag.values,
                                df.psfmagerr.values)
                flag=(df.catflags.values, #Catalog flags from PSF-fitting catalog
                        df.chi.values, #Chi-squared
                        df.relphotflags, # Relative photometry flags
                        df.sharp, # Sharpness of source
                        df.snr, #Signal-to-noise ratio
                        df.diq, #Derived image quality 
                        df.infobits)
                lightcurves.append(lightcurve)
                flags.append(flag)
            print('Done')

            return stats,lightcurves,flags

        else:
            # transform the lightcurves to a list of tuples [(t,y,dy),...]
            lightcurves = []
            for k,mID in enumerate(np.unique(matchids)):
                if not k%100:
                    print(k)
                df = merged[merged['matchid'] == mID]
                lightcurve=(df.hjd.values-t0,df.psfmag.values,df.psfmagerr.values)
                lightcurves.append(lightcurve)
            print('Done')

            return stats,lightcurves









            




