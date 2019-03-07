import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.xmatch import XMatch
from astropy.table import Table
import pickle
import os

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

def BSC5_brightstardist(ra,dec):
    # match the catalog to the Yale Bright Star Catalog
    catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    BSC5 = np.genfromtxt('bsc5_J2000.tsv',delimiter='|',skip_header=43,usecols=(0,1,4))
    BSC5 = BSC5[~np.isnan(BSC5).any(axis=1)] # remove lines with nans
    c_BSC5 = SkyCoord(ra=BSC5[:,0]*u.degree, dec=BSC5[:,1]*u.degree, frame='icrs')
    idx,sep,_ = catalog.match_to_catalog_sky(c_BSC5)
    return  np.c_[sep.arcsec,BSC5[idx,2]]

def Gaia_brightstardist(ra,dec):
    catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    Gaia = np.genfromtxt('Gaia_brightstars.tsv',delimiter='|')
    Gaia = Gaia[~np.isnan(Gaia).any(axis=1)] # remove lines with nans
    c_Gaia = SkyCoord(ra=Gaia[:,0]*u.degree, dec=Gaia[:,1]*u.degree, frame='icrs')
    idx,sep,_ = catalog.match_to_catalog_sky(c_Gaia)
    return np.c_[sep.arcsec,Gaia[idx,2]]

