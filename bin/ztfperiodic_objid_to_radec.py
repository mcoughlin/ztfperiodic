#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import json
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

import astropy
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import ascii, fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import combine_lcs

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-u","--user")
    parser.add_option("-w","--pwd")

    parser.add_option("--objid",default=10299092006866,type=int)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
objid = opts.objid

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

tmp, features = get_kowalski_features_objids([objid], kow)
if len(features) == 0:
    exit(0)

period = features.period.values[0]
amp = features.f1_amp.values[0]
lightcurves, coordinates, filters, ids, absmags, bp_rps, names, baseline = get_kowalski_objids([objid], kow)

ra, dec = coordinates[0][0], coordinates[0][1]
print(ra, dec)
print(period)


