#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
from functools import reduce
import traceback

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

from joblib import Parallel, delayed
from tqdm import tqdm

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    #parser.add_option("--doUpload",  action="store_true", default=False)

    parser.add_option("-o","--outfile",default="/home/mcoughlin/ZTF/labels/golden_sets/delta_scuti.dat")
    parser.add_option("-l","--labels",default="/home/mcoughlin/ZTF/labels/dataset.d12.csv")

    parser.add_option("-t","--type",default="Delta Scu")

    parser.add_option("-n","--N",default=100,type=int)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

data_out = pd.read_csv(opts.labels)
idx = np.where(data_out[opts.type] == 1)[0]
idx = np.random.choice(idx, size=opts.N)

ras, decs = data_out.iloc[idx]["ra"], data_out.iloc[idx]["dec"]
fid = open(opts.outfile, 'w')
fid.write('ra,dec\n')
for ra, dec in zip(ras, decs):
    fid.write('%.10f,%.10f\n' % (ra, dec))
fid.close()
