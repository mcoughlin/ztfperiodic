#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
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

from panoptes_client import Panoptes, Project, SubjectSet, Subject, Workflow

from ztfperiodic.utils import convert_to_hex
from ztfperiodic.utils import get_kowalski
from ztfperiodic.utils import get_kowalski_features_objids 
from ztfperiodic.utils import get_kowalski_classifications_objids
from ztfperiodic.utils import get_kowalski_objids
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import combine_lcs
from ztfperiodic.zooniverse import ZooProject

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

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/classifications")

    parser.add_option("--zooniverse_user")
    parser.add_option("--zooniverse_pwd")
    parser.add_option("--zooniverse_id",default=13032,type=int)

    parser.add_option("-w","--workflow",default=16000,type=int)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)
classFileDir = os.path.join(outputDir,'files')
if not os.path.isdir(classFileDir):
    os.makedirs(classFileDir)

zoo = ZooProject(username=opts.zooniverse_user,
                 password=opts.zooniverse_pwd,
                 project_id=opts.zooniverse_id) 
ans = zoo.get_answers(opts.workflow)

last_id = -1
classFiles = glob.glob(os.path.join(classFileDir,'*.h5'))
for classFile in classFiles:
    classFileName = classFile.split('/')[-1].split('.h5')[0].split('-')
    classifications_id_min = int(classFileName[0])
    classifications_id_max = int(classFileName[1])
    last_id = np.max([last_id, classifications_id_max])

if last_id < 0:
    last_id = None

classifications = zoo.parse_classifications(last_id=last_id)
classifications_id_min = np.min(classifications.id)
classifications_id_max = np.max(classifications.id)

classification_file = os.path.join(classFileDir,
                                   '%d-%d.h5' % (classifications_id_min,
                                                 classifications_id_max))
classifications.to_hdf(classification_file, key='df', mode='w') 
