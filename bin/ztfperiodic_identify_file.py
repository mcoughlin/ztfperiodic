#!/usr/bin/env python

import os, sys
import time
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py
from astropy.table import Table

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy import units as u
from astropy.coordinates import SkyCoord

import ztfperiodic
from ztfperiodic.period import CE
from ztfperiodic.lcstats import calc_stats
from ztfperiodic.utils import angular_distance
from ztfperiodic.utils import convert_to_hex
from ztfperiodic.periodsearch import find_periods

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_Fermi_GCE_LS_AOV/catalog/crossmatch")

    parser.add_option("-r", "--ra", default=237.3234518, type=float)
    parser.add_option("-d", "--declination", default=39.8249067, type=float)
    parser.add_option("--radius", default=3.0, type=float)

    parser.add_option("-u", "--user")
    parser.add_option("-w", "--pwd")

    parser.add_option("--objid", default=10563221000366, type=int)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

kow = Kowalski(username=opts.user, password=opts.pwd)

#qu = { "query_type": "cone_search", "object_coordinates": { "radec": "[(%.5f,%.5f)]"%(opts.ra,opts.declination), "cone_search_radius": "%.2f"%opts.radius, "cone_search_unit": "arcsec" }, "catalogs": { "ZTF_sources_20191101": { "filter": "{}", "projection": "{'data.hjd': 1, 'data.mag': 1, 'data.magerr': 1, 'data.programid': 1, 'data.maglim': 1, 'data.ra': 1, 'data.dec': 1, 'filter': 1}" } } }
qu = {"query_type":"general_search","query":"db['ZTF_sources_20191101'].find({'_id':%d},{'_id':1,'field':1,'ccd':1,'quad':1,'data.programid':1,'data.hjd':1,'data.mag':1,'data.magerr':1,'data.ra':1,'data.dec':1,'filter':1,'data.catflags':1})"%(opts.objid)}
r = ztfperiodic.utils.database_query(kow, qu, nquery = 10)

h5names = ["objid", "ra", "dec", "period", "sig", "pdot",
           "stats0", "stats1", "stats2", "stats3", "stats4",
           "stats5", "stats6", "stats7", "stats8", "stats9",
           "stats10", "stats11", "stats12", "stats13", "stats14",
           "stats15", "stats16", "stats17", "stats18", "stats19",
           "stats20", "stats21", "stats22", "stats23", "stats24",
           "stats25", "stats26", "stats27", "stats28", "stats29",
           "stats30", "stats31", "stats32", "stats33", "stats34",
           "stats35"]

datas = r["result_data"]["query_result"]
for ii, data in enumerate(datas):
    field, ccd, quadrant = data["field"], data["ccd"], data["quad"]
    catalogDir = os.path.join(opts.outputDir,"%d_%d_%d*"%(field, ccd, quadrant))
    h5files = sorted(glob.glob(catalogDir))

    for h5file in h5files:
        with h5py.File(h5file, 'r') as f:
            name = f['names'].value
            filters = f['filters'].value
            stats = f['stats'].value
        data_tmp = Table(rows=stats, names=h5names)
        data_tmp['name'] = name
        data_tmp['filt'] = filters

        idx = np.where(data_tmp["objid"].astype(int) == opts.objid)[0]
        if len(idx) == 0: continue
        print(h5file,data_tmp[idx]["period"][0], data_tmp[idx]["sig"][0])
