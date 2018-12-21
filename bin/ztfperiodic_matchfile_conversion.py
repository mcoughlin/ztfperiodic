
import os, sys
import glob
import optparse
from pathlib import Path

import tables
import pandas as pd
import numpy as np
import h5py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="/media/Data/mcoughlin/Matchfiles")
    parser.add_option("-d","--dataDir",default="/media/Data2/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

dataDir = opts.dataDir
outputDir = opts.outputDir
directory="%s/*/*/*"%opts.dataDir

for f in glob.iglob(directory):
    #if not "ztf_000283_zr_c05_q3_match.pytable" in f: continue 
    fileend = "/".join(f.split("/")[-3:])
    fnew = "%s/%s"%(outputDir,fileend)
    filedir = "/".join(fnew.split("/")[:-1])
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    fnew = fnew.replace("pytable","h5")
    if os.path.isfile(fnew): continue

    print("Running %s"%fnew)

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
            continue

        matchids = np.array(merged.matchid)
        values, indices, counts = np.unique(matchids, return_counts=True,return_inverse=True)       
        idx = np.where(counts>50)[0]

        if len(idx) == 0:
            Path(fnew).touch()
            continue

        f = h5py.File(fnew, 'w')

        matchids = np.unique(matchids[idx])
        nmatchids = len(idx)

        cnt = 0
        for k in matchids:
            if np.mod(cnt,100) == 0:
                print('%d/%d'%(cnt,len(matchids)))
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd

            if len(x) < 50: continue

            data = np.vstack((obsHJD,x,err))
            key = "%d_%.10f_%.10f"%(k,RA.values[0],Dec.values[0])
            f.create_dataset(key, data=data, dtype='float64', compression="gzip",shuffle=True)
            cnt = cnt + 1

        f.close()

