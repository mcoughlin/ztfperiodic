
import os, sys
import glob

import tables
import pandas as pd
import numpy as np
import h5py


dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/"
directory="%s/*/*/*"%dataDir

dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc18/fr000251-000300/"
directory="%s/*"%dataDir

#dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc02/fr000301-000350/"
#directory="%s/*"%dataDir

outputDir = "../output/matchfile"
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

for f in glob.iglob(directory):
    #if not "ztf_000283_zr_c05_q3_match.pytable" in f: continue 
    print(f)
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
            continue

        fname = f.split("/")[-1].replace("pytable","h5") 
        filename = os.path.join(outputDir,fname)
        f = h5py.File(filename, 'w')

        matchids = np.unique(matchids[idx])
        nmatchids = len(idx)

        for k in matchids:
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd            

            data = np.vstack((obsHJD,x,err))
            key = "%d_%.10f_%.10f"%(k,RA.values[0],Dec.values[0])
            print(key)
            f.create_dataset(key, data=data, dtype='f', compression="gzip",shuffle=True)
        f.close()

        print(stop)
