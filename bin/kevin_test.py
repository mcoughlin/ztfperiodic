
import os, sys
import glob

import tables
import pandas as pd
import numpy as np

from pdtrend import FMdata, PDTrend

dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/"
directory="%s/*/*/*"%dataDir

lightcurves=[]
baseline=0
p=0
coordinates=[]

for f in glob.iglob(directory):
    with tables.open_file(f) as store:
        for tbl in store.walk_nodes("/", "Table"):
            if tbl.name in ["sourcedata", "transientdata"]:
                group = tbl._v_parent
                break
        srcdata = pd.DataFrame.from_records(store.root.matches.sourcedata.read_where('programid == 2'))
        srcdata.sort_values('matchid', axis=0, inplace=True)
        exposures = pd.DataFrame.from_records(store.root.matches.exposures.read_where('programid == 2'))
        merged = srcdata.merge(exposures, on="expid")

        if len(merged.matchid.unique()) == 0:
            continue

        matchids = np.array(merged.matchid)
        values, indices, counts = np.unique(matchids, return_counts=True,return_inverse=True)       
        idx = np.where(counts>50)[0]

        if len(idx) == 0:
            continue
 
        matchids = matchids[idx]
        nmatchids = len(idx)

        lcs, times = [], []
        for k in matchids:
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd            
         
            if not len(x)>50:
                continue
 
            newbaseline = max(obsHJD)-min(obsHJD)
            if newbaseline>baseline:
                baseline=newbaseline
            coordinate=(RA.values[0],Dec.values[0])
            coordinates.append(coordinate)
            lightcurve=(obsHJD.values,x.values,err.values)
            lightcurves.append(lightcurve)
            print(p,len(x.values))
            p=p+1

            idx = np.argsort(obsHJD.values)
            lcs.append(x.values[idx])
            times.append(obsHJD.values[idx])

        # Filling missing data points.
        fmt = FMdata(lcs, times, n_min_data=50)
        results = fmt.run()
        lcs = results['lcs']
        epoch = results['epoch']
        # Create PDT instance.
        pdt = PDTrend(lcs,dist_cut=0.6,n_min_member=5)
        # Find clusters and then construct master trends.
        pdt.run()

        # Detrend each light curve.
        for k,lc in zip(matchids,lcs):
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd

            if not len(x)>50:
                continue

            newbaseline = max(obsHJD)-min(obsHJD)
            if newbaseline>baseline:
                baseline=newbaseline
            coordinate=(RA.values[0],Dec.values[0])
            coordinates.append(coordinate)

            vals = np.interp(obsHJD.values,epoch,pdt.detrend(lc))          
            lightcurve=(obsHJD.values,vals,err.values)
            lightcurves.append(lightcurve)

        print(lightcurves)
        print(stop) 

