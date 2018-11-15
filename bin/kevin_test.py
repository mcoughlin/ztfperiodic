
import os, sys
import glob

import tables
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from pdtrend import FMdata, PDTrend

dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/"
directory="%s/*/*/*"%dataDir

dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc18/fr000251-000300/"
directory="%s/*"%dataDir

#dataDir = "/media/Data/Matchfiles/ztfweb.ipac.caltech.edu/ztf/ops/srcmatch/rc02/fr000301-000350/"
#directory="%s/*"%dataDir

outputDir = "../output/detrend"
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

lightcurves=[]
baseline=0
p=0
coordinates=[]

for f in glob.iglob(directory):
    #if not "ztf_000283_zr_c05_q3_match.pytable" in f: continue 
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

        fname = f.split("/")[-1].replace("pytable","png") 

        matchids = matchids[idx]
        nmatchids = len(idx)

        lcs, times = [], []
        #for k in matchids:
        for k in matchids[:50]:
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd            

            print(np.min(obsHJD), np.max(obsHJD))
         
            if not len(x)>50:
                continue
 
            newbaseline = max(obsHJD)-min(obsHJD)
            if newbaseline>baseline:
                baseline=newbaseline
            coordinate=(RA.values[0],Dec.values[0])
            coordinates.append(coordinate)
            lightcurve=(obsHJD.values,x.values,err.values)
            lightcurves.append(lightcurve)
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

        fig = plt.figure(figsize=(30,10))
        ax = fig.add_subplot(1, 1, 1)
        colors=cm.rainbow(np.linspace(0,1,len(lcs)))

        # Detrend each light curve.
        cnt = 0
        for k,lc in zip(matchids[:50],lcs[:50]):
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

            bins = np.linspace(np.min(vals),np.max(vals),11)
            hist, bin_edges = np.histogram(vals, bins=bins, density=True)
            bins = (bin_edges[1:] + bin_edges[:-1])/2.0
            plt.plot(bins, hist, color = colors[cnt], linestyle='-', drawstyle='steps')

            bins = np.linspace(np.min(x.values),np.max(x.values),11)
            hist, bin_edges = np.histogram(x.values, bins=bins, density=True)
            bins = (bin_edges[1:] + bin_edges[:-1])/2.0
            plt.plot(bins, hist, color = colors[cnt], linestyle='--', drawstyle='steps')

            #plt.scatter(obsHJD.values,x.values,s=20,marker='x',color=colors[cnt])
            #plt.scatter(obsHJD.values,vals,s=20,marker='o',color=colors[cnt])

            cnt = cnt+1

        #ax.set_xscale('log')
        ax.set_yscale('log')

        plotName = os.path.join(outputDir,fname)
        plt.savefig(plotName)
        plt.close()

