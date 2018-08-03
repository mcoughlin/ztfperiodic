
import pandas as pd

with tables.open_file(f) as store:
    for tbl in store.walk_nodes("/", "Table"):
        if tbl.name in ["sourcedata", "transientdata"]:
            group = tbl._v_parent
            break
        srcdata = pd.DataFrame.from_records(store.root.matches.sourcedata.read_where('programid == 2'))
        srcdata.sort_values('matchid', axis=0, inplace=True)
        exposures = pd.DataFrame.from_records(store.root.matches.exposures.read_where('programid == 2'))
        merged = srcdata.merge(exposures, on="expid")
        for k in merged.matchid.unique():
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfmag
            err=df.psfmagerr
            obsHJD = df.hjd
            #programid =df.programid
            if len(x)>50:
                newbaseline = max(obsHJD)-min(obsHJD)
                if newbaseline>baseline:
                    baseline=newbaseline
            coordinate=(RA.values[0],Dec.values[0])
            coordinates.append(coordinate)
            lightcurve=(obsHJD.values,x.values,err.values)
            lightcurves.append(lightcurve)
    
    
    
