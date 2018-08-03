#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:05:27 2017

@author: kburdge
"""
import pandas as pd
import tables
import glob
for f in glob.iglob("directory"):
    with tables.open_file(f) as store:
        print(f)
        for tbl in store.walk_nodes("/", "Table"):
            if tbl.name in ["sourcedata", "transientdata"]:
                group = tbl._v_parent
                continue
        srcdata = pd.DataFrame.from_records(group.sourcedata[:])
        srcdata.sort_values('matchid', axis=0, inplace=True)
        exposures = pd.DataFrame.from_records(group.exposures[:])
        merged = srcdata.merge(exposures, on="expid")
        for k in merged.matchid.unique():
            print(len(merged.matchid.unique()))
            df = merged[merged['matchid'] == k]
            RA = df.ra
            Dec = df.dec
            x = df.psfflux
            err=df.psffluxerr
            hjd = df.hjd

