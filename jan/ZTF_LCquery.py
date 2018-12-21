import glob
import sys
import os
import argparse
import math
import numpy as np
import pandas as pd
import tables
from concurrent.futures import as_completed
from astropy.io import fits
import matplotlib.pyplot as plt
import ZTF_matchfile_query



if __name__ == "__main__":

    # pass commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",help='name of file with coordinate')
    parser.add_argument("-r",type=float,default=1.0,help='match radius in arcsec')    
    parser.add_argument("--delimiter",type=str,default=None,help='delimiter of coord file')    
    parser.add_argument("--verbose",'-v',action="store_true", default=False,help='verbose') 
    parser.add_argument("--save",'-s',action="store_true", default=False,help='save to txt files')    
    args = parser.parse_args()

    verbose = args.verbose

    # load the coordinates
    coords=np.loadtxt(args.filename,usecols=[0,1],skiprows=0,delimiter=args.delimiter)
    coords=np.atleast_2d(coords) # failsafe for the case of only 1 line in file

    # fetch matchIDs from database
    if verbose:
        print("Getting longmatchIDs from Kowalski")
    objIDlist = ZTF_matchfile_query.query_db(coords,r=args.r)
    objIDlist.sort() # sort
    if verbose:
        print("%d lightcurves found" %(len(objIDlist)))

    # put in a pd dataframe for easy splitting by unique matchfiles
    objID = pd.DataFrame([(s[:8],s[8:]) for s in objIDlist],dtype=int,columns=['fileID','matchID'])

    # loop over matchfiles
    info = []
    lightcurves = []
    for fileID in objID['fileID'].unique():
        # get an array of matchIDs for this particular matchfiles
        matchIDs = objID.loc[objID['fileID']==fileID]['matchID']
        
        # get info from name
        fID,rcID,filt = ZTF_matchfile_query.split_objID(fileID)
        filepath = ZTF_matchfile_query.make_filename(fID,rcID,filt)

        if verbose:
            print("%d in %s" %(matchIDs.size,os.path.basename(filepath)))

        # get data from matchfile
        n = 30 # the matchfiles cannot handle more than ~30
        for chunk in [matchIDs[i:i+n] for i in range(0,matchIDs.shape[0],n)]:
            c,l = ZTF_matchfile_query.query_matchfile_by_matchID(filepath,chunk)
            info.extend(np.c_[c,filt*np.ones_like(c[:,0])])
            lightcurves.extend(l)

    if args.save:
        # using the original input file, find all the sources and save them to a file
        from astropy.coordinates import match_coordinates_sky, SkyCoord
        from astropy import units as u
        c1 = SkyCoord(ra=coords[:,0]*u.degree, dec=coords[:,1]*u.degree)
        info = np.array(info)
        c2 = SkyCoord(ra=info[:,0]*u.degree, dec=info[:,1]*u.degree)
        idx, d2d, d3d = c2.match_to_catalog_sky(c1)
        idx_g = idx[d2d<1*u.arcsec]

	Nlc = np.bincount(idx,minlength=np.size(coords[:,0]))
	
        if verbose:
            print("# i: N        Ra       Dec")
            for k,(c,Nlc) in enumerate(zip(coords,Nlc)):
                print("% 3.d: %d % 9.4f % 9.4f" %(k+1,Nlc,c[0],c[1]))

        for i in np.unique(idx_g):
            # get coordinates
            ra,dec = coords[i]
            filename = "%08.4f_%.4f.dat" %(ra,dec)

            # get the data
            output = []
            for k in np.where(idx==i)[0]:
                lc = lightcurves[k]
                lc = np.c_[lc,info[k,2]*np.ones_like(lc[:,0])]
                output.append(lc)

            if lc.size:
                np.savetxt(filename,np.vstack(output))
