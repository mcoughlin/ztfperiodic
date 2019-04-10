
import os, sys, glob, time
import optparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
from matplotlib import pyplot as plt

import astropy.io

from astroquery.vizier import Vizier

import astropy.table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.table import Table
from astropy.time import Time
from astroplan.plots import plot_airmass

from astroplan import Observer
from astroplan import FixedTarget
from astroplan.constraints import AtNightConstraint, AirmassConstraint
from astroplan import observability_table

from astroquery.simbad import Simbad

import ztfperiodic
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    #parser.add_option("-i","--infile",default="/Users/mcoughlin/Desktop/CE/CE_xray/obj.dat")
    #parser.add_option("-o","--outputDir",default="/Users/mcoughlin/Desktop/CE/CE_xray/")
    parser.add_option("-i","--infile",default="/Users/mcoughlin/Desktop/CE/CE_rosat/obj.dat")
    parser.add_option("-o","--outputDir",default="/Users/mcoughlin/Desktop/CE/CE_rosat/")
    parser.add_option("-c","--significance_cut",default=0.0,type=float)

    opts, args = parser.parse_args()

    return opts

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Parse command line
opts = parse_commandline()
outputDir = opts.outputDir
infile = opts.infile

lines = [line.rstrip('\n') for line in open(infile)]
names = ["objname", "ra", "dec", "period", "sig", "gmag", "col", "mag", "P_min", "name"]
data = astropy.io.ascii.read(infile,names=names)

period = np.array(data["period"])
P_min = np.array(data["P_min"])
sig = np.array(data["sig"])
idx = np.where(sig >= opts.significance_cut)[0]
period, P_min = period[idx], P_min[idx]

diff = np.abs(period-P_min)/np.max([period, P_min],axis=0)
idx1 = np.where(diff <= 0.2)[0]
diff = np.abs(2*period-P_min)/np.max([2*period, P_min],axis=0)
idx2 = np.where(diff <= 0.2)[0]
idx = np.hstack((idx1,idx2))
filename = os.path.join(outputDir,'minima.dat')
astropy.io.ascii.write(data[idx],filename)

plotName = os.path.join(outputDir,'minima.png')
plt.figure(figsize=(20,10))
plt.loglog(period*24.0, P_min*24.0, 'kx')
hrs = np.linspace(1,1000,1000)
plt.plot(hrs,hrs,'b-')
plt.plot(hrs,hrs*1.2,'b--')
plt.plot(hrs,hrs*0.8,'b--')
plt.xlim([1,1000])
plt.ylim([1,1000])
plt.xlabel('ZTF Period [hrs]')
plt.ylabel('Minimum Period [hrs]')
plt.savefig(plotName)
plt.close()

filenames = get_filepaths(outputDir)
filenames = [x for x in filenames if ("png" in x and not "minima" in x and not "obj" in x)]

sigs, ras, decs, periods = [], [], [], []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    sig, ra, dec, period = float(filenameSplit[0]), float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
    sigs.append(sig)
    ras.append(ra)
    decs.append(dec)
    periods.append(period)

sigs, ras, decs, periods = np.array(sigs), np.array(ras), np.array(decs), np.array(periods)

minimaDir = os.path.join(outputDir,'minima')
if not os.path.isdir(minimaDir):
    os.makedirs(minimaDir)

for ii in idx:
    row = data[ii]
    ra, dec, period, sig = row["ra"], row["dec"], row["period"], row["sig"]

    diff = np.sqrt((ras-ra)**2 + (decs-dec)**2)
    idx = np.where((diff <= 0.001) & (sigs >= 9))[0]
    idx2 = np.argmax(sigs[idx])

    filename = filenames[idx[idx2]]
    cp_command = "cp %s %s/%s"%(filename,minimaDir,filename.split("/")[-1])
    os.system(cp_command)

