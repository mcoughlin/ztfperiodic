
import os, sys, glob, time
import optparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
from matplotlib import pyplot as plt

from astroquery.vizier import Vizier

import astropy.table
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
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

    parser.add_option("-i","--infile",default="/Users/mcoughlin/Code/KP84/KevinPeriods/obj.dat")
    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Code/KP84/KevinPeriods/plan.dat")

    parser.add_option("-a","--airmass",default=2.0,type=float)
    parser.add_option("-s","--deltat_start",default=0.0,type=float)
    parser.add_option("-e","--deltat_end",default=24.0,type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()
infile = opts.infile
outfile = opts.outfile
airmass_limit = opts.airmass
deltat_start = opts.deltat_start
deltat_end = opts.deltat_end

observed = ["ZTFJ0713-0126"]

names = ["objname", "ra", "dec", "period", "sig", "gmag", "col", "mag", "P_min", "name"]
data = astropy.io.ascii.read(infile,names=names)
data.add_row(["1815f", 234.884000, 50.460778, 7.2*60.0, 100, -1, -1, -1, -1, "NA"])
data.add_row(["ZTFJ1858-2024", 284.5247891, -20.4135043, 8.6*60.0, 100, -1, -1, -1, -1, "NA"])
data.sort("sig")
data.reverse()
data = astropy.table.unique(data, keys=["objname"])

location = EarthLocation.from_geodetic(-111.5967*u.deg, 31.9583*u.deg,
                                       2096*u.m)
kp = Observer(location=location, name="Kitt Peak",timezone="US/Arizona")

observe_time = Time.now()
observe_time = observe_time + np.linspace(deltat_start, deltat_end, 90)*u.hour
tstart, tend = observe_time[0],observe_time[-1]
frame = AltAz(obstime=observe_time, location=location)

rise_time = kp.sun_rise_time(tstart, which=u'next', horizon=-12*u.deg).iso
set_time = kp.sun_set_time(tstart, which=u'next', horizon=-12*u.deg).iso

mode = 1
filt = 'g'
exposure_time = 3600
exposure_segment = 1

fid = open(outfile,'w')
fid.write('# name ra dec mode filter exposure_time exposure_segment window_start window_end priority\n')
for row in data:
    if row["objname"] in observed: continue

    coord = SkyCoord(ra=row["ra"]*u.deg, dec=row["dec"]*u.deg)
    tar = FixedTarget(coord=coord, name="tmp")

    altazs = coord.transform_to(frame)
    airmass = np.array(altazs.secz)

    airmass_peak = np.argmin(np.abs(airmass-1.0))
    airmass_dx = np.abs(airmass-airmass_limit)

    airmass_min, airmass_max = np.argmin(airmass_dx[:airmass_peak]), np.argmin(airmass_dx[airmass_peak+1:]) + airmass_peak+1

    if set_time > observe_time[airmass_min]:
        window_start = str(set_time).replace(" ","T")
    else:
        window_start = str(observe_time[airmass_min]).replace(" ","T")

    if rise_time < observe_time[airmass_max]:
        window_end = str(rise_time).replace(" ","T")
    else:
        window_end = str(observe_time[airmass_max]).replace(" ","T")

    objname, ra, dec = row['objname'], row['ra'], row['dec']
    priority = row['sig']

    print('%s %.5f %.5f %d %s %d %.1f %s %s %.5f'%(objname, ra, dec, mode, filt, exposure_time, exposure_segment, window_start, window_end, priority),file=fid,flush=True)
fid.close()

