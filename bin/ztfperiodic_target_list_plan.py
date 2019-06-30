
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
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.time import Time
from astroplan.plots import plot_airmass

from astroplan import Observer
from astroplan import FixedTarget
from astroplan.constraints import AtNightConstraint, AirmassConstraint
from astroplan.scheduling import Transitioner
from astroplan import observability_table
from astroplan.scheduling import SequentialScheduler
from astroplan.scheduling import PriorityScheduler
from astroplan.scheduling import Schedule
from astroplan import ObservingBlock

from astroquery.simbad import Simbad

import ztfperiodic
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-i","--infile",default="/Users/mcoughlin/Desktop/Kevin/Candidates/obj.dat")
    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Desktop/Kevin/Candidates/plan.dat")

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

names = ["requestID", "programID", "objectID", "ra_hex", "dec_hex", "epoch", "ra_rate", "dec_rate", "mag", "exposure_time", "filter", "mode", "pi", "comment"]
targets = astropy.io.ascii.read(infile,names=names)
targets.add_row(["XXX0000",1,"1815f", "15:39:32.16", "50:27:38.8", 2000, 0.0, 0.0, 19.0, 3600, "FILTER_SLOAN_G", 9, "Coughlin", "100.0_0.00479"])

sigs, periods = [], []
coords, target = [], []
ras, decs = [], []
for row in targets:
    comment = row["comment"]
    commentSplit = comment.split("_")
    sig, period = float(commentSplit[0]), float(commentSplit[1])
    sigs.append(sig)
    periods.append(period)

    ra_hex, dec_hex = row["ra_hex"], row["dec_hex"]

    ra  = Angle(ra_hex, unit=u.hour).deg
    dec = Angle(dec_hex, unit=u.deg).deg

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    tar = FixedTarget(coord=coord, name=row["objectID"])
    coords.append(coord)
    target.append(tar)
    ras.append(ra)
    decs.append(dec)

targets["sig"] = sigs
targets["periods"] = periods
targets["coords"] = coords
targets["target"] = target
targets["ra"] = ras
targets["dec"] = decs

targets.sort("sig")
targets.reverse()
targets = astropy.table.unique(targets, keys=["objectID"])

location = EarthLocation.from_geodetic(-111.5967*u.deg, 31.9583*u.deg,
                                       2096*u.m)
kp = Observer(location=location, name="Kitt Peak",timezone="US/Arizona")

observe_time = Time.now()
observe_time = observe_time + np.linspace(deltat_start, deltat_end, 90)*u.hour
tstart, tend = observe_time[0],observe_time[-1]
frame = AltAz(obstime=observe_time, location=location)
global_constraints = [AirmassConstraint(max = opts.airmass, 
                                        boolean_constraint = False),
                      AtNightConstraint.twilight_civil()]

rise_time = kp.sun_rise_time(tstart, which=u'next', horizon=-12*u.deg).iso
set_time = kp.sun_set_time(tstart, which=u'next', horizon=-12*u.deg).iso

blocks = []
read_out = 10.0 * u.s
nexp = 1
for ii, target in enumerate(targets):
    bandpass = target["filter"]
    exposure_time = int(target["exposure_time"]) * u.s
    priority = target["sig"]

    b = ObservingBlock.from_exposures(target["target"],priority,
                                      exposure_time, nexp, read_out,
                                      configuration = {'filter': bandpass})
    blocks.append(b)

# Initialize a transitioner object with the slew rate and/or the
# duration of other transitions (e.g. filter changes)
slew_rate = 2.0*u.deg/u.second
transitioner = Transitioner(slew_rate,
                            {'filter':{'default': 10*u.second}})

# Initialize the sequential scheduler with the constraints and transitioner
prior_scheduler = PriorityScheduler(constraints = global_constraints,
                                    observer = kp,
                                    transitioner = transitioner)
# Initialize a Schedule object, to contain the new schedule
priority_schedule = Schedule(tstart, tend)

# Call the schedule with the observing blocks and schedule to schedule the blocks
prior_scheduler(blocks, priority_schedule)

fid = open(outfile,'w')
for schedule in priority_schedule.to_table():
    tar = schedule["target"]
    if tar == "TransitionBlock": continue

    idx = np.where(targets["objectID"] == tar)[0]
    target = targets[idx]
    filt = schedule["configuration"]["filter"]
    obsstart, obsend = Time(schedule["start time (UTC)"]), Time(schedule["end time (UTC)"])

    expt = int(schedule["duration (minutes)"]*60.0)

    c = SkyCoord(ra=target["ra"][0]*u.degree, dec=target["dec"][0]*u.degree,
                 frame='icrs')
    ra = c.ra.to_string(unit=u.hour, sep=':')
    dec = c.dec.to_string(unit=u.degree, sep=':')

    print('%s,%d,%s,%s,%s,%.1f,%.2f,%.2f,%.2f,%.0f,%s,%d,%s,%s'%(target["requestID"][0], target["programID"][0], target["objectID"][0], target["ra_hex"][0], target["dec_hex"][0], target["epoch"][0], target["ra_rate"][0], target["dec_rate"][0], target["mag"][0], target["exposure_time"][0], target["filter"][0], target["mode"][0], target["pi"][0], target["comment"][0]),file=fid,flush=True)

fid.close()

