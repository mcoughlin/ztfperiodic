
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

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Code/KP84/KevinPeriods/NewSearch/obj.dat")
    parser.add_option("-i","--inputDir",default="/Users/mcoughlin/Code/KP84/KevinPeriods/NewSearch/ForFollowUp/")
    parser.add_option("-z","--ztfperiodicInputDir",default="../input")

    parser.add_option("--doXray",  action="store_true", default=False)
    parser.add_option("--doCheckObservable",  action="store_true", default=False)
    parser.add_option("--doMagnitudeCut",  action="store_true", default=False)
   
    parser.add_option("-m","--magnitude",default=16.0,type=float)

    parser.add_option("-s","--deltat_start",default=0.0,type=float)
    parser.add_option("-e","--deltat_end",default=24.0,type=float)

    opts, args = parser.parse_args()

    return opts

def convert_to_hex(val, delimiter=':', force_sign=False):
    """
    Converts a numerical value into a hexidecimal string

    Parameters:
    ===========
    - val:           float
                     The decimal number to convert to hex.

    - delimiter:     string
                     The delimiter between hours, minutes, and seconds
                     in the output hex string.

    - force_sign:    boolean
                     Include the sign of the string on the output,
                     even if positive? Usually, you will set this to
                     False for RA values and True for DEC

    Returns:
    ========
    A hexadecimal representation of the input value.
    """
    s = np.sign(val)
    s_factor = 1 if s > 0 else -1
    val = np.abs(val)
    degree = int(val)
    minute = int((val  - degree)*60)
    second = (val - degree - minute/60.0)*3600.
    if degree == 0 and s_factor < 0:
        return '-00{2:s}{0:02d}{2:s}{1:.2f}'.format(minute, second, delimiter)
    elif force_sign or s_factor < 0:
        deg_str = '{:+03d}'.format(degree * s_factor)
    else:
        deg_str = '{:02d}'.format(degree * s_factor)
    return '{0:s}{3:s}{1:02d}{3:s}{2:.2f}'.format(deg_str, minute, second, delimiter)

def ps1_query(ra_deg, dec_deg, rad_deg, maxmag=25,
               maxsources=1):
    """
    Query Pan-STARRS @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field
                                          radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RAJ2000', 'DEJ2000',
                             'gmag','rmag','imag','zmag','ymag'],
                    column_filters={"gmag":
                                    ("<%f" % maxmag),
                                   "imag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    return vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="II/349/ps1")[0]

# Parse command line
opts = parse_commandline()
inputDir = opts.inputDir
outfile = opts.outfile
ztfperiodicInputDir = opts.ztfperiodicInputDir
deltat_start = opts.deltat_start
deltat_end = opts.deltat_end

if opts.doXray:
    xrayfile = os.path.join(ztfperiodicInputDir,'xray.dat')
    xray = np.genfromtxt(xrayfile, dtype="S20,f8,f8,f8,f8,f8",names=['names','ras','decs','errs','appfluxes','absfluxes'], delimiter=" ")

observed = ["ZTFJ20071648","ZTFJ19274408","ZTFJ18320856","ZTFJ19385841","ZTFJ19541818","ZTFJ18492550","ZTFJ18390938","ZTFJ17182524","ZTFJ17483237","ZTFJ18174120","ZTFJ19244707","ZTFJ20062220","ZTFJ19243104","ZTFJ1913-1205","ZTFJ1909-0654","ZTFJ18494132","ZTFJ18461355","ZTFJ1913-1105","ZTFJ1921-0815","ZTFJ1905-1910","ZTFJ1843-2041","ZTFJ1924-1258","ZTFJ1914-0718","ZTFJ1909-1437","ZTFJ18261000","ZTFJ18221209","ZTFJ1903-1730","ZTFJ1903-0721","ZTFJ1903-0721","ZTFJ18562916","ZTFJ18451515","ZTFJ18321130"]
observed = []

fid = open(outfile,'w')
filenames = glob.glob(os.path.join(inputDir,"*.png"))
sigs = []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    sig, ra, dec, period = float(filenameSplit[0]), float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
 
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    tar = FixedTarget(coord=coord, name="tmp")

    if opts.doXray:
        idx1 = np.where(np.abs(xray["ras"] - ra)<=2e-3)[0]
        idx2 = np.where(np.abs(xray["decs"] - dec)<=2e-3)[0]
        idx = np.intersect1d(idx1,idx2)
        if len(idx) > 0:
            appflux = xray["appfluxes"][idx[0]]
            absflux = xray["absfluxes"][idx[0]]
        else:
            appflux, absflux = np.nan, np.nan
        sigs.append(appflux)
    else:
        sigs.append(sig)
filenames = [x for _,x in sorted(zip(sigs,filenames))]

location = EarthLocation.from_geodetic(-111.5967*u.deg, 31.9583*u.deg,
                                       2096*u.m)

kp = Observer(location=location, name="Kitt Peak",timezone="US/Arizona")
kp = Observer(location=location, name="Mauna Kea",timezone="US/Hawaii")

#observe_time = Time('2018-11-04 1:00:00')
observe_time = Time.now()
observe_time = observe_time + np.linspace(deltat_start, deltat_end, 55)*u.hour
tstart, tend = observe_time[0],observe_time[-1]

global_constraints = [AirmassConstraint(max = 1.5, boolean_constraint = False),
    AtNightConstraint.twilight_civil()]

FixedTargets = []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    sig, ra, dec, period = float(filenameSplit[0]), float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
    ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
  
    if np.abs(period - 1.0) < 0.01:
        continue
 
    if dec_hex[0] == "-":
        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
    else:
        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])

    if objname in observed: continue

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    tar = FixedTarget(coord=coord, name=objname)

    if opts.doCheckObservable:
        table = observability_table(global_constraints, kp, [tar],
                            time_range=[tstart,tend])
        idx = np.where(table["ever observable"])[0]
        if len(idx) == 0:
            continue 

    ps1 = ps1_query(ra, dec, 5/3600.0)
    gmag = ps1["gmag"]
    if opts.doMagnitudeCut:
        if gmag < opts.magnitude:
            continue

    FixedTargets.append(tar)

    time.sleep(1.0)
    try:
        result_table = Simbad.query_region(coord, radius=10.0/3600.0 * u.deg)
        if not result_table is None:
            name = result_table[0]["MAIN_ID"].decode()
        else:
            name = "NA"
    except:
        name = "NA"

    if opts.doXray:
        idx1 = np.where(np.abs(xray["ras"] - ra)<=2e-3)[0]
        idx2 = np.where(np.abs(xray["decs"] - dec)<=2e-3)[0]
        idx = np.intersect1d(idx1,idx2)
        if len(idx) > 0:
            appflux = xray["appfluxes"][idx[0]]
            absflux = xray["absfluxes"][idx[0]]
        else:
            appflux, absflux = np.nan, np.nan

        fid.write('%s %.5f %.5f %.5f %.5f %.5f %.5e %.5e "%s"\n'%(objname, ra, dec, period, sig, gmag, appflux, absflux, name))
    else:
        fid.write('%s %.5f %.5f %.5f %.5f %.5f "%s"\n'%(objname, ra, dec, period, sig, gmag, name))

fid.close()

table = observability_table(global_constraints, kp, FixedTargets,
                            time_range=[tstart,tend])
idx = np.where(table["ever observable"])[0]
print("%d/%d targets observable from %s-%s"%(len(idx),len(table),tstart,tend))
FixedTargets = [FixedTargets[i] for i in idx]

plt.figure(figsize=(30,20))
ax = plot_airmass(FixedTargets, kp, observe_time, brightness_shading=True)
plt.legend(shadow=True, loc=2)
altitude_ticks = np.array([90, 60, 50, 40, 30, 20])
airmass_ticks = 1./np.cos(np.radians(90 - altitude_ticks))
ax2 = ax.twinx()
ax2.invert_yaxis()
ax2.set_yticks(airmass_ticks)
ax2.set_yticklabels(altitude_ticks)
ax2.set_ylim(ax.get_ylim())
ax2.set_ylabel('Altitude [degrees]')
plotName = outfile.replace(".dat",".png").replace(".txt",".png")
plt.savefig(plotName)
plt.close()

