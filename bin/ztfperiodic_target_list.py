
import os, sys, glob, time
import string
import random
import optparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.colors as colors
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

import ztfperiodic
from ztfperiodic.utils import gaia_query
from ztfperiodic.utils import ps1_query

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Desktop/Kevin/Candidates/obj.dat")
    parser.add_option("-i","--inputDir",default="/Users/mcoughlin/Desktop/Kevin/Candidates/")
    parser.add_option("-z","--ztfperiodicInputDir",default="../input")

    parser.add_option("--doGaia",  action="store_true", default=False)

    parser.add_option("--doXray",  action="store_true", default=False)
    parser.add_option("--doCheckObservable",  action="store_true", default=False)
    parser.add_option("--doMagnitudeCut",  action="store_true", default=False)
    parser.add_option("--doKPED",  action="store_true", default=False)
   
    parser.add_option("-c","--significance_cut",default=0.0,type=float)
    parser.add_option("-m","--magnitude",default=16.0,type=float)
    parser.add_option("-s","--deltat_start",default=0.0,type=float)
    parser.add_option("-e","--deltat_end",default=24.0,type=float)

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

def mass_luminosity(L):
    M = L**(1/3.5)
    return M

def random_char(y):
    return ''.join(random.choice(string.ascii_letters) for x in range(y))

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

filenames = get_filepaths(inputDir)
filenames = [x for x in filenames if "png" in x]

sigs = []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    if filenameSplit[0] == "obj":
        continue

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
#kp = Observer(location=location, name="Mauna Kea",timezone="US/Hawaii")

#observe_time = Time('2018-11-04 1:00:00')
observe_time = Time.now()
observe_time = observe_time + np.linspace(deltat_start, deltat_end, 55)*u.hour
tstart, tend = observe_time[0],observe_time[-1]

global_constraints = [AirmassConstraint(max = 1.5, boolean_constraint = False),
    AtNightConstraint.twilight_civil()]

cols, mags, ecols, emags, periods = [], [], [], [], []

requestID = random_char(3).upper()

tottime = 3600.0
filt = 'FILTER_SLOAN_G'
mode = 9

fid = open(outfile,'w')
cnt = 0
FixedTargets = []
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    if filenameSplit[0] == "obj": 
        continue

    sig, ra, dec, period = float(filenameSplit[0]), float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
    ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=':'), convert_to_hex(dec,delimiter=':')
    ra_hex_nodelim, dec_hex_nodelim = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='') 

    if sig < opts.significance_cut: continue
 
    if np.abs(period - 1.0) < 0.01:
        continue
 
    if dec_hex[0] == "-":
        objname = "ZTFJ%s%s"%(ra_hex_nodelim[:4],dec_hex_nodelim[:5])
    else:
        objname = "ZTFJ%s%s"%(ra_hex_nodelim[:4],dec_hex_nodelim[:4])

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
    if not ps1:
        gmag, rmag = np.nan, np.nan
    else:
        gmag, rmag = ps1["gmag"], ps1["rmag"]
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

    col, mag, P_min = np.nan, np.nan, np.nan
    if opts.doGaia:
        gaia = gaia_query(ra, dec, 5/3600.0)
        if gaia:
            col = gaia['BP-RP'][0]
            ecol = np.sqrt(gaia['e_BPmag'][0]**2 + gaia['e_RPmag'][0]**2)
            mag = gaia['Gmag'][0] + 5*(np.log10(gaia['Plx'][0]) - 2)
            emag = np.sqrt(gaia['e_Gmag'][0]**2 + (5*gaia['e_Plx'][0]/(gaia['Plx'][0]*np.log(10)))**2)

            L, rad = gaia['Lum'][0], gaia['Rad'][0]
            # mass-luminosity
            M = L**(1/3.5)
            M = M * 2*1e33 # grams

            V = (4/3)*np.pi*(rad * 7.0*1e10)**3.0 # cm^3
            density = M / V

            P_min = 12.6/24.0 * (density)**(-1/2.0)

            cols.append(col)
            mags.append(mag)
            periods.append(period)
            ecols.append(ecol)
            emags.append(emag)

    if opts.doXray:
        idx1 = np.where(np.abs(xray["ras"] - ra)<=2e-3)[0]
        idx2 = np.where(np.abs(xray["decs"] - dec)<=2e-3)[0]
        idx = np.intersect1d(idx1,idx2)
        if len(idx) > 0:
            appflux = xray["appfluxes"][idx[0]]
            absflux = xray["absfluxes"][idx[0]]
        else:
            appflux, absflux = np.nan, np.nan

        if opts.doKPED:
            print('%s%04d,1,%s,%s,%s,2000.0,0.00,0.00,%.2f,%.0f,%s,%d,Coughlin,comment'%(requestID, cnt, objname, ra_hex, dec_hex, gmag, tottime, filt, mode),file=fid,flush=True)
        else:
            print('%s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5e %.5e "%s"'%(objname, ra, dec, period, sig, gmag, col, mag, P_min, appflux, absflux, name),file=fid,flush=True)
    else:
        if opts.doKPED:
            print('%s%04d,1,%s,%s,%s,2000.0,0.00,0.00,%.2f,%.0f,%s,%d,Coughlin,comment'%(requestID, cnt, objname, ra_hex, dec_hex, gmag, tottime, filt, mode),file=fid,flush=True)
        else:
            print('%s %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f "%s"'%(objname, ra, dec, period, sig, gmag, col, mag, P_min, name),file=fid,flush=True)
    cnt = cnt + 1

fid.close()

if opts.doGaia:
    plt.figure(figsize=(30,20))
    plt.errorbar(cols, mags, xerr=ecols, yerr=emags, marker='p', markerfacecolor='w', markeredgecolor='w', fmt='ko', markersize=0)
    sc = plt.scatter(cols, mags, s=200, c=periods, alpha=1.0,
                     norm=colors.LogNorm(vmin=np.min(periods),
                                         vmax=np.max(periods)),
                     cmap='PuBu_r')
    cbar = plt.colorbar(sc, extend='max')
    cbar.set_label('Period [days]')
    plt.xlabel('Gaia BP-RP color')
    plt.ylabel('Gaia G absolute magnitude')
    plt.ylim([-5,10])
    plt.gca().invert_yaxis()
    plotName = outfile.replace(".dat","_hr.png").replace(".txt","_hr.png")
    plt.savefig(plotName)
    plt.close()

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

