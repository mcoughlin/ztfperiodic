
import os, sys, glob
import optparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
from matplotlib import pyplot as plt

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Code/KP84/KevinPeriods/true.dat")
    parser.add_option("-i","--inputDir",default="/Users/mcoughlin/Code/KP84/KevinPeriods/KevinPure/,/Users/mcoughlin/Code/KP84/KevinPeriods/NewSearch/ForFollowUp")

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

# Parse command line
opts = parse_commandline()
inputDirs = opts.inputDir.split(",")
outfile = opts.outfile

observed = ["ZTFJ18320856","ZTFJ19385841","ZTFJ17182524","ZTFJ17483237","ZTFJ20062220","ZTFJ19244707","ZTFJ1913-1205","ZTFJ19243104","ZTFJ1913-1105","ZTFJ1905-1910","ZTFJ1843-2041","ZTFJ18461355","ZTFJ1924-1258","ZTFJ18261000"]

fid = open(outfile,'w')
for inputDir in inputDirs:
    filenames = glob.glob(os.path.join(inputDir,"*.png"))
    for filename in filenames:
        filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
        sig, ra, dec, period = float(filenameSplit[0]), float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
        ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
   
        if dec_hex[0] == "-":
            objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
        else:
            objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])

        if objname not in observed: continue

        fid.write('%s %.5f %.5f %.5f %.5f\n'%(objname,ra,dec,period,sig))

fid.close()


