
import os, sys, glob
import optparse
import numpy as np

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outfile",default="/Users/mcoughlin/Code/KP84/KevinPeriods/NewSearch/obj.dat")
    parser.add_option("-i","--inputDir",default="/Users/mcoughlin/Code/KP84/KevinPeriods/NewSearch/ForFollowUp/")

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
inputDir = opts.inputDir
outfile = opts.outfile

fid = open(outfile,'w')
filenames = glob.glob(os.path.join(inputDir,"*.png"))
for filename in filenames:
    filenameSplit = filename.split("/")[-1].replace(".png","").split("_")
    ra, dec, period = float(filenameSplit[1]), float(filenameSplit[2]), float(filenameSplit[3])
    ra_hex, dec_hex = convert_to_hex(ra*24/360.0,delimiter=''), convert_to_hex(dec,delimiter='')
   
    if dec_hex[0] == "-":
        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:5])
    else:
        objname = "ZTFJ%s%s"%(ra_hex[:4],dec_hex[:4])
    fid.write('%s %.5f %.5f %.5f\n'%(objname,ra,dec,period))
fid.close()

