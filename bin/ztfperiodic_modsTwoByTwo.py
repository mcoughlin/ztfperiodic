#! /usr/bin/env python
#
# modsAdd - Add (or average) FITS images
#
# Usage: modsAdd [options] inFile?.fits outFile.fits
#
# Where:
#   inFile?.fits = input FITS files to add together.
#   outFile.fits = output file to create
# 
# Options:
#   -a  = compute the average instead of the sum
#   -f  = force overwrite of the output file ("clobber")
#   -v  = print verbose debugging info during execution
#   -V  = print version information and exit
#
# Computes the sum of two or more images, creating a single output file.
# If -a (--average) is given, it computes the average of the images.
#
# Authors:
#   R. Pogge, OSU Astronomy Dept
#   pogge.1@osu.edu
#
# Module Dependencies:
#   numpy
#   pyfits
#
# Modification History
#   2012 Mar 28 - fixed bug in file counter [rwp/osu]
#   2012 Apr 01 - minor fixes, added -a (--average) option to compute
#                 the average (mean) instead of the sum
#   2014 Aug 24 - using astropy.io.fits instead of unsupported pyfits and
#                 suppressing nuisance FITS header warnings [rwp/osu]
#   2016 May 08 - release version adopting astropy.io fits [rwp/osu]
#
#-----------------------------------------------------------------------------

import string as str
import os 
import sys 
import getopt
import numpy as np 
from astropy.io import fits
import skimage.measure

# Version Number and Date

versNum = '2.0.2'
versDate = '2016-03-08'

# Runtime flags

Verbose=False      # Verbose debugging output on/off (see -v/--verbose)
NoClobber=True     # default: no clobber on output (see -f/--clobber)
doAverage=False    # default: straight sum w/o averaging (see -a/--average)

# Suppress nuisance warning from astropy.io.fits, which can
# be unusually strict about FITS standard, but the gunk it 
# complains about is otherwise harmless (blank keywords)

import warnings
warnings.filterwarnings('ignore',category=UserWarning, append=True)
warnings.filterwarnings('ignore',category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore',category=FutureWarning, append=True)
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore',category=AstropyWarning, append=True)

#----------------------------------------------------------------
#
# printUsage - print a simple usage message
#

def printUsage(): 
  print('\nUsage: modsAdd [options] inFile1.fits [inFile2.fits ...] outFile.fits')
  print('\nWhere:')
  print('  inFile?.fits = input FITS files to sum together.')
  print('  outFile.fits = output file to create')
  print('\nOptions:')
  print('  -a  = compute the average instead of the sum')
  print('  -f  = force overwrite of the output file (aka clobber)')
  print('  -v  = print verbose debugging info during execution')
  print('  -V  = print version information and exit\n')

#----------------------------------------------------------------
#
# Main Program starts here...
#

# Parse the command-line arguments (GNU-style getopt)
  
try:
  opts, files = getopt.gnu_getopt(sys.argv[1:],'afvV',
                                  ['average','clobber','verbose','version',])
except getopt.GetoptError as err:
  print('\n** ERROR: %s' % (err))
  printUsage()
  sys.exit(2)

if len(opts)==0 and len(files)==0:
  printUsage()
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-f','--clobber'):
    NoClobber = False
  elif opt in ('-a','--average'):
    doAverage = True
  elif opt in ('-v','--verbose'):
    Verbose = True
  elif opt in ('-V','--version'):
    print('modsTwoByTwo.py v%s [%s]' % (versNum, versDate))
    sys.exit(0)

numFiles = len(files)

if numFiles < 2:
  printUsage()
  sys.exit(1)

# Make sure we won't clobber the output file unless -f is used

# start!
if Verbose:
  print('\nBinning 2x2 %d images...' % (numFiles-1))

kount=0
for rawFile in files:
    if os.path.isfile(rawFile):
      outFile = rawFile.replace(".fits",".2x2.fits")
      if NoClobber and os.path.isfile(outFile):
        print('\n** ERROR: Operation would overwrite existing FITS file %s' % (outFile))
        print('          Use -f to force output (\'clobber\').')
        print('          modsAdd aborting.\n')
        sys.exit(1)

      if Verbose:
        print('  Reading image %2d of %2d: %s...' % (kount+1,numFiles-1,rawFile))
      fitsFile = fits.open(rawFile,uint=False)
      d0 = fitsFile[0].data
      subsamp = 2
      img = skimage.measure.block_reduce(d0, (subsamp,subsamp), func=np.sum)
      img = img.astype(int)
      fitsFile[0].data = img
      fitsFile[0].header.add_history('modsTwoByTwo v%s %s' % (versNum,versDate))
      fitsFile[0].header["CCDXBIN"] = subsamp
      fitsFile[0].header["CCDYBIN"] = subsamp
      fitsFile.writeto(outFile,output_verify='ignore',overwrite=True)

if Verbose:
  print('modsTwoByTwo Done')

sys.exit(0)
