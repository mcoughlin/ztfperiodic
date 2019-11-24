
import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("--hostname",default="mcoughlin@schoty.caltech.edu")
    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")
    parser.add_option("--matchfileDir",default="/home/mcoughlin/ZTF/matchfiles/")
    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

matchfileDir = opts.matchfileDir
outputDir = opts.outputDir

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

outfile = os.path.join(outputDir,'matchfiles.txt')
fid = open(outfile,'w') 
directory="%s/*/*/*.pytable"%opts.matchfileDir
for f in glob.iglob(directory):
    fid.write('%s:%s\n' % (opts.hostname,f))
fid.close()

