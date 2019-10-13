
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

    parser.add_option("-c","--condor_file",default="/home/mcoughlin/ZTF/output_lamost_multiepoch_snr/condor/condor.dag.rescue005")
    parser.add_option("-i","--input_file",default="/home/mcoughlin/ZTF/output_lamost_multiepoch_snr/condor/condor.sh")
    parser.add_option("-o","--output_file",default="/home/mcoughlin/ZTF/output_lamost_multiepoch_snr/condor/condor_slice.sh")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

input_file = opts.input_file
output_file = opts.output_file
condor_file = opts.condor_file

lines = [line.rstrip('\n') for line in open(input_file)]
njobs = len(lines)

jobs = np.arange(njobs)
jobs_done = []

lines_new = [line.rstrip('\n') for line in open(condor_file)]
for line in lines_new:
    lineSplit = line.split(" ")
    if ("DONE" in line) and (len(lineSplit) == 2):
        jobs_done.append(int(lineSplit[1]))
jobs = np.setdiff1d(jobs,jobs_done)

fid = open(output_file, 'w')
for ii, job in enumerate(jobs):
    fid.write('%s\n' % lines[job])
fid.close()

