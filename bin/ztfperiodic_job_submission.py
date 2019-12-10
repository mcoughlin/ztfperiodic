
import os, sys
import glob
import optparse

import tables
import pandas as pd
import numpy as np
import h5py

import ztfperiodic.utils

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-p","--python",default="python")

    parser.add_option("-o","--outputDir",default="/home/mcoughlin/ZTF/output")
    parser.add_option("-a","--algorithm",default="CE")
    parser.add_option("-l","--max_jobs",default=1000,type=int)

    parser.add_option("--doSubmit",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

dir_path = os.path.dirname(os.path.realpath(__file__))

algorithm = opts.algorithm
outputDir = opts.outputDir
max_jobs = opts.max_jobs

qsubDir = os.path.join(outputDir,'qsub')
if not os.path.isdir(qsubDir):
    os.makedirs(qsubDir)

catalogDir = os.path.join(outputDir,'catalog',algorithm)

if opts.doSubmit:
    while True:
        numjobs = os.system("qstat -r |wc -l").read()
        print(numjobs)
    if numjobs < max_jobs:
        break
    else:
        time.sleep(30)

