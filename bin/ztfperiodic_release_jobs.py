#!/usr/bin/python

import os
import sys
import time
import glob
import optparse
import random
import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u

def docondor(opts):

    condor_command = "condor_q -dag -nobatch %s"%opts.user
    filename = "tmp.dat"
    os.system("rm %s"%(filename))
    os.system("%s > %s"%(condor_command,filename))
    
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        lineSplit = list(filter(None,line.split(" ")))
        if len(lineSplit) == 0: continue
        if lineSplit[0] in ["--","ID","Total"]: continue
        if lineSplit[1] == opts.user: continue
        run_time = lineSplit[4]
    
        if opts.doTimeLimit:
            days, total_time = run_time.split("+")
            run_time = TimeDelta(int(days)*u.day) + Time('2020-01-01 %s' % total_time, format='iso', scale='utc') - Time('2020-01-01 00:00:00', format='iso', scale='utc')
            if run_time > TimeDelta(opts.maxtime * u.hour): 
                condor_command = "condor_rm %d"%(jobid)
                os.system(condor_command)   
    
        if lineSplit[5] == opts.jobtype:
            jobid = float(lineSplit[0])
            if opts.doMemory:
                condor_command = "condor_qedit %d RequestMemory %d"%(jobid,opts.memory)
                os.system(condor_command)
            if opts.doRemove:
                condor_command = "condor_rm %d"%(jobid)
                os.system(condor_command)
            if opts.doUniverse:
                print("Assigning %d to JobUniverse %d"%(jobid,opts.universe))
                condor_command = "condor_qedit %d JobUniverse %d"%(jobid,opts.universe)
                os.system(condor_command)
    
    if opts.doMemory or opts.doUniverse:
        condor_command = "condor_release %s"%opts.user
        os.system(condor_command)

parser = optparse.OptionParser(usage=__doc__)
parser.add_option("-u", "--user", default="michael.coughlin")
parser.add_option("-m", "--memory", type="int", default=16384)
parser.add_option("-j", "--jobtype", default="H")
parser.add_option("--universe", type="int", default=12)
parser.add_option("--doTimeLimit",  action="store_true", default=False)
parser.add_option("-t", "--maxtime", type="float", default=3)
parser.add_option("--doMemory",  action="store_true", default=False)
parser.add_option("--doUniverse",  action="store_true", default=False)
parser.add_option("--doRemove",  action="store_true", default=False)
parser.add_option("--doRunContinuously",  action="store_true", default=False)

opts, args = parser.parse_args()

if opts.doRunContinuously:
    while True:
        docondor(opts)
else:
    docondor(opts)
