
import os, sys
import numpy as np

filenames = ["../input/3-10.dat","../input/10-50.dat","../input/50-baseline.dat","../input/12_72hours.dat"]
filenames = ["../input/12_72hours.dat","../input/4_12hours.dat"]
filenames = ["../input/pure.dat"]
outputDir = '../output/pure'

for filename in filenames:
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        row = line.split(" ")
        ra, dec, period = float(row[1]), float(row[2]), float(row[3])
        path_out_dir='%s/%.5f_%.5f'%(outputDir,ra,dec)
        filename = os.path.join(path_out_dir,'harmonics.dat')
        if os.path.isfile(filename): continue

        system_call = "python ztfperiodic_matchfiles.py --doPlots --doPhase --user mcoughli@caltech.edu --pwd mchoop --ra %.5f --declination %.5f --phase %.5f --outputDir %s"%(ra, dec, period, outputDir)
        print(system_call)
        os.system(system_call)

