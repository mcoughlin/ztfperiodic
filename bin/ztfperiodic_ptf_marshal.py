
import os, sys
import re
import requests
from bs4 import BeautifulSoup

from penquins import Kowalski
import numpy as np
import pandas as pd

catalog_file = "../input/ptf_catalog.dat"
lines = [line.rstrip('\n') for line in open(catalog_file)]

outDir = "/home/michael.coughlin/ZTF/ztfperiodic-papers/data/spectra/PTF"
filename = "/home/michael.coughlin/ZTF/ztfperiodic-papers/papers/PTF/obj.dat"

fid = open(filename, 'w')

for line in lines:
    lineSplit = list(filter(None,line.split("\t")))
    name, ra, dec = lineSplit[0], lineSplit[1], lineSplit[2]
    namesub = lineSplit[0].replace("PTFS","")

    wget_command = "wget --load-cookies=cookies.txt --output-document=spectra.dat 'http://skipper.caltech.edu:8082/marshals/variable/source.php?name=%s&page=spectra'" % namesub
    os.system(wget_command)

    spec_file = "spectra.dat"
    filelines = [line.rstrip('\n') for line in open(spec_file)]
    for fileline in filelines:
        if ("Download" in fileline):
            lineSplit = fileline.split('FITS')[1].split('ASCII')[0].split('"')
            spec_html = lineSplit[1]

            outfile = os.path.join(outDir, '%s.dat' % (name))
            wget_command = "wget --load-cookies=cookies.txt --output-document=%s '%s'" % (outfile, spec_html) 
            os.system(wget_command)
        
    wget_command = "wget --load-cookies=cookies.txt --output-document=obj.dat 'http://skipper.caltech.edu:8082/marshals/variable/source.php?name=%s'" % namesub 
    os.system(wget_command)

    obj_file = "obj.dat"
    filelines = [line.rstrip('\n') for line in open(obj_file)]
    for fileline in filelines:
        if ("Classification" in fileline) and ("notes" in fileline):
            lineSplit = fileline.split(',')
            for spl in lineSplit:
                if "description" in spl:
                    splSplit = spl.split(":")
                    classification = splSplit[-1].replace('"','').replace(" ","_")
                    break

    fid.write('%s %s %s %s\n' % (name, ra, dec, classification))

fid.close()

