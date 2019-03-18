#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:24:46 2018

@author: kburdge
"""

import os
import copy
import numpy as np
import glob
import optparse
import ellc
import time

from astropy.time import Time
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, EarthLocation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import scipy.stats as ss

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/lightcurves")
    parser.add_option("-e","--errorbudget",default=0.0,type=float)

    opts, args = parser.parse_args()

    return opts

def BJDConvert(mjd, RA, Dec):
	times=mjd
	t = Time(times,format='mjd',scale='utc')
	t2=t.tdb
	c = SkyCoord(RA,Dec, unit="deg")
	d=c.transform_to(BarycentricTrueEcliptic)
	Palomar=EarthLocation.of_site('Palomar')
	delta=t2.light_travel_time(c,kind='barycentric',location=Palomar)
	BJD_TDB=t2+delta

	return BJD_TDB

def basic_model(t,pars,grid='default'):
    """ a function which returns model values at times t for parameters pars

    input:
        t    a 1D array with times
        pars a 1D array with parameter values; r1,r2,J,i,t0,p

    output:
        m    a 1D array with model values at times t

    """
    try:
        m = ellc.lc(t_obs=t,
                radius_1=pars[0],
                radius_2=pars[1],
                sbratio=pars[2],
                incl=pars[3],
                t_zero=pars[4],
                q=pars[7],
                period=0.004800824101665522,
                shape_1='sphere',
                shape_2='roche',
                ldc_1=0.2,
                ldc_2=0.4548,
                gdc_2=0.61,
                f_c=0,
                f_s=0,
                t_exp=3.0/86400,
                grid_1=grid,
                grid_2=grid, heat_2 = pars[6],exact_grav=True,
                verbose=0)
        m *= pars[5]

    except:
        print("Failed with parameters:", pars)
        return t * 10**99

    return m

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*1.0 + 0.0
        cube[1] = cube[1]*1.0 + 0.0
        cube[2] = cube[2]*1.0 + 0.0
        cube[3] = cube[3]*90.0 + 0.0
        cube[4] = cube[4]*(tmax-tmin) + tmin
        cube[5] = cube[5]*1.0 + 0.0
        cube[6] = cube[6]*10.0 + 0.0
        cube[7] = cube[7]*1.0 + 0.0

def myloglike(cube, ndim, nparams):

    r1 = cube[0]
    r2 = cube[1]
    J = cube[2]
    i = cube[3]
    t0 = cube[4]
    scale = cube[5]
    heat_2 = cube[6]
    q = cube[7]

    model_pars = [r1,r2,J,i,t0,scale,heat_2,q]
    model = basic_model(t[:],model_pars)
 
    x = model - y
    prob = ss.norm.logpdf(x, loc=0.0, scale=dy)
    prob = np.sum(prob)

    if np.isnan(prob):
        prob = -np.inf

    print(prob)
    return prob

# Parse command line
opts = parse_commandline()

dataDir = opts.dataDir
baseplotDir = os.path.join(opts.plotDir,'Lightcurve_40')
baseplotDir = os.path.join(baseplotDir,"%.5f"%opts.errorbudget)

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

lightcurveFile = os.path.join(dataDir,'40minKPED.dat')
data=np.loadtxt(lightcurveFile,skiprows=1,delimiter=' ')
data=data[:510,:]

data[:,4] = np.abs(data[:,4])
#y, dy=Detrending.detrending(data)

y=data[:,3]/np.max(data[:,3])
dy=np.sqrt(data[:,4]**2 + opts.errorbudget**2)/np.max(data[:,3])
t=data[:,0]

RA, Dec = 285.3559192, 53.1581982
t = BJDConvert(t, RA, Dec)
t = np.array([x.value for x in t])

r1 = 0.3
r2 = 0.3
J = 1/15.0
i = 75
t0 = t[0]
p = 0.029166666666666667
scale = np.median(y)/1.3
heat_2 = 5
bfac_2 = 1.30
q=0.4
ldc_1=0.2
ldc_2=0.4548
gdc_2=0.61
f_c=0
f_s=0

tmin, tmax = np.min(t), np.max(t)
tmin, tmax = np.min(t), np.min(t)+p

# generate the test light curve given parameters

model_pars = [r1,r2,J,i,t0,scale,heat_2,q] # the parameters

# and add errors

lc = np.c_[t,y,dy]

# save the test lc to disk
np.savetxt(os.path.join(baseplotDir,'test.lc'),lc)

plt.figure()
# lets have a look:
plt.errorbar(lc[:,0],lc[:,1],lc[:,2],fmt='k.')
plt.ylabel('flux')
plt.xlabel('time')
# my initial guess (r1,r2,J,i,t0,p,scale)
guess = model_pars
plt.plot(t[:],basic_model(t[:],model_pars),zorder=4)
plt.show()
plotName = os.path.join(baseplotDir,'test.png')
plt.savefig(plotName)
plt.close()

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["r1","r2","J","i","t0","scale","heat_2","q"]
labels = [r"$r_1$",r"$r_2$","J","i",r"$t_0$","scale",r"${\rm heat}_2$","q"]
n_params = len(parameters)

plotDir = os.path.join(baseplotDir,'posteriors')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

r1, r2, J, i, t0,scale, heat_2, q, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4] 
idx = np.argmax(loglikelihood)
r1_best, r2_best, J_best, i_best, t0_best, scale_best, heat_2_best, q_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

