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
import scipy.signal

from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, EarthLocation
import astropy.units as u

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec

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
    parser.add_option("-t","--timestampDir",default="../../data/timestamps")
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
                q=pars[6],
                period=p,
                shape_1='sphere',
                shape_2='roche',
                ldc_1=0.0,
                ldc_2=0.0,
                gdc_2=0.4,
                f_c=0,
                f_s=0,
                t_exp=t_exp,
                grid_1=grid,
                grid_2=grid, 
                heat_2 = 1.0,
                exact_grav=True,
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
        cube[5] = cube[5]*2.0 - 1.0
        cube[6] = cube[6]*5.0

def myloglike(cube, ndim, nparams):

    r1 = cube[0]
    r2 = cube[1]
    J = cube[2]
    i = cube[3]
    t0 = cube[4]
    scale = cube[5]
    q = cube[6]

    #model_pars = [r1,r2,J,i,t0,scale,heat_2,q,ldc_1,ldc_2,gdc_2]
    model_pars = [r1,r2,J,i,t0,scale,q]
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
basetimestampDir = opts.timestampDir
baseplotDir = os.path.join(opts.plotDir,'Lightcurve_40')
baseplotDir = os.path.join(baseplotDir,"%.5f"%opts.errorbudget)

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

timestampDir = os.path.join(basetimestampDir,'40min-Chimera-g-20190808')
timestampfiles = glob.glob(os.path.join(timestampDir, '*.txt'))
times = []
for timestampfile in timestampfiles:
    lines = [line.rstrip('\n') for line in open(timestampfile)]
    if len(lines) == 101:
        lines = lines[:-1]
    elif len(lines) == 152:
        lines = lines[:-1]
        lines.pop(100)
    elif len(lines) == 210:
        lines = lines[-100:]
        
    for line in lines:
        lineSplit = list(filter(None,line.split(" ")))
        time = Time('%s %s' % (lineSplit[-3], lineSplit[-2]))
        dt = TimeDelta(2.9999990463256836/2.0 * u.s)
        time = time - dt
        times.append(time.mjd)
times = np.sort(times)

lightcurveFile = os.path.join(dataDir,'40min-Chimera-g-20190808.dat')
data=np.loadtxt(lightcurveFile,skiprows=1,delimiter=' ')
data[:,0] = data[:,0] - 1.0/24.0 # daylight savings...

idx1 = 100*12 + 95
idx2 = 100*14
data = np.vstack((data[:idx1,:],data[idx2:,:]))

times = times[100:]
data = data[100:,:]
times = times[:2300]
data = data[:2300]

t0 = np.min(times)
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)
plt.plot(np.arange(len(data[:,0])),(data[:,0]-t0)*86400.0,'kx',alpha=0.2)
plt.plot(np.arange(len(times)),(times-t0)*86400.0,'bo',alpha=0.2,zorder=-10)
plt.ylabel('Time [s]')
plt.setp(ax1.get_xticklabels(), visible=False)

plt.axes(ax2)
plt.plot(np.arange(len(data[:,0])),(data[:,0]-times)*86400.0,'bo',alpha=0.2,zorder=-10)
plt.ylabel('Cube - GPS')
plt.xlabel('Image Number')

plt.show()
plotName = os.path.join(baseplotDir,'timing.pdf')
plt.savefig(plotName)
plt.close()

fig = plt.figure(figsize=(8, 12))
plt.plot(np.arange(len(data[:,0])-1),np.diff(data[:,0]),'kx',alpha=0.2)
plt.plot(np.arange(len(times)-1),np.diff(times),'bo',alpha=0.2,zorder=-10)
plt.ylim([0,0.0002])
plt.show()
plotName = os.path.join(baseplotDir,'timing_diff.pdf')
plt.savefig(plotName)
plt.close()

data[:,4] = np.abs(data[:,4])
#y, dy=Detrending.detrending(data)

y=data[:,3]/np.max(data[:,3])
y=(y-np.median(y)) + 1.0

dy=np.sqrt(data[:,4]**2 + opts.errorbudget**2)/np.max(data[:,3])
t=data[:,0]

RA, Dec = 285.3559192, 53.1581982
t = BJDConvert(t, RA, Dec)
t = np.array([x.value for x in t])
t_exp=t[1]-t[0]
fs = 1/(t_exp*86400)

r1 = 0.2
r2 = 0.125
J = 0.25
i = 90
idx = np.argmin(y)
t0 = t[idx]
p = 0.02823
p = 2*0.01409783493869
scale = np.median(y)/1.0
#scale = 1.0
heat_2 = 0.0
q=1.0
ldc_1=0.0
ldc_2=0.0
gdc_2=0.0
f_c=0
f_s=0

tmin, tmax = np.min(t), np.max(t)
tmin, tmax = np.min(t), np.min(t)+p

# generate the test light curve given parameters

model_pars = [r1,r2,J,i,t0,scale,q] # the parameters

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
plotName = os.path.join(baseplotDir,'lightcurve.pdf')
plt.savefig(plotName)
plt.close()

f, Pxx_den = scipy.signal.periodogram(lc[:,1], fs)
period = 1/f #s
period = period / 60.0 # minutes
plt.loglog(period, Pxx_den)
plt.plot([1/p, 1/p], [1e-6, 1e0], 'k--')
#plt.ylim([1e-7, 1e2])
plt.xlabel('Time [minutes]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
plotName = os.path.join(baseplotDir,'periodogram.pdf')
plt.savefig(plotName)
plt.close()

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["r1","r2","J","i","t0","scale","heat_2","q","ldc_1","ldc_2","gdc_2"]
labels = [r"$r_1$",r"$r_2$","J","i",r"$t_0$","scale",r"${\rm heat}_2$","q","$ldc_1$","$ldc_2$","$gdc_2$"]

parameters = ["r1","r2","J","i","t0","scale","q"]
labels = [r"$r_1$",r"$r_2$","J","i",r"$t_0$","scale","q"]

n_params = len(parameters)

plotDir = os.path.join(baseplotDir,'posteriors')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

#r1, r2, J, i, t0,scale, heat_2, q, ldc_1, ldc_2, gdc_2, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9], data[:,10], data[:,11] 
r1, r2, J, i, t0,scale, q, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
idx = np.argmax(loglikelihood)
#r1_best, r2_best, J_best, i_best, t0_best, scale_best, heat_2_best, q_best, ldc_1_best, ldc_2_best, gdc_2_best = data[idx,0:-1]
r1_best, r2_best, J_best, i_best, t0_best, scale_best, q_best = data[idx,0:-1]

labels = labels[:4]
data = data[:,:-4]

plotName = "%s/corner.pdf"%(baseplotDir)
figure = corner.corner(data, labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

#model_pars = [r1_best, r2_best, J_best, i_best, t0_best, scale_best, heat_2_best, q_best, ldc_1_best, ldc_2_best, gdc_2_best] # the parameters

model_pars = [r1_best, r2_best, J_best, i_best, t0_best, scale_best, q_best] # the parameters

t0 = lc[:,0][0]
fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)
plt.errorbar(lc[:,0]-t0,lc[:,1],lc[:,2],fmt='k.')
plt.ylabel('Flux')
# my initial guess (r1,r2,J,i,t0,p,scale)
guess = model_pars
plt.plot(t[:]-t0,basic_model(t[:],model_pars),zorder=4)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.axes(ax2)
plt.errorbar(t[:]-t0,lc[:,1]-basic_model(t[:],model_pars),lc[:,2],fmt='k.')
plt.ylabel('Model - Data')
plt.xlabel('Time [HJD$_0$ = %.5f]' % t0)
plt.show()
plotName = os.path.join(baseplotDir,'fit.pdf')
plt.savefig(plotName)
plt.close()

