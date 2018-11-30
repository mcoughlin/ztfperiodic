#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:24:46 2018

@author: kburdge
"""

import os
import numpy as np
import glob
import optparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import scipy.stats
from scipy.optimize import curve_fit
from scipy.special import wofz
from astropy.modeling.models import Voigt1D as voigt
from astropy.modeling.models import Gaussian1D as gaussian

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data/spectra")
    parser.add_option("-N","--N",type=float,default=12)
    parser.add_option("-s","--start",type=float,default=1750)
    parser.add_option("-e","--end",type=float,default=2250)

    opts, args = parser.parse_args()

    return opts

def loadspectra(f):
    data=np.loadtxt(f)
    return(data[start:stop,1])
    
def loadallspectra(N):
    f='%s/1.txt'%dataDir
    data=np.loadtxt(f)
    FullData=[]
    wavelengths=data[start:stop,0]
    FullData.append(wavelengths)
    i=1
    while i<N+1:
        f='%s/'%dataDir+str(i)+'.txt'
        FullData.append((10**17)*loadspectra(f))
        i=i+1
    return FullData

def func(x,p0,p1,p2,vel0,vel1,v1,v2,g1,g2):
        return p0+p1*x+p2*x**2+gaussian(v1,(1+(vel0)/300000)*(4340.462),v2)(x)+gaussian(g1,(1+vel1/300000)*(4340.462),g2)(x)

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*110.0 - 10.0
        cube[1] = cube[1]*110.0 - 10.0
        cube[2] = cube[2]*110.0 - 10.0
        cube[3] = cube[3]*1600.0 - 800.0
        cube[4] = cube[4]*2400.0 - 1200.0
        cube[5] = cube[5]*40.0 - 40.0
        cube[6] = cube[5]*35.0 + 5.0
        cube[7] = cube[6]*20.0
        cube[8] = cube[7]*10.0 + 5.0

def myloglike(cube, ndim, nparams):
    p0 = cube[0]
    p1 = cube[1]
    p2 = cube[2]
    vel0 = cube[3]
    vel1 = cube[4]
    v1 = cube[5]
    v2 = cube[6]
    g1 = cube[7]
    g2 = cube[8]

    model = func(wavelengths,p0,p1,p2,vel0,vel1,v1,v2,g1,g2)
    sigma = spec * 0.01

    x = model - spec
    prob = scipy.stats.norm.logpdf(x, loc=0.0, scale=sigma)
    prob = np.sum(prob)

    return prob

def fitspectrum(x,y,n):

    def func(x,p0,p1,p2,vel0,vel1,v1,v2,g1,g2):
        return p0+p1*x+p2*x**2+gaussian(v1,(1+(vel0)/300000)*(4340.462),v2)(x)+gaussian(g1,(1+vel1/300000)*(4340.462),g2)(x)
    xdata=x
    ydata=y
    #plt.figure(figsize=(20,20))
    #plt.plot(xdata, ydata, 'b-', label='data')
    popt, pcov = curve_fit(func, xdata, ydata,bounds=([-10,-10,-10,-800,-1200,-40,5,0,5],[100,100,100,800,1200,0,40,20,15]))
    popt
    #plt.plot(xdata, func(xdata, *popt))
    perr = np.sqrt(np.diag(pcov))
    v1=popt[3]
    v1Err=perr[3]
    g1=popt[4]
    g1Err=perr[4]
    print(v1)
    print(v1Err)
    print(g1)
    print(g1Err)
    return([n,v1,v1Err,1/v1Err**2,g1,g1Err,1/g1Err**2])

# Parse command line
opts = parse_commandline()

FullData=[]

N=opts.N
start=opts.start
stop=opts.end
dataDir = opts.dataDir
baseplotDir = opts.plotDir

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

newdat=loadallspectra(N)
n=1
final=[]

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

wavelengths, spectra = newdat[0], newdat[1:]

data_out = {}
for ii,spec in enumerate(spectra):
    parameters = ["p0","p1","p2","vel0","vel1","v1","v2","g1","g2"]
    labels = [r"$p_0$",r"$p_1$",r"$p_2$",r"${\rm vel}_0$",r"${\rm vel}_1$",r"$v_1$",r"$v_2$",r"$g_1$",r"$g_2$"]
    n_params = len(parameters)

    plotDir = os.path.join(baseplotDir,'%d'%ii)
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    multifile = "%s/2-post_equal_weights.dat"%plotDir
    data = np.loadtxt(multifile)

    p0,p1,p2,vel0,vel1,v1,v2,g1,g2,loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9]

    idx = np.argmax(loglikelihood)
    p0_best, p1_best, p2_best, vel0_best, vel1_best, v1_best, v2_best, g1_best, g2_best = data[idx,0:-1]
    model = func(wavelengths,p0_best,p1_best,p2_best,vel0_best,vel1_best,v1_best,v2_best,g1_best,g2_best)

    key = str(ii)
    data_out[key] = {}
    data_out[key]["spec"] = spec
    data_out[key]["model"] = model
    data_out[key]["wavelengths"] = wavelengths

    plotName = "%s/corner.pdf"%(plotDir)
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
    figure.set_size_inches(18.0,18.0)
    plt.savefig(plotName)
    plt.close()

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/spec_panels.pdf"%(baseplotDir)
fig = plt.figure(figsize=(22,28))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    plt.plot(data_out[key]["wavelengths"],data_out[key]["spec"],'k--',linewidth=2)
    plt.plot(data_out[key]["wavelengths"],data_out[key]["model"],'b-',linewidth=2)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    #plt.xlim([start,stop])
    #plt.ylim([0.0,1.0])
    plt.grid()
    plt.yticks(fontsize=36)

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()
