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
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic, EarthLocation

import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.special import wofz
from astropy.modeling.models import Voigt1D as voigt
from astropy.modeling.models import Gaussian1D as gaussian
from astropy.modeling.models import Lorentz1D as lorentzian

from statsmodels.nonparametric.smoothers_lowess import lowess

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/spectra/40minute/LRIS")
    parser.add_option("-N","--N",type=int,default=8)
    parser.add_option("--cwave",default="3889.0490,3970.0720,4101.7400,4340.4620,4861.3615,6564.5377")
    parser.add_option("--errorbudget",type=float,default=0.05)

    parser.add_option("--doInterp",  action="store_true", default=False)
    parser.add_option("--doMovie",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2]
    den_pts = pts[Npts/2:]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def loadspectra(f):
    data=np.loadtxt(f)
    return(data[startwave:stopwave,1])
    
def loadallspectra():
    filenames = glob.glob(os.path.join(dataDir,'*spec*'))
    exposure_starts = []
    for ii, filename in enumerate(filenames):
        data=np.loadtxt(filename)
        lines = [line.rstrip('\n') for line in open(filename)]
        for line in lines:
            if "MJDMID" in line:
                lineSplit = list(filter(None,line.split(" ")))
                obsdate = float(lineSplit[3])
                exposure_starts.append(obsdate)
        if ii == 0:
            FullData=[]
            wavelengths=data[:,0]
            idx = np.where((wavelengths >= startwave) & (wavelengths <= stopwave))[0]
            wavelengths = wavelengths[idx]
            FullData.append(wavelengths)

        vals = lowess(data[idx,1], wavelengths, frac=0.05)

        FullData.append((10**17)*vals[:,1])

    return FullData, exposure_starts

def interpallspectra(newdat, N):

    wavelengths, spectra = newdat[0], newdat[1:]
    nspectra = len(spectra)
 
    FullData = []
    FullData.append(wavelengths)
    tt = np.linspace(0,1,N)
    for ii in range(N):
        #weights1 = 1./np.abs(phases-tt[ii])
        #weights2 = 1./np.abs(phases-(1-tt[ii]))
        #weights = np.max(np.vstack((weights1,weights2)),axis=0)
        weights = 1./np.abs(phases-tt[ii])

        if np.any(~np.isfinite(weights)):
            idx = np.where(~np.isfinite(weights))[0]
            weights[:] = 0.0
            weights[idx] = 1.0

        weights = weights / np.sum(weights)
        weights[weights<0.1] = 0.0
        weights = weights / np.sum(weights)

        newspec = np.zeros(wavelengths.shape)
        for spec, weight in zip(spectra, weights):
            newspec = newspec + spec*weight

        FullData.append(newspec)
      
    return FullData

def func_trend(x,p0,p1,p2,p3):
        #print(p0)
        #print(p1*x)
        #print(p2*(x**2))
        #print(p3*(x**3))
        return p0+(p1*x)+(p2*(x**2))+(p3*(x**3))

def func(x,p0,p1,p2,vel0,vel1,v1,v2,g1,g2):
        return p0+p1*x+p2*x**2+gaussian(v1,(1+(vel0)/300000)*(cwave),v2)(x)+gaussian(g1,(1+vel1/300000)*(cwave),g2)(x)
        #return p0+p1*x+p2*x**2+lorentzian(v1,(1+(vel0)/300000)*(4340.462),v2)(x)+gaussian(g1,(1+vel1/300000)*(4340.462),g2)(x)

def myprior_trend(cube, ndim, nparams):

        cube[0] = cube[0]*10000
        cube[1] = cube[1]*10.0 - 5.0
        cube[2] = cube[2]*1e-3 - 5e-4
        cube[3] = cube[3]*8e-10 - 4e-10


def myprior_global_fit(cube, ndim, nparams):

        cube[0] = cube[0]*1000.0
        #cube[2] = cube[2]*0.02 + np.pi/2.0 - 0.01
        #cube[2] = cube[2]*0.1
        cube[1] = cube[1]*np.pi
        cube[2] = cube[2]*2000.0 - 1000.0
        #cube[4] = cube[4]*2000.0 - 1000.0

def myprior_fit(cube, ndim, nparams):

        cube[0] = cube[0]*1000.0
        cube[1] = cube[2]*np.pi
        cube[2] = cube[3]*2000.0 - 1000.0
        #cube[4] = cube[4]*2000.0 - 1000.0

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*500 - 250.0
        cube[1] = cube[1]*2.0 - 1.0
        #cube[1] = cube[1]*1e-10
        cube[2] = cube[2]*2e-4 - 1e-4
        #cube[2] = cube[2]*1e-10
        cube[3] = cube[3]*1000.0 - 500.0
        cube[4] = cube[4]*2000.0 - 1000.0
        cube[5] = cube[5]*500.0 - 500.0
        #cube[5] = cube[5]*1.0 - 1.0
        #cube[6] = cube[6]*50.0
        cube[6] = cube[6] + 28.5
        cube[7] = cube[7]*500.0 - 500.0
        #cube[7] = cube[7]*1.0 - 1.0
        #cube[8] = cube[8]*20.0
        cube[8] = cube[8] + 4.5

def myloglike_fit(cube, ndim, nparams):

    A = cube[0]
    phi = cube[1]
    c = cube[2]
    #d = cube[4]

    #if c<d:
    #    return -np.inf

    prob = 0
    for key in keys:
        vel0 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi))+c
        vel1 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi+np.pi))+c

        kdedir = data_out[key]["kdedir"]
        vals = np.array([vel0,vel1]).T
        kdeeval = kde_eval(kdedir,vals)[0]
        prob = prob + np.log(kdeeval)

        if not np.isfinite(prob):
            break

    if np.isnan(prob):
        prob = -np.inf

    #if np.isfinite(prob):
    #    print(A,B,phi,c,d,prob)

    return prob

def myloglike_global_fit(cube, ndim, nparams):

    A = cube[0]
    phi = cube[1]
    c = cube[2]
    #d = cube[4]

    #if c<d:
    #    return -np.inf

    prob = 0
    for key in keys:
        keySplit = key.split("_")
        cwave = float(keySplit[0])
        wavelengths = data_out[key]["wavelengths"]
        spec = data_out[key]["spec"]

        vel0 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi))+c
        vel1 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi+np.pi))+c

        (p0_best, p1_best, p2_best, vel0_best, vel1_best, v1_best, v2_best, g1_best, g2_best) = data_out[key]["params"]

        model = func(wavelengths,p0_best,p1_best,p2_best,vel0,vel1,v1_best,v2_best,g1_best,g2_best)
        sigma = spec * errorbudget

        x = model - spec

        thisprob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
        thisprob = np.sum(thisprob)

        prob = prob + thisprob

        if not np.isfinite(prob):
            break

    if np.isnan(prob):
        prob = -np.inf

    #if np.isfinite(prob):
    #    print(A,phi,c,prob)

    return prob


def myloglike_trend(cube, ndim, nparams):
    p0 = cube[0]
    p1 = cube[1]
    p2 = cube[2]
    p3 = cube[3]

    model = func_trend(wavelengths,p0,p1,p2,p3)
    sigma = spec * errorbudget

    x = model - spec

    idx = np.where((wavelengths < 4300.0) | (wavelengths > 4400.0))[0]
    prob = ss.norm.logpdf(x[idx], loc=0.0, scale=sigma[idx])
    prob = np.sum(prob)

    return prob

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

    if v2<g2:
        return -np.inf

    #if (np.abs(v1) > 2*np.abs(g1)):
    #    return -np.inf

    #v2_mu, v2_std = 20.0, 3.0
    #v2 = ss.norm(v2_mu, v2_std).ppf(v2)

    model = func(wavelengths,p0,p1,p2,vel0,vel1,v1,v2,g1,g2)
    sigma = spec * errorbudget

    x = model - spec
    prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
    prob = np.sum(prob)

    return prob

# Parse command line
opts = parse_commandline()

FullData=[]

N=opts.N
cwaves=np.array(opts.cwave.split(","),dtype=float)
dataDir = opts.dataDir
errorbudget = opts.errorbudget
basebaseplotDir = os.path.join(opts.plotDir,'LRIS_40_%.2f'%errorbudget)

if not os.path.isdir(basebaseplotDir):
    os.makedirs(basebaseplotDir)

n=1
final=[]

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

RA, Dec = 285.3559192, 53.1581982
c = SkyCoord(RA,Dec, unit="deg")
d=c.transform_to(BarycentricTrueEcliptic)
Palomar=EarthLocation.of_site('Palomar')

#spectra = [spectra[3]]
# T0 is in BJD
T0 = 0.586014316528247946E+05

p=(2*0.01409783493869)

phases = []
data_out = {}
for cwave in cwaves:
    baseplotDir = os.path.join(basebaseplotDir,'%.4f' % cwave)
    pcklFile = "%s/data.pkl"%(baseplotDir)
    f = open(pcklFile, 'r')
    (data_out_tmp, spectra_primary, spectra_diff) = pickle.load(f)
    f.close()

    keys = sorted(data_out_tmp.keys())
    for key_tmp in keys:
        phases.append(data_out_tmp[key_tmp]["phase"])

        key = "%.4f_%s"%(cwave,key_tmp)
        data_out[key] = data_out_tmp[key_tmp]

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotDir = os.path.join(basebaseplotDir,'simul')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

parameters = ["A","phi","c"]
labels = ["A",r"$\phi$","c"]
n_params = len(parameters)

pymultinest.run(myloglike_global_fit, myprior_global_fit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)
#data[:,2] = np.random.rand(len(data[:,2]))
A,phi,c,loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
A_best, phi_best, c_best = data[idx,0:-1]

plotName = "%s/corner_global.pdf"%(basebaseplotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

print(stop)

for ii, key in enumerate(keys):
    keySplit = key.split("_")
    cwave = float(keySplit[0])
    wavelengths = data_out[key]["wavelengths"]

    vel0 = A_best*np.sin(2*np.pi*(data_out[key]["phase"]+phi_best))+c_best
    vel1 = A_best*np.sin(2*np.pi*(data_out[key]["phase"]+phi_best+np.pi))+c_best

    (p0_best, p1_best, p2_best, vel0_best, vel1_best, v1_best, v2_best, g1_best, g2_best) = data_out[key]["params"]

    model = func(wavelengths,p0_best,p1_best,p2_best,vel0,vel1,v1_best,v2_best,g1_best,g2_best)
    model1 = func(wavelengths,p0_best,p1_best,p2_best,vel0,vel1,v1_best,v2_best,0,g2_best)
    model2 = func(wavelengths,p0_best,p1_best,p2_best,vel0,vel1,0,v2_best,g1_best,g2_best)

    data_out[key]["model"] = model
    data_out[key]["model1"] = model1
    data_out[key]["model2"] = model2

    vel0 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi))+c
    vel1 = A*np.sin(2*np.pi*(data_out[key]["phase"]+phi+np.pi))+c

    data_out[key]["vel0"] = vel0
    data_out[key]["vel1"] = vel1
    pts = np.vstack((vel0,vel1)).T
    data_out[key]["kdedir"] = greedy_kde_areas_2d(pts)

if opts.doMovie:
    moviedir = os.path.join(basebaseplotDir,'movie')
    if not os.path.isdir(moviedir):
        os.makedirs(moviedir)

keys = sorted(data_out_tmp.keys())
idx = np.argsort(np.array(keys, dtype=int))
keys = [keys[ii] for ii in idx]

ii = 0
for key_tmp in keys:
    fig = plt.figure(figsize=(10, 16))
    gs = gridspec.GridSpec(int(np.ceil(len(cwaves)/2.0)), 2)
    for jj, cwave in enumerate(cwaves):

        key = "%.4f_%s"%(cwave,key_tmp)
        idx1,idx2 = np.divmod(jj,2)
        idx1,idx2 = idx1.astype(int),idx2.astype(int)
         
        ax = fig.add_subplot(gs[int(idx1), int(idx2)])
        plt.axes(ax)          
        xmin, xmax = np.min(data_out[key]["wavelengths"]), np.max(data_out[key]["wavelengths"])
        ymin, ymax = np.min(data_out[key]["model"]), np.max(data_out[key]["model2"])

        plt.plot(data_out[key]["wavelengths"],data_out[key]["spec"],'k--',linewidth=2)  
        plt.plot(data_out[key]["wavelengths"],data_out[key]["model"],'b-',linewidth=2)
        plt.plot(data_out[key]["wavelengths"],data_out[key]["model1"],'g-',linewidth=2)
        plt.plot(data_out[key]["wavelengths"],data_out[key]["model2"],'c-',linewidth=2)
        vel0 = np.median(data_out[key]["vel0"])
        vel1 = np.median(data_out[key]["vel1"])

        lambda1 = cwave * (1+(vel0)/300000)
        lambda2 = cwave * (1+(vel1)/300000)

        plt.plot([lambda1,lambda1],[ymin,ymax],'g-')
        plt.plot([lambda2,lambda2],[ymin,ymax],'c-')
        plt.grid()
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])

    plt.suptitle('Phase: %.3f' % data_out[key]["phase"], fontsize=28)
    plt.savefig(plotName)
    plotName = "%s/spec.png"%(plotDir)
    plt.savefig(plotName)
    plt.close()

    if opts.doMovie:
        plotNameMovie = "%s/movie-%04d.png"%(moviedir,ii)
        cp_command = "cp %s %s" % (plotName, plotNameMovie)
        os.system(cp_command)
    ii = ii + 1

if opts.doMovie:
    moviefiles = os.path.join(moviedir,"movie-%04d.png")
    filename = os.path.join(moviedir,"spectra.mpg")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)
    filename = os.path.join(moviedir,"spectra.gif")
    ffmpeg_command = 'ffmpeg -an -y -r 20 -i %s -b:v %s %s'%(moviefiles,'5000k',filename)
    os.system(ffmpeg_command)

    rm_command = "rm %s/*.png"%(moviedir)
    os.system(rm_command)

print(stop)

colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotDir = os.path.join(basebaseplotDir,'combined')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

n_live_points = 500
evidence_tolerance = 0.5

parameters = ["A","phi","c"]
labels = ["A",r"$\phi$","c"]
n_params = len(parameters)

pymultinest.run(myloglike_fit, myprior_fit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)
#data[:,2] = np.random.rand(len(data[:,2]))
A,B,phi,c,loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
idx = np.argmax(loglikelihood)
A_best, B_best, phi_best, c_best = data[idx,0:-1]

color2 = 'coral'
color1 = 'cornflowerblue'
color3 = 'palegreen'
color4 = 'darkmagenta'

xs, vel0s, vel1s = [], np.zeros((len(keys),len(A))), np.zeros((len(keys),len(A))) 
for jj,key in enumerate(keys):
    x = data_out[key]["phase"]
    for ii in range(len(A)):
        vel0s[jj,ii] = A[ii]*np.sin(2*np.pi*(data_out[key]["phase"]+phi[ii]))+c[ii] 
        vel1s[jj,ii] = B[ii]*np.sin(2*np.pi*(data_out[key]["phase"]+phi[ii]+np.pi))+c[ii]
    xs.append(x)

idx = np.argsort(xs)
xs = np.array(xs)
xs, vel0s, vel1s = xs[idx], vel0s[idx,:], vel1s[idx,:]

vel0s_10 = np.percentile(vel0s,10,axis=1)
vel0s_50 = np.percentile(vel0s,50,axis=1)
vel0s_90 = np.percentile(vel0s,90,axis=1)

vel1s_10 = np.percentile(vel1s,10,axis=1)
vel1s_50 = np.percentile(vel1s,50,axis=1)
vel1s_90 = np.percentile(vel1s,90,axis=1)

plotName = "%s/violin.pdf"%(plotDir)
fig = plt.figure(figsize=(40,28))
for key, color in zip(keys,colors):
    parts = plt.violinplot(data_out[key]["vel0"],[data_out[key]["phase"]],widths=0.1)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1) 
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)
    parts = plt.violinplot(data_out[key]["vel1"],[data_out[key]["phase"]],widths=0.1)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color2)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color2)
        pc.set_edgecolor(color2)

plt.plot(xs,vel0s_50,'x--',color=color1,linewidth=5)
plt.plot(xs,vel1s_50,'o--',color=color2,linewidth=5)
plt.fill_between(xs,vel0s_10,vel0s_90,facecolor=color1,edgecolor=color1,alpha=0.2,linewidth=3)
plt.fill_between(xs,vel1s_10,vel1s_90,facecolor=color2,edgecolor=color2,alpha=0.2,linewidth=3)

plt.xlabel('Phase Bin',fontsize=28)
plt.ylabel('Velocity [km/s]',fontsize=28)
plt.grid()
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.savefig(plotName)
plt.close()

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

