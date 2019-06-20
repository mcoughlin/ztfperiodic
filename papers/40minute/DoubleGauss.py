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

import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.special import wofz
from astropy.modeling.models import Voigt1D as voigt
from astropy.modeling.models import Gaussian1D as gaussian
from astropy.modeling.models import Lorentz1D as lorentzian

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/spectra/40minute")
    parser.add_option("-N","--N",type=int,default=100)
    #parser.add_option("-s","--start",type=int,default=0)
    #parser.add_option("-e","--end",type=int,default=1500)
    parser.add_option("-s","--start",type=int,default=750)
    parser.add_option("-e","--end",type=int,default=1050)
    parser.add_option("--errorbudget",type=float,default=0.1)

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
    return(data[start:stop,1])
    
def loadallspectra():
    filenames = glob.glob(os.path.join(dataDir,'*txt'))
    for ii, filename in enumerate(filenames):
        data=np.loadtxt(filename)
        if ii == 0:
            FullData=[]
            wavelengths=data[start:stop,0]
            FullData.append(wavelengths)
        FullData.append((10**17)*data[start:stop,1])
    return FullData

def interpallspectra(newdat, N):

    wavelengths, spectra = newdat[0], newdat[1:]
    nspectra = len(spectra)
 
    FullData = []
    FullData.append(wavelengths)
    tt = np.linspace(0,N,nspectra)
    for ii in range(N):
        weights = 1./np.abs(tt-ii)
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

def func(x,p0,p1,p2,vel0,vel1,v1,v2,g1,g2):
        return p0+p1*x+p2*x**2+gaussian(v1,(1+(vel0)/300000)*(4340.462),v2)(x)+gaussian(g1,(1+vel1/300000)*(4340.462),g2)(x)
        #return p0+p1*x+p2*x**2+lorentzian(v1,(1+(vel0)/300000)*(4340.462),v2)(x)+gaussian(g1,(1+vel1/300000)*(4340.462),g2)(x)

def myprior_fit(cube, ndim, nparams):

        cube[0] = cube[0]*1000.0
        cube[1] = cube[1]*1000.0
        cube[2] = cube[2]*2*np.pi
        #cube[2] = np.pi
        cube[3] = cube[3]*1000.0 - 500.0
        cube[4] = cube[4]*1000.0 - 500.0

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*500 + 50
        cube[1] = cube[1]*2.0 - 1.0
        cube[2] = cube[2]*2e-4 - 1e-4
        cube[3] = cube[3]*1000.0 - 500.0
        cube[4] = cube[4]*1000.0 - 500.0
        cube[5] = cube[5]*100.0 - 100.0
        cube[6] = cube[6]*100.0
        #cube[6] = cube[6]*1.0 - 0.5 + 55.0 
        #cube[6] = cube[6]*1.0
        cube[7] = cube[7]*100.0 - 100.0
        cube[8] = cube[8]*100.0

def myloglike_fit(cube, ndim, nparams):

    A = cube[0]
    B = cube[1]
    phi = cube[2]
    c = cube[3]
    d = cube[4]

    #if c<d:
    #    return -np.inf

    prob = 0
    for key, color in zip(keys,colors):
        vel0 = A*np.sin((2*(np.pi/12.0)*float(key))+phi)+c
        vel1 = B*np.sin((2*(np.pi/12.0)*float(key))+phi+np.pi)+d

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
start=opts.start
stop=opts.end
dataDir = opts.dataDir
errorbudget = opts.errorbudget
baseplotDir = os.path.join(opts.plotDir,'spec_40_%.2f'%errorbudget)

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

moviedir = os.path.join(baseplotDir,'movie')
if not os.path.isdir(moviedir):
    os.makedirs(moviedir)

newdat=loadallspectra()
if opts.doInterp:
    newdat=interpallspectra(newdat, N)

n=1
final=[]

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

wavelengths, spectra = newdat[0], newdat[1:]
#spectra = [spectra[3]]

data_out = {}
for ii,spec in enumerate(spectra):

    plotDir = os.path.join(baseplotDir,'%d'%ii)
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    plotName = "%s/spec.pdf"%(plotDir)
    fig = plt.figure(figsize=(22,28))
    plt.plot(wavelengths,spec,'k--',linewidth=2)
    plt.ylim([0,100])
    plt.grid()
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.savefig(plotName)
    plotName = "%s/spec.png"%(plotDir)
    plt.savefig(plotName)
    plt.close()

    if opts.doMovie:
        plotNameMovie = "%s/movie-%04d.png"%(moviedir,ii)
        cp_command = "cp %s %s" % (plotName, plotNameMovie)
        os.system(cp_command)

    #continue

    parameters = ["p0","p1","p2","vel0","vel1","v1","v2","g1","g2"]
    labels = [r"$p_0$",r"$p_1$",r"$p_2$",r"${\rm vel}_0$",r"${\rm vel}_1$",r"$v_1$",r"$v_2$",r"$g_1$",r"$g_2$"]
    n_params = len(parameters)

    pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    multifile = "%s/2-post_equal_weights.dat"%plotDir
    data = np.loadtxt(multifile)

    #v2_mu, v2_std = 20.0, 3.0
    #v2 = ss.norm(v2_mu, v2_std).ppf(data[:,6])
    #data[:,6] = v2

    p0,p1,p2,vel0,vel1,v1,v2,g1,g2,loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8], data[:,9]
    idx = np.argmax(loglikelihood)
    p0_best, p1_best, p2_best, vel0_best, vel1_best, v1_best, v2_best, g1_best, g2_best = data[idx,0:-1]
    model = func(wavelengths,p0_best,p1_best,p2_best,vel0_best,vel1_best,v1_best,v2_best,g1_best,g2_best)
    model1 = func(wavelengths,p0_best,p1_best,p2_best,vel0_best,vel1_best,v1_best,0,g1_best,g2_best)
    model2 = func(wavelengths,p0_best,p1_best,p2_best,vel0_best,vel1_best,0,v2_best,g1_best,g2_best)

    key = str(ii)
    data_out[key] = {}
    data_out[key]["spec"] = spec
    data_out[key]["model"] = model
    data_out[key]["wavelengths"] = wavelengths
    data_out[key]["vel0"] = vel0
    data_out[key]["vel1"] = vel1
    pts = np.vstack((vel0,vel1)).T
    data_out[key]["kdedir"] = greedy_kde_areas_2d(pts)

    plotName = "%s/corner.pdf"%(plotDir)
    #if os.path.isfile(plotName): continue

    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
    figure.set_size_inches(18.0,18.0)
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/spec.pdf"%(plotDir)
    fig = plt.figure(figsize=(22,28))
    plt.plot(data_out[key]["wavelengths"],data_out[key]["spec"],'k--',linewidth=2)
    plt.plot(data_out[key]["wavelengths"],data_out[key]["model"],'b-',linewidth=2)
    plt.plot(data_out[key]["wavelengths"],model1,'g-',linewidth=2)
    plt.plot(data_out[key]["wavelengths"],model2,'c-',linewidth=2)    
    plt.grid()
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.ylim([0,100])
    plt.savefig(plotName)
    plotName = "%s/spec.png"%(plotDir)
    plt.savefig(plotName)
    plt.close()

    print(noooooooo)

    if opts.doMovie:
        plotNameMovie = "%s/movie-%04d.png"%(moviedir,ii)
        cp_command = "cp %s %s" % (plotName, plotNameMovie)
        os.system(cp_command)

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

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotDir = os.path.join(baseplotDir,'combined')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

n_live_points = 10000
evidence_tolerance = 0.1

parameters = ["A","B","phi","c","d"]
labels = ["A","B",r"$\phi$","c","d"]
n_params = len(parameters)

pymultinest.run(myloglike_fit, myprior_fit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)
#data[:,2] = np.random.rand(len(data[:,2]))
A,B,phi,c,d,loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
idx = np.argmax(loglikelihood)
A_best, B_best, phi_best, c_best, d_best = data[idx,0:-1]

color2 = 'coral'
color1 = 'cornflowerblue'
color3 = 'palegreen'
color4 = 'darkmagenta'

xs, vel0s, vel1s = [], np.zeros((len(keys),len(A))), np.zeros((len(keys),len(A))) 
for jj,key in enumerate(keys):
    x = float(key)
    for ii in range(len(A)):
        vel0s[jj,ii] = A[ii]*np.sin(2*(np.pi/12.0)*float(key)+phi[ii])+c[ii] 
        vel1s[jj,ii] = B[ii]*np.sin(2*(np.pi/12.0)*float(key)+phi[ii]+np.pi)+d[ii]
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

plotName = "%s/violin.pdf"%(baseplotDir)
fig = plt.figure(figsize=(22,28))
for key, color in zip(keys,colors):
    parts = plt.violinplot(data_out[key]["vel0"],[float(key)])
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1) 
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)
    parts = plt.violinplot(data_out[key]["vel1"],[float(key)])
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color2)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color2)
        pc.set_edgecolor(color2)

plt.plot(xs,vel0s_50,'x--',color=color1)
plt.plot(xs,vel1s_50,'o--',color=color2)
plt.fill_between(xs,vel0s_10,vel0s_90,facecolor=color1,edgecolor=color1,alpha=0.2,linewidth=3)
plt.fill_between(xs,vel1s_10,vel1s_90,facecolor=color2,edgecolor=color2,alpha=0.2,linewidth=3)

plt.xlabel('Phase Bin',fontsize=28)
plt.ylabel('Velocity [km/s]',fontsize=28)
plt.grid()
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.savefig(plotName)
plt.close()

color1 = 'r'
color2 = 'b'

plotName = "%s/sigma.pdf"%(baseplotDir)
filename = "%s/vel.dat"%(baseplotDir)
fid = open(filename,'w')
fig = plt.figure(figsize=(22,28))
for key, color in zip(keys,colors):

    fid.write('%.5f '%((float(key)+1)/12.0))

    sigma_low = np.percentile(data_out[key]["vel0"],2.5)
    sigma_high = np.percentile(data_out[key]["vel0"],97.5)
    med = np.median(data_out[key]["vel0"])
    plt.errorbar((float(key)+1)/12.0,med,yerr=np.atleast_2d(np.array([med-sigma_low,sigma_high-med])).T,c=color1,fmt='o-',linewidth=3)

    fid.write('%.5f %.5f %.5f '%(med,sigma_low,sigma_high))

    sigma_low = np.percentile(data_out[key]["vel1"],2.5)
    sigma_high = np.percentile(data_out[key]["vel1"],97.5)
    med = np.median(data_out[key]["vel1"])
    plt.errorbar((float(key)+1)/12.0,med,yerr=np.atleast_2d(np.array([med-sigma_low,sigma_high-med])).T,c=color2,fmt='o-',linewidth=3)

    fid.write('%.5f %.5f %.5f\n'%(med,sigma_low,sigma_high))

fid.close()

plt.plot(xs/12.0,vel0s_50,'x--',color=color1,linewidth=3)
plt.plot(xs/12.0,vel1s_50,'o--',color=color2,linewidth=3)
plt.xlabel('Phase',fontsize=36)
plt.ylabel('Velocity [km/s]',fontsize=36)
plt.grid()
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.savefig(plotName)
plt.close()

plotName = "%s/corner.pdf"%(baseplotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

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
#ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()