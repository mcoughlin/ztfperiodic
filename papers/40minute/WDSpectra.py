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

import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.special import wofz
from astropy.modeling.models import Voigt1D as voigt
from astropy.modeling.models import Gaussian1D as gaussian
from astropy.modeling.models import Lorentz1D as lorentzian

from astropy.io import fits

import corner
import pymultinest

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, RationalQuadratic

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/spectra/40minute")
    parser.add_option("-w","--wdDir",default="../../data/levenhagen17")
    parser.add_option("-N","--N",type=int,default=100)
    parser.add_option("-n","--nstack",default=1,type=int)

    parser.add_option("-s","--start",type=int,default=375)
    parser.add_option("-e","--end",type=int,default=1500)
    #parser.add_option("-s","--start",type=int,default=750)
    #parser.add_option("-e","--end",type=int,default=1050)
    parser.add_option("--errorbudget",type=float,default=0.1)

    parser.add_option("--doSaveModel",  action="store_true", default=False)
    parser.add_option("--doMovie",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

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

def myprior_fit(cube, ndim, nparams):

        cube[0] = cube[0]*83000 + 17000
        cube[1] = cube[1]*2.5 + 7.0

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*83000 + 17000
        cube[1] = cube[1]*2.5 + 7.0
        cube[2] = cube[2]*10.0 - 5.0

def myloglike_fit(cube, ndim, nparams):

    T = cube[0]
    logg = cube[1]

    prob = 0
    for key, color in zip(keys,colors):
        kdedir = data_out[key]["kdedir"]
        vals = np.array([T,logg]).T
        kdeeval = kde_eval(kdedir,vals)[0]
        prob = prob + np.log(kdeeval)

        if not np.isfinite(prob):
            break

    if np.isnan(prob):
        prob = -np.inf

    return prob

def myloglike(cube, ndim, nparams):

    T = cube[0]
    logg = cube[1]
    K = 10**cube[2]

    param_list = [T,logg]
    model = calc_spec(param_list, wddata=wddata, svd_model=svd_model)
    sigma = spec * errorbudget

    x = K*model - spec
    prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
    prob = np.sum(prob)

    print(prob)

    return prob

def calc_svd(wddata, n_coeff = 10):

    print("Calculating SVD model...")

    spec_array = []
    param_array = []
    keys = list(wddata.keys())
    for key in keys:
        spec_array.append(wddata[key]["spec"])
        param_array.append([wddata[key]["temp"],wddata[key]["logg"]])

    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    spec_array_postprocess = np.array(spec_array)
    mins,maxs = np.min(spec_array_postprocess,axis=0),np.max(spec_array_postprocess,axis=0)
    for i in range(len(mins)):
        spec_array_postprocess[:,i] = (spec_array_postprocess[:,i]-mins[i])/(maxs[i]-mins[i])
    spec_array_postprocess[np.isnan(spec_array_postprocess)]=0.0

    UA, sA, VA = np.linalg.svd(spec_array_postprocess, full_matrices=True)
    VA = VA.T

    n, n = UA.shape
    m, m = VA.shape

    cAmat = np.zeros((n_coeff,n))
    cAvar = np.zeros((n_coeff,n))
    for i in range(n):
        if np.mod(i, 100) == 0:
            print("%d/%d" %(i, n))
        cAmat[:,i] = np.dot(spec_array_postprocess[i,:],VA[:,:n_coeff])
        ErrorLevel = 1.0
        errors = ErrorLevel*spec_array_postprocess[i,:]
        cAvar[:,i] = np.diag(np.dot(VA[:,:n_coeff].T,np.dot(np.diag(np.power(errors,2.)),VA[:,:n_coeff])))
    cAstd = np.sqrt(cAvar)

    nsvds, nparams = param_array_postprocess.shape
    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
    gps = []
    for i in range(n_coeff):
        if np.mod(i, 5) == 0:
            print("%d/%d" %(i, n_coeff))
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
        gp.fit(param_array_postprocess, cAmat[i,:])
        gps.append(gp)

    svd_model = {}
    svd_model["n_coeff"] = n_coeff
    svd_model["param_array"] = param_array
    svd_model["cAmat"] = cAmat
    svd_model["cAstd"] = cAstd
    svd_model["VA"] = VA
    svd_model["param_mins"] = param_mins
    svd_model["param_maxs"] = param_maxs
    svd_model["mins"] = mins
    svd_model["maxs"] = maxs
    svd_model["gps"] = gps

    print("Finished calculating SVD model...")

    return svd_model

def calc_spec(param_list, wddata=None, svd_model=None):

    if svd_model == None:
        svd_model = calc_svd(wddata, n_coeff = 10) 

    n_coeff = svd_model["n_coeff"]
    param_array = svd_model["param_array"]
    cAmat = svd_model["cAmat"]
    VA = svd_model["VA"]
    param_mins = svd_model["param_mins"]
    param_maxs = svd_model["param_maxs"]
    mins = svd_model["mins"]
    maxs = svd_model["maxs"]
    gps = svd_model["gps"]

    param_list_postprocess = np.array(param_list)
    for i in range(len(param_mins)):
        param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    cAproj = np.zeros((n_coeff,))
    for i in range(n_coeff):
        gp = gps[i]
        y_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
        cAproj[i] = y_pred

    spec_back = np.dot(VA[:,:n_coeff],cAproj)
    spec_back = spec_back*(maxs-mins)+mins

    return np.squeeze(spec_back)

# Parse command line
opts = parse_commandline()

FullData=[]

nstack = opts.nstack
N=opts.N
start=opts.start
stop=opts.end
dataDir = opts.dataDir
wdDir = opts.wdDir
errorbudget = opts.errorbudget
baseplotDir = os.path.join(opts.plotDir,'spec_wd_40_%.2f'%errorbudget)

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

ModelPath = '%s/svdmodels'%(baseplotDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

newdat=loadallspectra()
wavelengths, spectra = newdat[0], newdat[1:]
#spectra = [spectra[3]]

wddata = {}
filenames = glob.glob(os.path.join(wdDir,'*.fits'))
for ii, filename in enumerate(filenames):
    filenameSplit = filename.split("/")[-1]
    temp, logg = 1e3*int(filenameSplit[1:4]), 1e-1*int(filenameSplit[5:7])
    hdul = fits.open(filename)
    data = hdul[1].data

    idx = np.where((data['col1'] >= wavelengths[0]) &
                   (data['col1'] <= wavelengths[-1]))[0]

    spec = np.interp(wavelengths, data['col1'][idx]*1.0, data['col2'][idx]/1e8)

    wddata[ii] = {}
    wddata[ii]["spec"] = spec
    wddata[ii]["temp"] = temp
    wddata[ii]["logg"] = logg

    hdul.close()

modelfile = os.path.join(ModelPath,'model.pkl')
if opts.doSaveModel:
    svd_model = calc_svd(wddata, n_coeff = 2)
    with open(modelfile, 'wb') as handle:
        pickle.dump(svd_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(modelfile, 'rb') as handle:
        svd_model = pickle.load(handle)

keys = list(wddata.keys())
#plotName = "%s/wdspec.pdf"%(baseplotDir)
#fig = plt.figure(figsize=(22,28))
#for key in keys:
#    plt.loglog(wavelengths,wddata[key]["spec"],'k--',linewidth=2)
#plt.grid()
#plt.xlabel("Wavelength [$\AA$]", fontsize=36)
#plt.ylabel("$F_\lambda$ [$10^8$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]", fontsize=36)
#plt.yticks(fontsize=36)
#plt.xticks(fontsize=36)
#plt.savefig(plotName)
#plotName = "%s/wdspec.png"%(baseplotDir)
#plt.savefig(plotName)
#plt.close()

param_list = [wddata[0]["temp"],wddata[0]["logg"]]
sample_spec = calc_spec(param_list, wddata=wddata, svd_model=svd_model)

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

data_out = {}
for ii,spec in enumerate(spectra):

    plotDir = os.path.join(baseplotDir,'%d'%ii)
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    plotName = "%s/spec.pdf"%(plotDir)
    fig = plt.figure(figsize=(22,28))
    plt.plot(wavelengths,spec,'k--',linewidth=2)
    plt.plot(wavelengths,10*wddata[0]["spec"],'g.-',linewidth=2)
    plt.plot(wavelengths,10*sample_spec,'r-',linewidth=2)
    plt.ylim([0,100])
    plt.grid()
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.savefig(plotName)
    plotName = "%s/spec.png"%(plotDir)
    plt.savefig(plotName)
    plt.close()

    parameters = ["T","logg","K"]
    labels = ["T",r"$\log(g)$","K"]
    n_params = len(parameters)

    pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

    multifile = "%s/2-post_equal_weights.dat"%plotDir
    data = np.loadtxt(multifile)

    T, logg, K, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
    idx = np.argmax(loglikelihood)
    T_best, logg_best, K_best = data[idx,0:-1]

    param_list = [T_best,logg_best]
    model = 10**K_best * calc_spec(param_list, wddata=wddata, svd_model=svd_model)

    key = str(ii)
    data_out[key] = {}
    data_out[key]["spec"] = spec
    data_out[key]["model"] = model
    data_out[key]["wavelengths"] = wavelengths
    data_out[key]["T"] = T
    data_out[key]["logg"] = logg
    pts = np.vstack((T,logg)).T
    data_out[key]["kdedir"] = greedy_kde_areas_2d(pts)

    plotName = "%s/corner.pdf"%(plotDir)
    if os.path.isfile(plotName): continue

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
    plt.grid()
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.ylim([0,100])
    plt.savefig(plotName)
    plotName = "%s/spec.png"%(plotDir)
    plt.savefig(plotName)
    plt.close()

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotDir = os.path.join(baseplotDir,'combined')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

n_live_points = 10000
evidence_tolerance = 0.1

parameters = ["T","logg"]
labels = ["T",r"$\log(g)$"]
n_params = len(parameters)

pymultinest.run(myloglike_fit, myprior_fit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)
T, logg, loglikelihood = data[:,0], data[:,1], data[:,2]
idx = np.argmax(loglikelihood)
T_best, logg_best= data[idx,0:-1]

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
        plt.xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)

ax1.set_zorder(1)
plt.savefig(plotName)
plt.close()
