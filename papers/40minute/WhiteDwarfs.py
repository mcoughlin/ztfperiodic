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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, RationalQuadratic

import scipy.stats as ss
from scipy.optimize import curve_fit

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/he_hy_white_dwarf_models")

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

def loadallmodels():
    log10Teffmin, log10Teffmax = np.log10(20000), np.log10(25000)
    log10Gmin, log10Gmax = 7.0, 8.2
    Mmin, Mmax = 0.2, 0.8

    FullData = np.empty((0,12))
    filenames = glob.glob(os.path.join(dataDir,'*.Z*'))
    for filename in filenames:
        if "E00" in filename: # models  without hydrogen  envelope
            continue

        filenameSplit = filename.split('/')[-1].split(".")
        nameSplit = filenameSplit[0].split("_")
        solarMass = float(nameSplit[0][1:])*1e-1

        if ((Mmin>solarMass) or (Mmax<solarMass)):
            continue

        if nameSplit[1][0] == "E":
            fractionalMass = 1/float("1"+nameSplit[1])
        else:
            fractionalMass = 1/float(nameSplit[1])
        fractionalMass = np.log10(fractionalMass)
        metallicity = -float(filenameSplit[1][1:]) 

        lines = [line.rstrip('\n').rstrip('\r').rstrip('\x1a') for line in open(filename)]
        for line in lines:
            lineSplit = list(filter(None,line.split(" ")))
            if len(lineSplit) == 0: continue
            data_out = np.loadtxt(lineSplit,dtype=float)

            if len(data_out) == 10:
                log10L,log10Teff,log10Tcen,log10rho,mfrac,log10A,log10R,log10G,X_h,log10HL = data_out
            else:
                log10L,log10Teff,log10Tcen,log10rho,mfrac,log10A,log10R,log10G,X_h = data_out
                log10HL = 0

            if ((log10Teffmin>log10Teff) or (log10Teffmax<log10Teff)):
                continue

            if ((log10Gmin>log10G) or (log10Gmax<log10G)):
                continue

            data = [solarMass,fractionalMass,metallicity,log10L,log10Teff,log10Tcen,mfrac,log10A,log10R,log10G,X_h,log10HL]
            FullData = np.vstack((FullData,data))

    return FullData

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*0.6 + 0.2
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*3.0 - 3.0
        cube[3] = cube[3]*20.0 + 0.0

def myloglike(cube, ndim, nparams):

    mass = cube[0]
    fractionalMass = cube[1]
    metallicity = cube[2]   
    logR = cube[3]

    param_list = [mass, fractionalMass, metallicity, logR]
    param_list_postprocess = np.array(param_list)
    for i in range(len(param_mins)):
        param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    log10Teff_pred, log10Teff_sigma2 = gp_log10Teff.predict(np.atleast_2d(param_list_postprocess), return_std=True)
    log10G_pred, log10G_sigma2 = gp_log10G.predict(np.atleast_2d(param_list_postprocess), return_std=True)

    #vals = np.array([10**log10Teff_pred[0],log10G_pred[0]]).T
    #kdeeval = kde_eval(kdedir_pts,vals)[0]
    #prob = np.log(kdeeval)

    T_mu, T_std = 22500, 300
    T_prob = ss.norm(T_mu, T_std).logpdf(10**log10Teff_pred[0])

    logg_mu, logg_std = 7.6, 0.3
    logg_prob = ss.norm(logg_mu, logg_std).logpdf(log10G_pred[0])

    prob = T_prob + logg_prob

    if np.isnan(prob):
        prob = -np.inf

    return prob

# Parse command line
opts = parse_commandline()

dataDir = opts.dataDir
baseplotDir = os.path.join(opts.plotDir,'WhiteDwarf_40')

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

newdat=loadallmodels()
param_array = newdat[:,[0,1,2,8]]
param_array_postprocess = np.array(param_array)
param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
for i in range(len(param_mins)):
    param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i]) 

log10Teff = newdat[:,4]
log10G = newdat[:,9] 

kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
gp_log10Teff = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
gp_log10Teff.fit(param_array_postprocess, log10Teff)
gp_log10G = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
gp_log10G.fit(param_array_postprocess, log10G)

#T_mu, T_std = 47166, 887
#T_unit = np.random.rand(1000)
#T = ss.norm(T_mu, T_std).ppf(T_unit)

#logg_mu, logg_std = 7.82, 0.061
#logg_unit = np.random.rand(1000)
#logg = ss.norm(logg_mu, logg_std).ppf(logg_unit)

#pts = np.vstack((T,logg)).T
#kdedir = greedy_kde_areas_2d(pts)
#kdedir_pts = copy.deepcopy(kdedir)

n_live_points = 5000
evidence_tolerance = 0.1
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["Mass","fractionalMass","metallicity","logR"]
labels = [r"$M$","frac","metal",r"$\log R$"]
n_params = len(parameters)

plotDir = os.path.join(baseplotDir,'com')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

mass, fractionalMass, metallicity, logR, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4] 
idx = np.argmax(loglikelihood)
mass_best, fractionalMass_best, metallicity_best, logR_best = data[idx,0:-1]

param_list = [mass_best, fractionalMass_best, metallicity_best, logR_best]
param_list_postprocess = np.array(param_list)
for i in range(len(param_mins)):
    param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

log10Teff_pred, log10Teff_sigma2 = gp_log10Teff.predict(np.atleast_2d(param_list_postprocess), return_std=True)
log10G_pred, log10G_sigma2 = gp_log10G.predict(np.atleast_2d(param_list_postprocess), return_std=True)

T_best = 10**log10Teff_pred[0]
logg_best = log10G_pred[0]*1.0

print("Teff: %.5f, log(g): %.5f"%(T_best,logg_best))

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

