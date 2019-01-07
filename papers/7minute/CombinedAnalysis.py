
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:52:14 2018

@author: Kevin
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
plt.rcParams.update({'font.size': 30})

import scipy.stats as ss
from scipy import interpolate

import corner
import pymultinest

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../../output")
    parser.add_option("-p","--plotDir",default="../../plots")
    parser.add_option("-d","--dataDir",default="../../data/posteriors")

    opts, args = parser.parse_args()

    return opts

def mc2ms(mc,eta):
    """
    Utility function for converting mchirpmass,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1/fraction

    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)

    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
    return [m1,m2]


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

def ms2mc(m1,m2):
    eta = m1*m2/( (m1+m2)*(m1+m2) )
    mchirp = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
    q = m2/m1

    return (mchirp,eta,q)

def prior(m1,m2):

    mchirp,eta,q = ms2mc(m1,m2)
    if (mchirp < chirpmass_min) or (mchirp > chirpmass_max):
        return 0.0
    else:
        return 1.0

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*0.6 + 0.2
        cube[1] = cube[1]*0.5 

def myloglike(cube, ndim, nparams):

    m1 = cube[0]
    m2 = cube[1]

    mchirp,eta,q = ms2mc(m1,m2)

    vals = np.array([m1,m2]).T
    kdeeval_models = kde_eval(kdedir_models_pts,vals)[0]
    prob_models = np.log(kdeeval_models)

    kdeeval_lightcurve = kde_eval(kdedir_lightcurve_pts,vals)[0]
    prob_lightcurve = np.log(kdeeval_lightcurve)

    prob = prob_models + prob_lightcurve

    if prior(m1,m2) == 0.0:
        prob = -np.inf

    if np.isnan(prob):
        prob = -np.inf

    return prob

# Parse command line
opts = parse_commandline()

p=414.7
smear = 1.20072
chirpmass_min, chirpmass_max = 0.295638, 0.317229
G = 6.67408*1e-11

dataDir = opts.dataDir
baseplotDir = os.path.join(opts.plotDir,'combined')

if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

spectra_file = os.path.join(opts.dataDir,'spectra.dat')
data_spectra = np.loadtxt(spectra_file,usecols=(0,1))
ndata=len(data_spectra[:,0])

models_file = os.path.join(opts.dataDir,'models.dat')
data_models = np.loadtxt(models_file,usecols=(0,1,2,3))
ndatam=len(data_models[:,0])

lightcurves_file = os.path.join(opts.dataDir,'lightcurves.dat')
data_lightcurves = np.loadtxt(lightcurves_file)
ndatal=len(data_lightcurves[:,0])

lightcurves_r1_model = np.random.choice(data_lightcurves[:,0],size=ndatam)
lightcurves_r1 = np.random.choice(data_lightcurves[:,0],size=ndata)
lightcurves_r2 = np.random.choice(data_lightcurves[:,1],size=ndata)
lightcurves_q = data_lightcurves[:,7]
q_5, q_50, q_95 = np.percentile(lightcurves_q,5), np.percentile(lightcurves_q,50), np.percentile(lightcurves_q,95)
inclination = np.random.choice(data_lightcurves[:,3],size=ndata)

models_m1, models_r1 = data_models[:,0], 10**data_models[:,3]
models_a=models_r1/lightcurves_r1_model
models_m2 = (1/2e30)*(4.0*(np.pi**2)/G)*((1e-2*models_a)**3)/p**2 - models_m1
idx = np.where(models_m1>=models_m2)[0]
models_m1, models_m2 = models_m1[idx], models_m2[idx]

pts = np.vstack((models_m1,models_m2)).T
kdedir = greedy_kde_areas_2d(pts)
kdedir_models_pts = copy.deepcopy(kdedir)

# kevin calculation
pdot_m=np.random.normal(0.745134,0.00228,ndata)

# JAN (replace me)
col=np.random.normal(0.68,0.04,ndata)

inc=np.sin(inclination* np.pi / 180.)**3

k1=smear*data_spectra[:,0]*1000
k_m2=data_spectra[:,1]*1000

r2=lightcurves_r2*1.0
k2=1/2.0*((4*col*k1*k_m2*smear*r2+(-col*smear*k_m2*r2-smear*k_m2)**2)**0.5+col*smear*k_m2*r2+smear*k_m2)

m1=(k2**3*p/(2*np.pi*6.67e-11*inc)*(1+k1/k2)**2)
m2=(k1**3*p/(2*np.pi*6.67e-11*inc)*(1+k2/k1)**2)
M = ((m1*m2)**(3/5))/((m1 + m2)**(1/5))

pts = np.vstack((m1,m2)).T
kdedir = greedy_kde_areas_2d(pts)
kdedir_lightcurve_pts = copy.deepcopy(kdedir)

f=2/p
fdot = 96/5*np.pi**(8/3)*(6.67e-11*M/((3e8)**3))**(5/3)*f**(11/3)
pdot = -fdot*(p/2)**2*2
pdot2 = pdot*3.154*10**7*1000

m2_s=np.sort(m2)
m1_s=np.sort(m1)

n_live_points = 500
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["mass1","mass2"]
labels = [r"$m_1$",r"$m_2$"]
n_params = len(parameters)

plotDir = os.path.join(baseplotDir,'com')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

mass1, mass2, loglikelihood = data[:,0], data[:,1], data[:,2]
idx = np.argmax(loglikelihood)
mass1_best, mass2_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

plotName = "%s/constraints.pdf"%(baseplotDir)
fig = plt.figure(figsize=(28,28))

my_cmap = cm.get_cmap('gray')
my_cmap.set_bad((0,0,0))
(h, xedges, yedges, image) = plt.hist2d(m1/2e30,m2/2e30,bins=(100,100),norm = matplotlib.colors.LogNorm(),cmap=my_cmap)

H, xedges1, yedges1 = np.histogram2d(models_m1,models_m2,bins=(xedges, yedges))
x = (xedges1[1:] + xedges1[:-1])/2.0
y = (yedges1[1:] + yedges1[:-1])/2.0
X, Y = np.meshgrid(x, y)
H = H / np.sum(H)
n = 1000
t = np.linspace(0, H.max(), n)
integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
f = interpolate.interp1d(integral, t)
t_contours = f(np.array([0.68, 0.5, 0.32]))
CS = plt.contour(X, Y, H.T, t_contours, colors='r')

H, xedges2, yedges2 = np.histogram2d(mass1,mass2,bins=(xedges, yedges))
x = (xedges2[1:] + xedges2[:-1])/2.0
y = (yedges2[1:] + yedges2[:-1])/2.0
X, Y = np.meshgrid(x, y)
H = H / np.sum(H)
n = 1000
t = np.linspace(0, H.max(), n)
integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
f = interpolate.interp1d(integral, t)
t_contours = f(np.array([0.68, 0.5, 0.32]))
CS = plt.contour(X, Y, H.T, t_contours, colors='g')

plt.xlim(0.22,1.2)
plt.ylim(0,0.8)

eta=np.linspace(0,0.25,10000)
data=mc2ms(chirpmass_min,eta)
plt.plot(data[0],data[1],'w--')
data=mc2ms(chirpmass_max,eta)
plt.plot(data[0],data[1],'w--')

x=np.linspace(0,1.2,1000)
# measured q's by lightcurve (replace me)
y1=q_5*x
y2=q_95*x
#plt.plot(x,y1,'b--')
#plt.plot(x,y2,'b--')

plt.xlabel('$m_1$')
plt.ylabel('$m_2$')

plt.savefig(plotName)
plt.close()

a=(p**2*6.67*10**(-11)*(m1+m2)/(4*3.141592654**2))**(1/3)
a_s=np.sort(a)

# radii/a come from lightcurve (replace me) 
# a comes from spectroscopy
r1=lightcurves_r1*a
r2=lightcurves_r2*a

r1s=np.sort(r1)
r2s=np.sort(r2)

g=np.log10(100*(6.67*10**(-11)*m1/(r1)**2))
gs=np.sort(g)
q=m2/m1
qs=np.sort(q)

T1=np.random.normal(47166, 887,ndata)
sf=np.random.normal(0.032874764615433164,0.0030469788314203454,ndata)*0.00844432/0.210029
T2=T1*((sf)**(1/4))
T2_s=np.sort(T2)

# grid (like Tom's) + Teff + m1 # using 0.6
bolmag=np.random.normal(8.524, 0.185,ndata)
distance=10**(+(20.5-bolmag)/5-2)
D_s=np.sort(distance)


