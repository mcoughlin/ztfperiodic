#!/usr/bin/env python

import os, sys
import glob
import optparse
import copy
import time
import h5py
import pickle
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
font = {'size'   : 22}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import astropy
from astropy.table import Table, vstack
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from brutus import filters
from brutus import seds
from brutus import utils as butils
from brutus.fitting import BruteForce
from brutus import plotting as bplot

try:
    from penquins import Kowalski
except:
    print("penquins not installed... need to use matchfiles.")

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--doPlots",  action="store_true", default=False)

    parser.add_option("-o","--outputDir",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/brutus")
    parser.add_option("-c","--catalogFile",default="/home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/condor/catalog.tmp.h5")

    parser.add_option("--Ncatalog",default=1,type=int)
    parser.add_option("--Ncatindex",default=0,type=int)

    parser.add_option("-b","--brutusPath",default="/home/michael.coughlin/ZTF/brutus/data/DATAFILES/")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
catalogFile = opts.catalogFile
Ncatalog = opts.Ncatalog
Ncatindex = opts.Ncatindex

plotDir = os.path.join(outputDir,'plots')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

h5Dir = os.path.join(outputDir,'h5')
if not os.path.isdir(h5Dir):
    os.makedirs(h5Dir)

numpyDir = os.path.join(outputDir,'numpy')
if not os.path.isdir(numpyDir):
    os.makedirs(numpyDir)

filt = filters.wise + filters.ps[:-2]
#filt = filters.ps[:-2]

# import EEP tracks
nnfile = '%s/nn_c3k.h5' % opts.brutusPath
mistfile = '%s/MIST_1.2_EEPtrk.h5' % opts.brutusPath

sedfile = os.path.join(opts.brutusPath,'sed.pkl')
if not os.path.isfile(sedfile):
    sedmaker = seds.SEDmaker(nnfile=nnfile, mistfile=mistfile)
    with open(sedfile, 'wb') as handle:
        pickle.dump(sedmaker, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
with open(sedfile, 'rb') as handle:
    sedmaker = pickle.load(handle)

gridfile = '%s/grid_mist_v9_binaries.h5' % opts.brutusPath
if not os.path.isfile(gridfile):
    #sedmaker.make_grid(smf_grid=np.arange(0,1.2,0.2),  # no binaries
    #                   afe_grid=np.array([0.]))  # no afe
    
    sedmaker.make_grid(smf_grid=np.arange(0,1.5,0.5),  # no binaries
                       afe_grid=np.array([0.]))  # no afe

    with h5py.File(gridfile, "w") as out:
        # selection array
        sel = sedmaker.grid_sel
    
        # labels used to generate the grid
        labels = out.create_dataset("labels", data=sedmaker.grid_label[sel])
    
        # parameters generated interpolating over the MIST isochrones
        pars = out.create_dataset("parameters", data=sedmaker.grid_param[sel])
    
        # SEDS generated using the NN from the stellar parameters
        seds = out.create_dataset("mag_coeffs", data=sedmaker.grid_sed[sel])

# import MIST model grid
#gridfile = '%s/grid_mist_v9.h5' % opts.brutusPath
(models_mist, labels_mist, lmask_mist) = butils.load_models(gridfile, filters=filt, include_binaries=True)

# initialize `BruteForce` class
BF_mist = BruteForce(models_mist, labels_mist, lmask_mist)

off_file_mist = '%s/offsets_mist_v9.txt' % opts.brutusPath
off_mist = butils.load_offsets(off_file_mist, filters=filt)

dustfile = '%s/bayestar2019_v1.h5' % opts.brutusPath  # 3-D dust map

with h5py.File(catalogFile, 'r') as f:
    mag, magerr, parallax = f['mag'][:], f['magerr'][:], f['parallax'][:]
    objids, galcoords = f['objids'][:], f['galcoords'][:]

mag_split = np.array_split(mag,Ncatalog)
magerr_split = np.array_split(magerr,Ncatalog)
parallax_split = np.array_split(parallax,Ncatalog)
objids_split = np.array_split(objids,Ncatalog)
galcoords_split = np.array_split(galcoords,Ncatalog)

mag_all, magerr_all, parallax_all = mag_split[Ncatindex], magerr_split[Ncatindex], parallax_split[Ncatindex]
objids, galcoords_all = objids_split[Ncatindex], galcoords_split[Ncatindex]

rv_gauss=(3.32, 0.18)
rvlim=(rv_gauss[0]-5*rv_gauss[1],rv_gauss[0]+5*rv_gauss[1])

for ii in range(len(objids)): 
    if np.mod(ii,100) == 0:
        print('Loading %d/%d'%(ii,len(objids)))

    objid = objids[ii]
    mag = mag_all[ii,:]
    magerr = magerr_all[ii,:]
    parallax, parallax_err = parallax_all[ii,0], parallax_all[ii,1]
    galcoords = galcoords_all[ii,:]

    mask = np.isfinite(magerr) & np.not_equal(magerr,np.zeros(magerr.shape))  # create boolean band mask
    if len(np.where(mask)[0]) < 4: continue

    magerr = np.sqrt(magerr**2 + 0.02**2)
    phot, err = butils.inv_magnitude(mag, magerr)  # convert to flux (in maggies)
    filename = os.path.join(h5Dir, '%d' % objid)
    filenameh5 = filename + '.h5'

    if os.path.isfile(filenameh5):
        os.system('rm %s' % (filenameh5))

    if np.any(np.isnan(phot)) or np.any(np.isnan(err)) or np.isnan(parallax) or np.isnan(parallax_err) or (parallax - 5*parallax_err < 0):
        continue

    print(mag, magerr, np.atleast_2d(phot), np.atleast_2d(err), np.atleast_2d(mask), np.sum(mask))

    # fit a set of hypothetical objects
    BF_mist.fit(np.atleast_2d(phot),  # fluxes (in maggies)
                np.atleast_2d(err),  # errors (in maggies)
                np.atleast_2d(mask),  # band mask (True/False whether band was observed)
                merr_max=1.00,
                rv_gauss=rv_gauss,
                rvlim=rvlim,
                data_labels=np.atleast_2d(np.arange(1)).T,
                save_file=filename,  # filename where results are stored (.h5 automatically added)
                data_coords=np.atleast_2d(galcoords),  # array of (l, b) coordinates for Galactic prior
                parallax=np.atleast_2d(parallax).T,
                parallax_err=np.atleast_2d(parallax_err).T,  # parallax measurements (in mas)
                phot_offsets=off_mist,  # photometric offsets applied to **data**
                dustfile=dustfile,  # 3-D dustmap prior
                Ndraws=2500,  # number of samples to save to disk
                Nmc_prior=500,  # number of Monte Carlo draws used to incorporate priors
                logl_dim_prior=True,  # use chi2 distribution instead of Gaussian
                save_dar_draws=True,  # save (dist, Av, Rv) samples
                running_io=True,  # write out objects as soon as they finish
                verbose=True)  # print progress

    # load results
    f = h5py.File(filenameh5, 'r')
    idxs_mist = f['model_idx'][:]  # model indices
    chi2_mist = f['obj_chi2min'][:]  # best-fit chi2
    nbands_mist = f['obj_Nbands'][:]  # number of bands in fit
    dists_mist = f['samps_dist'][:]  # distance samples
    reds_mist = f['samps_red'][:]  # A(V) samples
    dreds_mist = f['samps_dred'][:]  # R(V) samples
    lnps_mist = f['samps_logp'][:]  # log-posterior of samples

    i = 0

    if opts.doPlots:
        # plot SED (posterior predictive)
        fig, ax, parts = bplot.posterior_predictive(models_mist,  # stellar model grid
                                                    idxs_mist[i],  # model indices
                                                    reds_mist[i],  # A(V) draws
                                                    dreds_mist[i],  # R(V) draws
                                                    dists_mist[i],  # distance draws
                                                    data=phot.flatten(),
                                                    data_err=err.flatten(),  # data
                                                    data_mask=mask.flatten(),  # band mask
                                                    offset=off_mist,  # photometric offsets
                                                    psig=2.,  # plot 2-sigma errors
                                                    labels=filt,  # filters 
                                                    vcolor='blue',  # "violin plot" colors for the posteriors
                                                    pcolor='black')  # photometry colors for the data
        plotName = os.path.join(plotDir,'%d_posterior.png' % objid)
        plt.savefig(plotName)
        plt.close()
       
        # plot corner
        print('Best-fit chi2 (MIST):', chi2_mist[i])
        fig, axes = bplot.cornerplot(idxs_mist[i],
                                     (dists_mist[i], reds_mist[i], dreds_mist[i]),
                                     labels_mist,  # model labels
                                     parallax=parallax, parallax_err=parallax_err,  # overplot if included
                                     show_titles=True, color='blue', pcolor='orange',
                                     fig=plt.subplots(12, 12, figsize=(55, 55)))  #  custom figure
        plotName = os.path.join(plotDir,'%d_corner.png' % objid)
        plt.savefig(plotName)
        plt.close()

    # Ignore age weights.
    labels = [x for x in labels_mist.dtype.names if x != 'agewt']

    # Deal with 1D results.
    samples = labels_mist[idxs_mist[i]]
    samples = np.array([samples[l] for l in labels]).T
    samples = np.atleast_1d(samples)

    idx = labels.index('smf')
    smfsamp = samples[:,idx]

    nmpyfile = os.path.join(numpyDir,'%d_smf.npy' % objid)
    with open(nmpyfile, 'wb') as f:   
        np.save(f, smfsamp)

    print('Fraction of smf > 0.5: %.5f' % (len(np.where(smfsamp > 0.5)[0])/len(smfsamp)))

