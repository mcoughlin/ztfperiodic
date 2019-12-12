#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:05:05 2019

@author: yuhanyao
"""
import os
import h5py
import numpy as np
import matplotlib
fs = 14
matplotlib.rcParams.update({'font.size': fs})
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_gaia_subplot(gaia, ax, tboutputDir, doTitle=False):
    if not len(gaia)==1:
        return None
    
    WDcat = os.path.join(tboutputDir,'GaiaHRSet.hdf5') # 993635 targets
    with h5py.File(WDcat, 'r') as f:
        gmag, bprpWD = f['gmag'][:], f['bp_rp'][:]
        parallax = f['parallax'][:]
    absmagWD = gmag + 5 * (np.log10(np.abs(parallax))-2)
    
    hist2 = ax.hist2d(bprpWD,absmagWD, bins=100,zorder=0,norm=LogNorm())
    ax.set_xlim([-1,4.0])
    ax.set_ylim([-5,18])
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(hist2[3])
    cbar.set_label('Object Count')
    ax.set_xlabel('Gaia BP - RP')
    ax.set_ylabel('Gaia $M_G$')
    
    bp_rp = gaia['BP-RP'].data.data[0]
    Plx = gaia['Plx'].data.data[0] # mas
    e_Plx = gaia['e_Plx'].data.data[0] # mas
    gofAL = gaia["gofAL"].data.data[0]
    Gmag = gaia['Gmag'].data.data[0]
    
    if not (~np.isnan(bp_rp))&(~np.isnan(Plx)): # if any of the parameters are nan
        return None
    
    # distance in pc
    if Plx > 0 :
        d_pc = 1 / (Plx*1e-3)
        d_pc_upper = 1 / ((Plx-e_Plx)*1e-3)
        d_pc_lower = 1 / ((Plx+e_Plx)*1e-3)
    
        absmag = Gmag - 5 * (np.log10(d_pc)-1)
        absmag_lower = Gmag - 5 * (np.log10(d_pc_lower)-1)
        if d_pc_upper<0:
            absmag_upper = -99
        else:
            absmag_upper = Gmag - 5 * (np.log10(d_pc_upper)-1)
            
        if ~np.isnan(bp_rp):
            ax.plot(bp_rp, absmag, 'o', c='r', zorder=1, markersize=5)
            ax.plot([bp_rp, bp_rp], [absmag_lower, absmag_upper], 'r-')
        
        if doTitle:
            ax.set_title("d = %d [pc], gof = %.1f"%(d_pc, gofAL), fontsize = fs)
        
       
