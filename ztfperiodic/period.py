
import os, sys
import glob
import optparse
from functools import partial

import tables
import pandas as pd
import numpy as np
import h5py

def rephase(data, period=1.0, shift=0.0, col=0, copy=True):
    """
    Returns *data* (or a copy) phased with *period*, and shifted by a
    phase-shift *shift*.

    **Parameters**

    data : array-like, shape = [n_samples, n_cols]
        Array containing the time or phase values to be rephased in column
        *col*.
    period : number, optional
        Period to phase *data* by (default 1.0).
    shift : number, optional
        Phase shift to apply to phases (default 0.0).
    col : int, optional
        Column in *data* containing the time or phase values to be rephased
        (default 0).
    copy : bool, optional
        If True, a new array is returned, otherwise *data* is rephased
        in-place (default True).

    **Returns**

    rephased : array-like, shape = [n_samples, n_cols]
        Array containing the rephased *data*.
    """
    rephased = np.ma.array(data, copy=copy)
    rephased[:, col] = get_phase(rephased[:, col], period, shift)

    return rephased

def get_phase(time, period=1.0, shift=0.0):
    """
    Returns *time* transformed to phase-space with *period*, after applying a
    phase-shift *shift*.

    **Parameters**

    time : array-like, shape = [n_samples]
        The times to transform.
    period : number, optional
        The period to phase by (default 1.0).
    shift : number, optional
        The phase-shift to apply to the phases (default 0.0).

    **Returns**

    phase : array-like, shape = [n_samples]
        *time* transformed into phase-space with *period*, after applying a
        phase-shift *shift*.
    """
    return (time / period - shift) % 1

def CE(period, data, xbins=10, ybins=5):
    """
    Returns the conditional entropy of *data* rephased with *period*.

    **Parameters**

    period : number
        The period to rephase *data* by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    xbins : int, optional
        Number of phase bins (default 10).
    ybins : int, optional
        Number of magnitude bins (default 5).
    """
    if period <= 0:
        return np.PINF

    import fast_histogram

    #r = rephase(data, period)
    #r = np.ma.array(data, copy=True)
    data[:, 0] = np.mod(data[:, 0], period)
    #bins, xedges, yedges = np.histogram2d(r[:,0], r[:,1], bins=[xbins, ybins], range=[[0,1], [0,1]])
    bins = fast_histogram.histogram2d(data[:,0], data[:,1], range=[[0, 1], [0, 1]], bins=[xbins, ybins])

    size = data.shape[0]

    if size > 0:
        # bins[i,j] / size
        divided_bins = bins / size
        # indices where that is positive
        # to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1) #changed 0 by 1

        # array is repeated row-wise, so that it can be sliced by arg_positive
        #column_sums = np.repeat(np.reshape(column_sums, (1,-1)), xbins, axis=0)
        column_sums = np.repeat(np.atleast_2d(column_sums).T, ybins, axis=1)

        # select only the elements in both arrays which correspond to a
        # positive bin
        select_divided_bins = divided_bins[arg_positive]
        select_column_sums  = column_sums[arg_positive]

        # initialize the result array
        A = np.empty((xbins, ybins), dtype=float)
        # store at every index [i,j] in A which corresponds to a positive bin:
        # bins[i,j]/size * log(bins[i,:] / size / (bins[i,j]/size))
        A[ arg_positive] = select_divided_bins \
                         * np.log(select_column_sums / select_divided_bins)
        # store 0 at every index in A which corresponds to a non-positive bin
        A[~arg_positive] = 0

        # return the summation
        return np.sum(A)
    else:
        return np.PINF

def CE_cupy(period, data, xbins=10, ybins=5):
    """
    Returns the conditional entropy of *data* rephased with *period*.

    **Parameters**

    period : number
        The period to rephase *data* by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    xbins : int, optional
        Number of phase bins (default 10).
    ybins : int, optional
        Number of magnitude bins (default 5).
    """

    import cupy as cp

    if period <= 0:
        return np.PINF

    shift=0.0
    col=0
    
    r = cp.array(data)
    r[:, 0] = cp.mod(r[:, col], period)

    print(r)
    print(stop)
   
    bins, xedges, yedges = np.histogram2d(r[:,0], r[:,1], [xbins, ybins], [[0,1], [0,1]])
    size = r.shape[0]

    if size > 0:
        # bins[i,j] / size
        divided_bins = bins / size
        # indices where that is positive
        # to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1) #changed 0 by 1
        # array is repeated row-wise, so that it can be sliced by arg_positive
        column_sums = np.repeat(np.reshape(column_sums, (xbins,1)), ybins, axis=1)
        #column_sums = np.repeat(np.reshape(column_sums, (1,-1)), xbins, axis=0)


        # select only the elements in both arrays which correspond to a
        # positive bin
        select_divided_bins = divided_bins[arg_positive]
        select_column_sums  = column_sums[arg_positive]

        # initialize the result array
        A = np.empty((xbins, ybins), dtype=float)
        # store at every index [i,j] in A which corresponds to a positive bin:
        # bins[i,j]/size * log(bins[i,:] / size / (bins[i,j]/size))
        A[ arg_positive] = select_divided_bins \
                         * np.log(select_column_sums / select_divided_bins)
        # store 0 at every index in A which corresponds to a non-positive bin
        A[~arg_positive] = 0

        # return the summation
        return np.sum(A)
    else:
        return np.PINF
