#!/usr/bin/env python

#
# Python wrapper for period search routines
# (C) Alex Schwarzenberg-Czerny, 2011                alex@camk.edu.pl
# Based on the wrapper scheme contributed by Ewald Zietsman <ewald.zietsman@gmail.com>

import aov as _aov
import matplotlib.pyplot as pl
import numpy as np
print("")
print("Time Series Analysis Package (TSA)")
print("by Alex Schwarzenberg-Czerny (C)2011")
print("")
print("Routines: amhw, aovw, atrw, covar, fgrid, fouw, normalize, peak,")
print("pldat, plper, pspw, totals")
print("For help type e.g.: help(pyaov.amhw)")
print("")
print("General Reference:")
print("Schwarzenberg-Czerny, A., 1998, Baltic Astronomy, v.7, p.43-69")
print("")

def peak(f):
    '''
    xm,fm,dx = peak(f)

     Scan EVENSAMPLED series for a peak
     Result location & width is in units of f indices (starting from 0)
     INPUT:
       f:- values at even spaced arguments, numpy array (n*1)
     OUTPUT:
       xm-peak location
       fm-peak value
       dx-peak halfwidth at zero level
     SPECIAL CONDITIONS:
       dx<0 no valid peak with positive value
     METHOD:
       Fits parabola to top 3 points
       (C) Alex Schwarzenberg-Czerny, 1999-2005 alex@camk.edu.pl
  '''

    try:
        assert f.size > 2
    except AssertionError:
        print('Input arrays length must be >2')
        return 0
    try:
        assert f.max> 0.
    except AssertionError:
        print('Peak value must be positive')
        return 0

    # maybe something else can go wrong?
    try:
    # any maximum?
        nm=f.argmax()
        fm=f[nm]
        xm=nm
        dx=-1.
        if fm>0. :
            nm=min(f.size,max(3,nm+2))
            f=f[nm-3:nm]
            b=(f[2]-f[0])*0.5
            a2=-(f[0]-2.*f[1]+f[2])
            if a2>0. :
    # parabolic case
                d3=b/a2
                if abs(d3)<=1. :
                   fm=0.5*b*b/a2+f[1]
                   xm=d3+nm-2
                   dx=np.sqrt(b*b+2.*a2*(f[1]-np.median(f)))/a2
        return xm,fm,dx
    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

def normalize(x,mean=0.,var=1.):
    '''
    y = normalize(x,mean=0.,var=1.)

    Purpose: converts series to one of specified mean and variance var.
    Input:
        x: input time series, numpy array (n*1)
    Optional Input:
        mean,var: final mean and variance of the normalized series. 
             For var==0 only mean is shifted.

    Output:  
        x: normalized time series, numpy array (n*1)

    '''
    
    # check the arrays here, make sure they are all the same size
    try:
        assert x.size > 1
    except AssertionError:
        print('Input arrays length must be >1')
        return 0
    try:
        assert var >= 0.
    except AssertionError:
        print('negative target variance')
        return 0

    # maybe something else can go wrong?
    try:
        y=x-np.mean(x)
        va = np.sum(y*y)/(x.size-1)
        if (var>0. and va>0.):
          y=y*np.sqrt(var/va)
        y=y+mean     
        return y

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0


# a wrapper for the covar function

def covar(t1,d1,v1,t2,d2,v2,nct=11,eps=0.0,iscale=0,ifunct=0):
    '''
    lav,lmi,lmx,cc,cmi,cmx=pyaov.covar(t1,d1,v1,t2,d2,v2, \           
                              nct=11,eps=0.0,iscale=0,ifunct=0)    

    Purpose: Calculates cross-correlation function (CCF) for unevenly sampled data
    Input:
        t1,v1,e1: time, value & variance of data set 1, numpy arrays of size (n1*1)
        t2,v2,e2: same for data 2, float size (n2*1)
    Optional Input:
        nct: minimum number of pairs per bin 
        eps: minimum separation of consecutive lags on bin flexible boundary
  
    Output:  
        lav,lmi,lmx(nlag)- average & range of lags for a lag bin
                , numpy arrays of size (nlag*1), and ...                  
        cc,cmi,cmx(nlag)-       ... average & range of correlations

    Reference: Alexander, T., 1997, in Astronomical Time Series, Eds. D. Maoz et al., 
        Dordrecht: Kluwer, ASSL, v218, p163
    '''
# unimplemented    Optional Input:
#        iscale=0: output scale: iscale/=1 -linear; iscale=1 -logarythmic
#        ifunct=0: output function: ifunct/=1 -correlation; ifunct=1 -structure

    # check the arrays here, make sure they are all the same size
    try:
        assert t1.size == d1.size == v1.size
    except AssertionError:
        print('Input arrays1 must be the same dimensions')
        return 0
    try:
        assert t2.size == d2.size == v2.size
    except AssertionError:
        print('Input arrays2 must be the same dimensions')
        return 0
    try:
        assert nct >0
    except AssertionError:
        print('Non positive minimum count per bin')
        return 0

    # maybe something else can go wrong?
    try:
        llav,lmi,lmx,cc,cmi,cmx,ilag = _aov.aov.covar(t1,d1,v1,t2,d2,v2,
            t1.size*t2.size/nct,eps=eps,iscale=iscale,ifunct=ifunct)
       
        return llav[:ilag],lmi[:ilag],lmx[:ilag],cc[:ilag],cmi[:ilag],cmx[:ilag]

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the totals function

def totals(x):
    '''
    pyaov.totals(x)

    Purpose: Evaluate statistical parameters of vector x
    Input:
        x: vector of data, numpy array of size (n*1)
    
    Output:  
        Prints some statistical info
    '''
    # check the arrays here, make sure they are all the same size
    try:
        assert x.size >0
    except AssertionError:
        print('Too few components')
        return 0

    # maybe something else can go wrong?
    try:
        _aov.aov.totals(x)
       
        return 1

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the fgrid function

def fgrid(time):
    '''
    fstop,fstep,fr0=pyaov.fgrid(time)

    Purpose: Evaluate time column and derive a suitable frequency grid
    Input:
        time: numpy array of size (n*1)
    
    Output:  
        fstop,fstep,fr0: Frequency maximum, step & minimum, float
    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size >=5
    except AssertionError:
        print('Too few time moments')
        return 0

    # maybe something else can go wrong?
    try:
        fstop,fstep,fr0 = _aov.aov.fgrid(time)
       
        return fstop,fstep,fr0

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the fouw function

def fouw(time, valin, error, frin, nh2=2):
    '''
    fr,valout,cof,dcof=pyaov.fouw(time, valin, error, frin, nh2=2)

    Purpose: fit data with Fourier series, adjusting frequency if needed.
        Returns frequency, pre-whitened residuals,Fourier coefficients and errors
    Input:
        time, valin, error : numpy arrays of size (n*1)
        frin: initial frequency, float
    Optional input:
        nh2[=2]: no. of model parms. (number of harmonics=nh2/2)
    
    Output:  
        frout: final frequency, float
        valout: residuals from fit/data prewhitened with freq
               numpy array of size (n*1)
        cof, dcof: Fourier coefficients & errors,
               numpy arrays of size (m*1) where m = nh2/2*2+2
               cof(m)-float epoch t0, for double use directly time values
               dcof(m)-error of frequency       
    Fourier series:
        fin(ph)=cof(1)+sum_{n=1}^{nh2/2}(cof(2n-1)Cos(ph*n)+cof(2n)*Sin(ph*n))
        where ph=2pi*frout*(t-t0) and t0=(max(t)+min(t))/2

    Please quote:
        A.Schwarzenberg-Czerny, 1995, Astr. & Astroph. Suppl, 110,405
    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == valin.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # maybe something else can go wrong?
    try:
        frout,valout,cof,dcof = _aov.aov.fouw(time, valin, error, frin, nh2)
       
        return frout,valout,cof,dcof

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the amhw function

def amhw(time, amplitude, error, fstop, fstep, nh2=3, fr0=0.):
    '''
    th,fr,frmax=pyaov.amhw(time, valin, error, fstop, fstep, nh2=3, fr0=0.)

    Purpose: Returns multiharmonic AOV periodogram, obtained by fitting data
        with a series of trigonometric polynomials. For default nh2=3 this
        is Lomb-Scargle periodogram corrected for constant shift.
    Input:
        time, amplitude, error : numpy arrays of size (n*1)
        fstop: frequency to stop calculation at, float
        fstep: size of frequency steps, float
    Optional input:
        nh2[=3]: no. of model parms. (number of harmonics=nh2/2)
        fr0[=0.]: start frequency
    
    Output:
        th,fr: periodogram values & frequencies: numpy arrays of size (m*1)
              where m = (fstop-fr0)/fstep+1
        frmax: frequency of maximum

    Method:
        General method involving projection onto orthogonal trigonometric 
        polynomials is due to Schwarzenberg-Czerny, 1996. For nh2=2 or 3 it reduces
        Ferraz-Mello (1991), i.e. to Lomb-Scargle periodogram improved by constant 
        shift of values. Advantage of the shift is vividly illustrated by Foster (1995).
    Please quote:
        A.Schwarzenberg-Czerny, 1996, Astrophys. J.,460, L107.   
    Other references:
	Foster, G., 1995, AJ v.109, p.1889 (his Fig.1).
        Ferraz-Mello, S., 1981, AJ v.86, p.619.
	Lomb, N. R., 1976, Ap&SS v.39, p.447.
        Scargle, J. D., 1982, ApJ v.263, p.835.
    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == amplitude.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # check the other input values
    try:
        assert fstop > 0
        assert fstep > 0
    except AssertionError:
        print('Frequency stop and step values must be greater than 0')
        return 0

    # maybe something else can go wrong?
    try:
        th,frmax = _aov.aov.aovmhw(time, amplitude, error, fstep, int((fstop-fr0)/fstep+1),fr0=fr0,nh2=nh2)
        
        # make an array that contains the frequencies too
        freqs = np.linspace(fr0, fstop, int((fstop-fr0)/fstep+1))
        return th, freqs,frmax

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the powspw function

def pspw(time, amplitude, error, fstop, fstep, fr0=0.):
    '''
    th,fr,frmax=pyaov.pspw(time, valin, error, fstop, fstep, fr0=0.)

    Purpose: Returns Power Spectrum. Principal use for calculation of window function.  
        For periodogram we recommend amhv routine with nh2=2 instead.
    Input:
        time, amplitude, error : numpy arrays of size (n*1)
        fstop: frequency to stop calculation at, float
        fstep: size of frequency steps, float
    Optional input:
        fr0[=0.]: start frequency
    
    Output:
        th,fr: periodogram values & frequencies: numpy arrays of size (m*1)
              where m = (fstop-fr0)/fstep+1
        frmax: frequency of maximum
    Reference:
        Deeming, T. J., 1975, Ap&SS, v36, pp.137-158

    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == amplitude.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # check the other input values
    try:
        assert fstop > 0
        assert fstep > 0
    except AssertionError:
        print('Frequency stop and step values must be greater than 0')
        return 0

    # maybe something else can go wrong?
    try:
        th,frmax = _aov.aov.powspw(time, amplitude, error, fstep, (fstop-fr0)/fstep+1,fr0=fr0)
        
        # make an array that contains the frequencies too
        freqs = np.linspace(fr0, fstop, (fstop-fr0)/fstep+1)
        return th, freqs,frmax

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the aovtrw function

def atrw(time, amplitude, error, fstop, fstep, nh2=30, fr0=0., ncov=2):
    '''
    th,fr,frmax=pyaov.atrw(time, amplitude, error, fstop, fstep, nh2=30, fr0=0., ncov=2)

    Purpose: Returns AOV periodogram for search of transits/eclipses.
        For this purpose data are fitted with a box-like (inverted top-hat) function.

    Input:
        time, amplitude, error : numpy arrays of size (n*1)
        fstop: frequency to stop calculation at, float
        fstep: size of frequency steps, float
    Optional input:
        nh2[=30]:  number of phase bins
        fr0[=0.]: start frequency
        ncov[=2]: number of coverages
    
    Output:
        th,fr: periodogram values & frequencies: numpy arrays of size (m*1)
              where m = (fstop-fr0)/fstep+1
        frmax: frequency of maximum

    Please quote:
        A. Schwarzenberg-Czerny & J.-Ph. Beaulieu, 2006, MNRAS 365, 165 
    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == amplitude.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # check the other input values
    try:
        assert fstop > 0
        assert fstep > 0
    except AssertionError:
        print('Frequency stop and step values must be greater than 0')
        return 0

    # maybe something else can go wrong?
    try:
        th,frmax = _aov.aov.aovtrw(time, amplitude, error, fstep, (fstop-fr0)/fstep+1,nh2=nh2,fr0=fr0,ncov=ncov)
        
        # make an array that contains the frequencies too
        freqs = np.linspace(fr0, fstop, (fstop-fr0)/fstep+1)
        return th, freqs,frmax

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

# a wrapper for the aovw function

def aovw(time, amplitude, error, fstop, fstep, nh2=3, fr0=0., ncov=2):
    '''
    th,fr,frmax=pyaov.aovw(time, valin, error, fstop, fstep, nh2=3, fr0=0., ncov=2)

    Purpose: Returns AOV periodogram, obtained by phase-folding and binning
        of data.
    Input:
        time, amplitude, error : numpy arrays of size (n*1)
        fstop: frequency to stop calculation at, float
        fstep: size of frequency steps, float
    Optional input:
        nh2[=3]:  number of phase bins
        fr0[=0.]: start frequency
        ncov[=2]: number of coverages
    
    Output:
        th,fr: periodogram values & frequencies: numpy arrays of size (m*1)
              where m = (fstop-fr0)/fstep+1
        frmax: frequency of maximum

    Please quote:
        A. Schwarzenberg-Czerny, 1989, M.N.R.A.S. 241, 153 
    '''

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == amplitude.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # check the other input values
    try:
        assert fstop > 0
        assert fstep > 0
    except AssertionError:
        print('Frequency stop and step values must be greater than 0')
        return 0

    # maybe something else can go wrong?
    try:
        th,frmax = _aov.aov.aovw(time, amplitude, error, fstep, (fstop-fr0)/fstep+1, nh2=nh2, fr0=fr0, ncov=ncov)
        
        # make an array that contains the frequencies too
        freqs = np.linspace(fr0, fstop, (fstop-fr0)/fstep+1)
        return th, freqs,frmax

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0

    # make a plot of data
def pldat(time, value):
    '''
    pyaov.pldat(time, value)

    Purpose:
        Plots periodogram and data folded with its peak frequency

    Input:
       time, value: numpy arrays of size (n*1)
    
    Output:
        Plot window. Close it to proceed further.
    '''

    pl.figure(figsize=(9,9))
    
    pl.subplot(211)
    pl.plot(time, value, 'k.')
#    pl.title('Data')
    pl.xlabel('Time')
    pl.ylabel('Data Value')

    pl.show()

    # make a periodogram plot
def plper(frmax, time, value, freqs, th):
    '''
    pyaov.plper(frmax, time, value, freqs, th)

    Purpose:
        Plots periodogram and data folded with its peak frequency

    Input:
        frmax: the frequency used for phase folding
           set frmax=0 to avoid phase folded plot
        time, value: numpy arrays of size (n*1)
        freqs, th: frequencies & periodogram (m*1)
    
    Output:
        Plot window. Close it to proceed further.
    '''

    pl.figure(figsize=(9,9))
    
    if frmax>0. :   # plot phase folded data
       pl.subplot(211)
       ph=np.mod(time*frmax,1.)
       pl.plot(ph, value, 'k.')
       pl.xlabel('Phase')
       pl.ylabel('Data Value')

    pl.subplot(212) # plot periodogram
    pl.plot(freqs, th, 'k')
    pl.ylabel('Periodogram Value')
    pl.xlabel('Frequency')

    pl.show()

# if you run this script from the command line this will
# run. It will not run if you import this file.
if __name__ == "__main__":
    
    # import aov as _aov
    # generate random data
    x, y, z = _aov.aov.test(100)

    print("Close plots to see the next demo")
    # calculate the periodogram
    th,fr,frmax=amhw(x,y,z,2.8,0.0001,nh2=5,fr0=2.7) 
    # make a plot
    plper(frmax, x, y, fr, th)

    # calculate the periodogram
    th,fr,frmax=pspw(x,y,z,2.8,0.0001,fr0=2.7) 
    # make a plot
    plper(frmax, x, y, fr, th)

    # calculate the periodogram
    th,fr,frmax=atrw(x,y,z,2.8,0.0001,nh2=5,fr0=2.7) 
    print("This example of atrw is poor as data contain no sharp eclipses")
    # make a plot
    plper(frmax, x, y, fr, th)

    # calculate the periodogram
    th,fr,frmax=aovw(x,y,z,2.8,0.0001,nh2=5,fr0=2.7) 
    # make a plot
    plper(frmax, x, y, fr, th)

