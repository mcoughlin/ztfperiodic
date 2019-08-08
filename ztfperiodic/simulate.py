"""
@author: 0cooper

Creates light curve with given binary parameters, including Pdot
    
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import ellc
from ellc import lc
import pandas as pd

# default params
m1 = 0.5 # mass 1 (msun)
m2 = 0.25 # mass 2 (msun)
r1 = 1.5e-2 # radius 1 (rsun)
r2 = 3e-2 # radius 2 (rsun)
a = 10e-2 # semimajor axis (rsun)
i = 80 # inclination (deg)
P0 = 0.005 # period (days)
sbratio = 0.5 # surface brightness ratio
Pdot = -1e-11 # rate of change of period (days/days)


def phase_fold(t,y,p):
    """
    Phase folds curve to given period
    
    INPUT:
            t = time array
            
            y = flux array
            
            p = period of function 
        
    OUTPUT:
            returns plot of phase folded curve
        
    """
    phases=np.remainder(t,p)/p
    phases=np.concatenate((phases,phases+1))
    y=np.concatenate((y,y))
    plt.plot(phases,y,'c.',label=r'\.P = 0')
    plt.legend()
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    return
  

def time(n=100,mean_dt=3,sig_t=2):
    """
    Returns time array of length n with dt given by Gaussian distribution
    
    INPUT:
            n = length of time array
                Default: 100
                
            mean_dt = average dt between each time value
                Default: 3
                
            sig_t = standard deviation of the Gaussian
                Default: 2
            
    OUTPUT:
            t = time array
        
    """
    t = np.zeros(n)

    for i in range(len(t)):
        if i != 0:
            t[i] += t[i-1] + np.abs(np.random.normal(mean_dt,sig_t,1)) 
            
    return t        
            
            

def ellc(t_obs, radius_1=r1/a, radius_2=r2/a, sbratio=0.5, incl=80, 
       light_3 = 0, 
       t_zero = 0, period = P0,
       a = a,
       q = m1/m2,
       f_c = None, f_s = None,
       ldc_1 = None, ldc_2 = None,
       gdc_1 = None, gdc_2 = None,
       didt = None,
       domdt = None,
       rotfac_1 = 1, rotfac_2 = 1,
       hf_1 = 1.5, hf_2 = 1.5,
       bfac_1 = None, bfac_2 = None,
       heat_1 = None, heat_2 = None, 
       lambda_1 = None, lambda_2 = None,
       vsini_1 = None, vsini_2 = None,
       t_exp=None, n_int=None, 
       grid_1='default', grid_2='default',
       ld_1=None, ld_2=None,
       shape_1='sphere', shape_2='sphere',
       spots_1=None, spots_2=None, 
       exact_grav=False, verbose=1):

    """
    Uses ellc binary star model [1] to create light curve of binary star system

    See lc.py for documentation of all params and defaults
    
    INPUT:
            t = time array from time() function
                Default: n=100, mean_dt=3, sig_t=2
                
            r1, r2 = radii of the objects
                Default: r1 = 1.5e-2 rsun, r2 = 3e-2 rsun
                
            a = semi-major axis of binary system
                Default: 10e-2 rsun
                
            sbratio = surface brightness ratio
                Default: 0.5
                
            incl = inclination of orbit
                Default: 80 deg
                
            period = period of orbit (assumes p dot = 0)
                Default: 0.005 days (432 s)
                
            m1, m2 = masses of objects
                Default: m1 = 0.5 msun, m2 = 0.25 msun
           
           (for other params, see ellc/lc.py documentation)
                
    OUTPUT:
            flux = flux array for binary system
       

    ------------
    References:
        [1] Maxted, P.F.L. 2016. A fast, flexible light curve model for detached
            eclipsing binary stars and transiting exoplanets. A&A 591, A111, 2016.

    """  

    flux = lc(t_obs=t_obs, radius_1=radius_1, radius_2=radius_2, sbratio=sbratio,
         incl=incl,period=period,q=q,a=a,
         light_3=light_3,t_zero=t_zero,f_c=f_c, f_s=f_s,
         ldc_1=ldc_1, ldc_2=ldc_2,
         gdc_1=gdc_1, gdc_2=gdc_2,
         didt=didt,
         domdt=domdt,
         rotfac_1=rotfac_1, rotfac_2=rotfac_2,
         hf_1=hf_1, hf_2=hf_2,
         bfac_1=bfac_1, bfac_2=bfac_2,
         heat_1=heat_1, heat_2=heat_2, 
         lambda_1=lambda_1, lambda_2=lambda_2,
         vsini_1=vsini_1, vsini_2=vsini_2,
         t_exp=t_exp, n_int=n_int, 
         grid_1=grid_1, grid_2=grid_2,
         ld_1=ld_1, ld_2=ld_2,
         shape_1=shape_1, shape_2=shape_2,
         spots_1=spots_1, spots_2=spots_2, 
         exact_grav=exact_grav, verbose=verbose)
    
    return np.array(flux)


        
def pdot_phasefold(times, P, Pdot, t0=0):
    """
    @author: kburdge
       
    Function which returns phases corresponding to timestamps in a lightcurve 
    given a period P, period derivative Pdot, and reference epoch t0
    
    If no reference epoch is supplied, reference epoch is set to earliest time in lightcurve
    
    INPUTS:
            times = time array
            
            P = starting period
            
            Pdot = rate of change of period in units of time/time
            
            t0 = start time
            
    OUTPUTS:
            phases = phases for given time array, period, and Pdot
    
    """
    
    if t0==0:
            times=times-np.min(times)
    else:
            times=times-t0
    
    phases=((times-1/2*Pdot/P*(times)**2) % P)/P
    
    return phases



def pdot_lc(t_obs, mag=None, absmag=True, d=None, Pdot=Pdot, radius_1=r1/a, radius_2=r2/a, sbratio=sbratio, incl=i, 
       light_3 = 0, t_zero = 0, period = P0, a = a, q = m1/m2,  
       f_c = None, f_s = None,
       ldc_1 = None, ldc_2 = None,
       gdc_1 = None, gdc_2 = None,
       didt = None, domdt = None,
       rotfac_1 = 1, rotfac_2 = 1,
       hf_1 = 1.5, hf_2 = 1.5,
       bfac_1 = None, bfac_2 = None,
       heat_1 = None, heat_2 = None, 
       lambda_1 = None, lambda_2 = None,
       vsini_1 = None, vsini_2 = None,
       t_exp=None, n_int=None, 
       grid_1='default', grid_2='default',
       ld_1=None, ld_2=None,
       shape_1='sphere', shape_2='sphere',
       spots_1=None, spots_2=None, 
       exact_grav=False, verbose=1, plot_nopdot=True,savefig=False, **kwargs):
   
    """
    Calculates ellc binary light curve with orbital decay (p dot)
    
    INPUT:
        
        pdot = rate of change of period in units of time/time
            Default: 1e-11 days/days
            
        plot_nopdot = True/False, returns plot with phase folded light curves for 
                      both Pdot=0 and Pdot=Pdot
            Default: True
        
        savefig = True/False, saves .png of plot
            Default: False
            
        mag = absolute magnitude of system (optional)
            Default: None
        
        d = distance to system in pc (optional)
            Default: None
            
        absmag = True/False, determines whether to plot absolute or apparent mag
            Default: True
            
        (see ellc() docstring for documentation of other input params)
        
    OUTPUT:
        
        fluxes/magarr = array for binary light curve, in flux (arb units) or magnitudes if mag value is passed
        
        phases = phase array for phase folded light curve
        
        errors = error array for mag array
        
        plots phase folded light curve(s) in flux/mag
        
    EXAMPLE:
        
        >>> import pdot
        >>> t_obs = pdot.time()
        >>> mag,phase,err = pdot.pdot_lc(t_obs=t_obs,mag=17,ld_1='quad',ldc_1=[0.65,0.2],ld_2='lin',
        ...                ldc_2=0.45,shape_1='poly3p0',shape_2='poly1p5')
    
    """
    
    fluxes = []
    tmods = []
    flux_nopdot = ellc(t_obs=t_obs, radius_1=radius_1, radius_2=radius_2, sbratio=sbratio,
         incl=incl,period=period,q=q,a=a,
         light_3=light_3,t_zero=t_zero,f_c=f_c, f_s=f_s,
         ldc_1=ldc_1, ldc_2=ldc_2,
         gdc_1=gdc_1, gdc_2=gdc_2,
         didt=didt,
         domdt=domdt,
         rotfac_1=rotfac_1, rotfac_2=rotfac_2,
         hf_1=hf_1, hf_2=hf_2,
         bfac_1=bfac_1, bfac_2=bfac_2,
         heat_1=heat_1, heat_2=heat_2, 
         lambda_1=lambda_1, lambda_2=lambda_2,
         vsini_1=vsini_1, vsini_2=vsini_2,
         t_exp=t_exp, n_int=n_int, 
         grid_1=grid_1, grid_2=grid_2,
         ld_1=ld_1, ld_2=ld_2,
         shape_1=shape_1, shape_2=shape_2,
         spots_1=spots_1, spots_2=spots_2, 
         exact_grav=exact_grav, verbose=verbose) 
    

    for ii in range(len(t_obs)):
        P_new = P0 + Pdot*t_obs[ii]
        flux = ellc(t_obs=t_obs, radius_1=radius_1, radius_2=radius_2, sbratio=sbratio,
         incl=incl,period=P_new,q=q,a=a,
         light_3=light_3,t_zero=t_zero,f_c=f_c, f_s=f_s,
         ldc_1=ldc_1, ldc_2=ldc_2,
         gdc_1=gdc_1, gdc_2=gdc_2,
         didt=didt,
         domdt=domdt,
         rotfac_1=rotfac_1, rotfac_2=rotfac_2,
         hf_1=hf_1, hf_2=hf_2,
         bfac_1=bfac_1, bfac_2=bfac_2,
         heat_1=heat_1, heat_2=heat_2, 
         lambda_1=lambda_1, lambda_2=lambda_2,
         vsini_1=vsini_1, vsini_2=vsini_2,
         t_exp=t_exp, n_int=n_int, 
         grid_1=grid_1, grid_2=grid_2,
         ld_1=ld_1, ld_2=ld_2,
         shape_1=shape_1, shape_2=shape_2,
         spots_1=spots_1, spots_2=spots_2, 
         exact_grav=exact_grav, verbose=verbose)  
        tmod = np.mod(t_obs[ii],P_new)
        tmods.append(tmod)
        phot = np.interp(tmod,t_obs,flux,period=P_new)
        fluxes.append(phot)
        
    phases = pdot_phasefold(tmods,P=P0,Pdot=Pdot,t0=0)

    fig = plt.figure()
    
    script = os.path.realpath(__file__)
    magerrdir = os.path.join("/".join(script.split("/")[:-2]),"input")
    gmagerr = os.path.join(magerrdir,'gmagerr.txt')
    rmagerr = os.path.join(magerrdir,'Rmagerr.txt')
    
    if mag is not None:
        gerr = pd.read_csv(gmagerr,sep=' ',names=['Mag','Err'])
        rerr = pd.read_csv(rmagerr,sep=' ',names=['Mag','Err'])
        gmags, gerrs = gerr['Mag'], gerr['Err']
        rmags, rerrs = rerr['Mag'], rerr['Err']
        
        if absmag is True:
            magarr = mag - 2.5*np.log10(fluxes/max(fluxes))
            errors = np.interp(magarr,gmags,gerrs)
            plt.errorbar(phases,magarr,errors,ls='none',c='k',label=r'\.P = '+str(Pdot))
            plt.xlabel('Phase', fontsize=12)
            plt.ylabel('Absolute Mag', fontsize=12)
            plt.xlim(0,1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()
            
        if absmag is False:    
            appmag = 5*np.log10(d) - 5 + mag
            magarr = appmag - 2.5*np.log10(fluxes/max(fluxes))
            errors = np.interp(magarr,gmags,gerrs)
            plt.errorbar(phases,magarr,errors,ls='none',c='k',label=r'\.P = '+str(Pdot))
            plt.xlabel('Phase', fontsize=12)
            plt.ylabel('Apparent Mag', fontsize=12)
            plt.xlim(0,1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()
            
        return np.array(magarr),np.array(phases),np.array(errors)
       
    else:
        errors = np.zeros_like(fluxes)
        if plot_nopdot is True:
            phase_fold(t_obs,flux_nopdot,P0)
            plt.plot(phases,fluxes,'k.',label=r'\.P = '+str(Pdot))
            plt.xlabel('Phase', fontsize=12)
            plt.ylabel('Flux', fontsize=12)
            plt.xlim(0,1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.show()

        if plot_nopdot is False:
            plt.plot(phases,fluxes,'k.',label=r'\.P = '+str(Pdot))
            plt.xlabel('Phase', fontsize=12)
            plt.ylabel('Flux', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend()
            plt.show()
        
        return np.array(fluxes),np.array(phases)
        
    
        
    if savefig is True:
        fig.savefig(str(Pdot)+'_pdotlightcurve.png',dpi=100)