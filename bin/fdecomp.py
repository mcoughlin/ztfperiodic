import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit



def convert_fourier_components(arr):
    """ convert an array of fourier components (A,B) to amp,phase and normalise
    """

    # convert A,B to amp and phi
    for n in np.arange(0,np.size(arr),2):
        amp = np.sqrt(arr[n]**2 + arr[n+1]**2)
        phi = np.arctan2(arr[n],arr[n+1])
        arr[n] = amp
        arr[n+1] = phi

    # normalise
    arr[1::2] -= arr[1] # remove phase 0
    arr[1::2] = arr[1::2]%(2*np.pi) # [0,2pi>
    arr[2::2] /= arr[0] # normalise amplitudes

    return arr



def make_f(p):
    """ Function that returns a fourier function for period p
    input 
        p: period

    output:
        f: function
    
    """
    def f(t, *pars):
        """A function which returns a fourier model, inluding offset and slope
        input:
            t: array with times
            pars: list of parameters: [offset, slope, a_1,b_1,a_2,b_2,...]    
    
        """

        # a offset a[0], and slope
        y = pars[0] + pars[1] * (t-np.min(t))
        # fourier components, loops from 1 to ?
        for n in np.arange(1,(len(pars)-2)/2+1,1):
            phi = 2*np.pi*t/p
            y += pars[n*2] * np.cos(n * phi)
            y += pars[n*2+1] * np.sin(n * phi)
        return y
    return f




def fit_best(LC,p,maxNterms=5,output='compact',plotname=False):
    """ fit a lightcurve with a fourier model given period p
    input
        LC: 2D-array; [t,y,dy]
        p: float; the period

    output:
        array; model values for times t
    """

    t = LC[:,0]
    y = LC[:,1]
    dy = LC[:,2]
    N = np.size(t)

    # 
    f = make_f(p=p)
    chi2 = np.zeros(maxNterms+1,dtype=float) # fill in later
    Fstats = np.zeros(maxNterms+1,dtype=float) # fill in later
    pars = np.zeros((maxNterms+1,(maxNterms+1)*2)) # fill in later

    
    init = [16.0,0.001] # the initial values for the minimiser
    for i,Nterms in enumerate(np.arange(maxNterms+1.,dtype=int)):
        # fit using scipy curvefit
        popt, pcov = curve_fit(make_f(p), # function
            t, y, # t,dy
            init, # initial values [1.0]*2
            sigma=dy # dy
            )

        init = init + [0.05]*2

        pars[i,:2*(i+1)] = popt
        # make the model
        model = f(t, *popt)
        chi2[i] = np.sum(((y-model)/dy)**2)

    #print pars
    # calc BICs
    BIC = chi2 + np.log(N)* (2+2*np.arange(maxNterms+1,dtype=float))
    plt.plot(np.arange(maxNterms+1),BIC)
    plt.show()
    best = np.argmin(BIC)
    
    power = (chi2[0]-chi2[best])/chi2[0]
    bestBIC = BIC[best]
    bestpars = pars[best,:]

    if output == 'compact':
        bestpars[2:2+2*best] = convert_fourier_components(bestpars[2:2+2*best])

    return power,bestBIC,bestpars




def test_EB(p=1.0,Nterms=3):
    """ using ellc, make an eclipsing binary LC and fit using fourier components

    input:
    p: float the period
    Nterms number of fourier terms to be used for fitting

    output:
        0

    """

    import ellc    
    
    # make testdata
    t = 100.*np.random.rand(1000)
    t.sort()
    y = ellc.lc(t,0.05,0.25,0.12,85.,t_zero=0.32321,period=p,q=0.23,
                shape_2='roche')
    y += 0.001*(t-np.min(t))
    dy = 0.01*np.ones_like(t)
    y += dy*np.random.randn(np.size(t))

    # fit data
    LC = np.c_[t,y,dy]
    power,BIC,pars = fit_best(LC,p,maxNterms=5,output='compact',plotname='test.pdf')

    #print power,BIC,pars

    return 0

