
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import copy


def AB2AmpPhi(arr):
    """ convert an array of fourier components (A,B) to amp,phase and normalise
    WARNING; this function changes the variables inplace

    input:
        arr : array of fourier components, A&B
    
    output:
        arr : array of fourier amplitudes and phases (normalised to the first)

    """

    # convert A,B to amp and phi
    for n in np.arange(0,np.size(arr),2):
        amp = np.sqrt(arr[n]**2 + arr[n+1]**2)
        phi = np.arctan2(arr[n],arr[n+1])
        arr[n] = amp
        arr[n+1] = phi

    # normalise
    arr[2::2] /= arr[0] # normalise amplitudes
    arr[3::2] -= arr[1] # remove phase 0
    arr[3::2] = arr[3::2]%(2*np.pi) # [0,2pi>

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
            y += pars[int(n*2)] * np.cos(n * phi)
            y += pars[int(n*2+1)] * np.sin(n * phi)
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

    if N <= 4+maxNterms*2:
        return np.nan*np.ones(4+maxNterms*2)

    
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

    # calc BICs
    BIC = chi2 + np.log(N)* (2+2*np.arange(maxNterms+1,dtype=float))
    best = np.argmin(BIC)
    
    power = (chi2[0]-chi2[best])/chi2[0]
    bestBIC = BIC[best]
    bestpars = pars[best,:]

    if plotname:
        for n in [0,1]:
            plt.errorbar(t/p%1+n,y,dy,fmt='k.')
            plt.plot(t/p%1+n,f(t,*bestpars),'r.')
        plt.ylim(plt.ylim()[::-1])
        plt.savefig(plotname)
        plt.close()

    if output == 'compact':
        # convert to amplitude and phase
        if best > 0:
            bestpars[2:2+2*best] = AB2AmpPhi(bestpars[2:2+2*best])

    return np.r_[power,bestBIC,bestpars]



def fit(LC,p,Nterms=3,output='compact',plotname=False):
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

    # fit using scipy curvefit
    f = make_f(p=p)
    popt, pcov = curve_fit(f, # function
        t, y, # t,dy
        [1.0] * (2+2*Nterms), # initial values
        sigma=dy # dy
        )

    model = f(t, *popt)

    # calc BICs
    print(np.r_[popt[:2], np.zeros(Nterms*2)])
    m0 = f(t, *np.r_[popt[:2], np.zeros(Nterms*2)])
    chi2_0 = np.sum(((y-m0)/dy)**2)
    chi2 = np.sum(((y-model)/dy)**2)
    BIC = chi2 + np.log(N)*(2+2*Nterms)
    power = (chi2_0-chi2)/chi2_0

    if output == 'compact':
        # convert to amplitude and phase
        bestpars[2:2+2*Nterms] = AB2AmpPhi(bestpars[2:2+2*Nterms])

    return np.r_[power,BIC,popt]



def test_EB(p=1.0,maxNterms=10):
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
    y += 0.0000001*(t-np.min(t))
    dy = 0.01*np.ones_like(t)
    y += dy*np.random.randn(np.size(t))

    # fit data
    LC = np.c_[t,y,dy]
    output1 = fit_best(LC,p,maxNterms=maxNterms,output='full')
    output2 = fit(LC,p,Nterms=5,output='full')

    plt.plot(AB2AmpPhi(copy.deepcopy(output1[4:]).reshape(maxNterms,2)),'ro')
    plt.plot(AB2AmpPhi(copy.deepcopy(output2[4:]).reshape(Nterms,2)),'bo')
    plt.show()

    # show result
    f = make_f(p=p)
    model1 = f(t, *output1[2:])
    model2 = f(t, *output2[2:])

    #plt.errorbar(t,y,dy,fmt='k,')
    #plt.plot(t,model,'r.')
    #plt.show()

    # fold 
    plt.errorbar(t/p%1,y,dy,fmt='k,')
    plt.plot(t/p%1,model1,'r.')
    plt.plot(t/p%1,model2,'b.')
    plt.show()
        

    return 0



################################################################################
# OLD CODE, need to cleanup or remove
################################################################################



def test(p=1.23432,Nterms=3,output='compact'):
    """ make a test LC and fit using fourier components

    input:
    p: float the period
    Nterms: number of fourier terms to be used for fitting

    output:
        0stats

    """


    # make testdata
    t = 100*np.random.rand(100)
    t.sort()
    x0 = np.array([16.85,0.00001, 0.3,0.05,0.02,0.01,0.03,0.06])
    f = make_f(p=p)
    y = f(t,*x0) + 1.0
    dy = 0.03*np.ones_like(t)
    y += dy*np.random.randn(np.size(t))
    LC = np.c_[t,y,dy]

    maxNterms = 5
    res = fit_best(LC,p,maxNterms=maxNterms,output='full',plotname=False)

    print(res)
    #plt.errorbar(LC[:,0]/p%1,LC[:,1],LC[:,2],fmt='k.')
    #plt.show()


    return 0







