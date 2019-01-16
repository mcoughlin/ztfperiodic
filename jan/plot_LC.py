import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse
import sys
import os





def search_and_plot(filename,p=0,Nmin=40,Nharm=5):

    lc = np.loadtxt(filename)
    if np.shape(np.atleast_2d(lc))[0]<Nmin:
	    return 1

    m = lc[:,2] < 0.06
    lc = lc[m]
    print(np.sum(~m))

    
    if p==0:
	    from gatspy import periodic
	    model = periodic.LombScargleMultibandFast(Nterms=Nharm)
	    baseline = np.max(lc[:,0])-np.min(lc[:,0])
	    model.optimizer.period_range=(30./60/24, np.min(np.array([5.,0.9*baseline])) )
	    model.fit(lc[:,0], lc[:,1], lc[:,2], lc[:,3]);
	    p = model.best_period

    plt.figure(figsize=(8,6))
    for n in range(1,4):
	    m = lc[:,3]==n
	    plt.title('%f'%p)
	    if p:
		    plt.errorbar(lc[m,0]/p%1,lc[m,1],lc[m,2],fmt='.',c=colors[n],elinewidth=0.2)
		    plt.errorbar(lc[m,0]/p%1+1,lc[m,1],lc[m,2],fmt='.',c=colors[n],elinewidth=0.2)
	    else:
		    plt.errorbar(lc[m,0],lc[m,1],lc[m,2],fmt='.',c=colors[n],elinewidth=0.2)

    plt.ylim(plt.ylim()[::-1])
    plt.savefig('%s_folded.png' %os.path.splitext(filename)[0])
    plt.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot lightcurve')
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--period','-p',type=float,default=0)
    args = parser.parse_args()
    p = args.period

    colors = {1:'g',2:'r',3:'purple'}

    args.files.sort()
    for f in args.files:
        print(f)
        try:
    		search_and_plot(f,p=0)
        except Exception as e:
            print(e)
        

