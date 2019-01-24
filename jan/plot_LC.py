import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse
import sys
import os
import joblib





def search_and_plot(filename,p=0,Nmin=40,Nharm=5,filts=[1,2,3]):
    try:
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
	        model.optimizer.period_range=(10./60/24, np.min(np.array([10.,0.9*baseline])) )
                mask = np.isin(lc[:,3],filts)
	        model.fit(lc[mask,0], lc[mask,1], lc[mask,2], lc[mask,3]);
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
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot lightcurve')
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--period','-p',type=float,default=0)
    parser.add_argument('--Njobs',type=int,default=1)
    args = parser.parse_args()
    p = args.period

    colors = {1:'g',2:'r',3:'purple'}
    filts = [1,]

    args.files.sort()
    """
    for f in args.files:
        print(f)

	print('periodfinding with ', filts)
        try:
    		search_and_plot(f,p=0,filts=filts,Nharm=3)
        except Exception as e:
            print(e)
    """
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.Njobs)(delayed(search_and_plot)(f,p=0,filts=filts,Nharm=3) for f in args.files)

 
