import pandas as pd
import tables
import numpy as np
import glob
import matplotlib.pyplot as plt
from cuvarbase.ce import ConditionalEntropyAsyncProcess
import numpy as np
import sys
lightcurves=[]
p=0
minlength=99
def periodicity(f):
	if 'zr' in f:
		fil='r'
	elif 'zg' in f:
		fil='g'
	else:
		fil='i'
	lightcurves=[]
	baseline=0
	p=0
	print(f)
	coordinates=[]
	with tables.open_file(f) as store:
		for tbl in store.walk_nodes("/", "Table"):
			if tbl.name in ["sourcedata", "transientdata"]:
				group = tbl._v_parent
				break
		srcdata = pd.DataFrame.from_records(store.root.matches.sourcedata.read_where('programid > 1'))
		srcdata.sort_values('matchid', axis=0, inplace=True)
		exposures = pd.DataFrame.from_records(store.root.matches.exposures.read_where('programid > 1'))
		merged = srcdata.merge(exposures, on="expid")
		if len(merged.matchid.unique()) == 0:
			return

		matchids = np.array(merged.matchid)
		values, indices, counts = np.unique(matchids, return_counts=True,return_inverse=True)       
		idx = np.where(counts>minlength)[0]

		if len(idx) == 0:
			return
		for k in merged.matchid.unique():
			df = merged[merged['matchid'] == k]
			RA = df.ra
			Dec = df.dec
			x = df.psfmag
			err=df.psfmagerr
			obsHJD = df.hjd				
			if len(x)>minlength:
				newbaseline = max(obsHJD)-min(obsHJD)
				if newbaseline>baseline:
					baseline=newbaseline
				coordinate=(RA.values[0],Dec.values[0])
				coordinates.append(coordinate)
				lightcurve=(obsHJD.values,x.values,err.values)
				lightcurves.append(lightcurve)
				print(p)
				p=p+1
	if baseline<10:
		proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=20, mag_bins=10, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
		fmin, fmax = 18, 1440
		samples_per_peak = 10
		df = 1./(samples_per_peak * baseline)
		nf = int(np.ceil((fmax - fmin) / df))
		freqs = fmin + df * np.arange(nf)
		print(f)
		results = proc.batched_run_const_nfreq(lightcurves, batch_size=100, freqs = freqs, only_keep_best_freq=True,show_progress=True)
		finalresults=(([x[0] for x in coordinates]),([x[1] for x in coordinates]),([x[0] for x in results]),([x[2] for x in results]))
		np.concatenate(finalresults)
		finalresults=np.transpose(finalresults)
		#with open("/media/Data/kburdge/ZTFNew/dataC.txt",'ab') as f:
		#	np.savetxt(f, finalresults, delimiter=',')
		k=0
		while k<len(lightcurves):
			out  = results[k]
			period = 1./out[0]
			significance=out[2]
			if significance>6:
				phases=[]
				for i in lightcurves[k][0]:
					y=float(i)
					phases.append(np.remainder(y,2*period)/2*period)
	
				magnitude=lightcurves[k][1]
				err=lightcurves[k][2]
				RA=coordinates[k][0]
				Dec=coordinates[k][1]
	
				fig = plt.figure(figsize=(10,10))
				plt.gca().invert_yaxis()
				ax=fig.add_subplot(1,1,1)
				ax.errorbar(phases, magnitude,err,ls='none',c='k')
				period2=period
				ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))
	
				if period < 0.002777778:
					plt.close()
				elif period > 0.002777778 and period < 0.003472222:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/4min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.003472222 and period < 0.004166667:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/5min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.004166667 and period < 0.004861111:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/6min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.004861111 and period < 0.006944444:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/7_10min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.006944444 and period < 0.020833333:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/10_30min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.020833333 and period < 0.041666667:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/30_60min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.041666667 and period < 0.083333333:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/1_2hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.083333333 and period < 0.166666667:
    					fig.savefig('CECaltech/2_4hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.166666667 and period < 0.5:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/4_12hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.5 and period < 3:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/12_72hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 3 and period < 10:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/3_10days/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 10 and period < 50:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/10_50days/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 50:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CEHC/50_baseline/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
			plt.close()
			k=k+1
	else:
		proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=20, mag_bins=10, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
		fmin, fmax = 2/baseline, 480
		samples_per_peak = 10
		df = 1./(samples_per_peak * baseline)
		nf = int(np.ceil((fmax - fmin) / df))
		freqs = fmin + df * np.arange(nf)
		print(f)
		results = proc.batched_run_const_nfreq(lightcurves, batch_size=5, freqs = freqs, only_keep_best_freq=True,show_progress=True)
		finalresults=(([x[0] for x in coordinates]),([x[1] for x in coordinates]),([x[0] for x in results]),([x[2] for x in results]))
		np.concatenate(finalresults)
		finalresults=np.transpose(finalresults)
		#with open("/media/Data/kburdge/ZTFNew/dataC.txt",'ab') as f:
		#	np.savetxt(f, finalresults, delimiter=',')
		k=0
		while k<len(lightcurves):
			out  = results[k]
			period = 1./out[0]
			significance=out[2]
			if significance>15:
				phases=[]
				for i in lightcurves[k][0]:
					y=float(i)
					phases.append(np.remainder(y,2*period)/2*period)

				magnitude=lightcurves[k][1]
				err=lightcurves[k][2]
				RA=coordinates[k][0]
				Dec=coordinates[k][1]
	
				fig = plt.figure(figsize=(10,10))
				plt.gca().invert_yaxis()
				ax=fig.add_subplot(1,1,1)
				ax.errorbar(phases, magnitude,err,ls='none',c='k')
				period2=period
				ax.set_title(str(period2)+"_"+str(RA)+"_"+str(Dec))

				if period < 0.002777778:
					plt.close()
				elif period > 0.002777778 and period < 0.003472222:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/4min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.003472222 and period < 0.004166667:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/5min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.004166667 and period < 0.004861111:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/6min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.004861111 and period < 0.006944444:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/7_10min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.006944444 and period < 0.020833333:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/10_30min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.020833333 and period < 0.041666667:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/30_60min/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.041666667 and period < 0.083333333:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/1_2hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.083333333 and period < 0.166666667:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/2_4hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.166666667 and period < 0.5:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/4_12hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 0.5 and period < 3:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/12_72hours/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 3 and period < 10:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/3_10days/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 10 and period < 50:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/10_50days/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
				elif period > 50:
    					fig.savefig('/media/Data/ZTFPeriodic/NewSearch/CE/50_baseline/'+str(significance)+'_'+str(RA)+'_'+str(Dec)+'_'+str(period)+'_'+fil+'.png')
					plt.close()
			plt.close()
			k=k+1

periodicity(sys.argv[1])
