#include <stdio.h>
#include <stdlib.h>

#define NMAX 10000

/* C implementation of analysis of variance.
 * 
 * P. Mroz @ Caltech, 6 Mar 2020
 */

double AOV (double freq, double *t, double *m, double avg, int npts, int r) 
{    
    int i,idx,n[r];
    double aux,s1,s2,F,phase[NMAX],sum1[r],sum2[r];
    
    /* calculate orbital phase */
    for (i=0; i<npts; i++)
    {
        aux = (t[i]-t[0])*freq;
        phase[i] = aux-(int)aux;
    }
    // calculate mean and variance in each phase bin [ 0 .. r-1 ]
    for (i=0; i<r; i++) 
    {
        n[i] = 0;
        sum1[i] = 0.0;
        sum2[i] = 0.0;
    }
    for (i=0; i<npts; i++) 
    {
        idx = (int)(phase[i]*r);
        sum1[idx] += m[i];
        sum2[idx] += m[i]*m[i];
        n[idx] += 1;
    }
    s1 = 0.0; s2 = 0.0;
    for (i=0; i<r; i++) {
        if (n[i] == 0) continue;
        sum1[i] /= (double)n[i]; // mean in each phase bin
        s1 += n[i]*(sum1[i]-avg)*(sum1[i]-avg);
        s2 += sum2[i]-n[i]*sum1[i]*sum1[i];
    }
    F = s1/s2;
    F *= (double)(npts-r);
    F /= (double)(r-1);
    
    return F;
}

int main (int argc, char *argv[]) 
{
    FILE *fp;
    double hjd[NMAX],mag[NMAX],err[NMAX],freq_min,freq_max,delta_freq,freq;
    double power,power_max,freq_best,t,m,e,mean_mag;
    int npts,n_freq,i,n_rep,t_int,t_int_old;
    
    if (argc != 4)
    {
        fprintf(stderr,"Usage: AOV filename P_min P_max\n");
        return 1;
    }
    
    fp = fopen(argv[1],"r");
    if (fp == NULL) 
    {
        fprintf(stderr,"Error while opening file %s\n",argv[1]);
        return 1;
    }
    
    npts = 0;
    t_int_old = 0;
    n_rep = 0;
    while (fscanf(fp,"%lf %lf %lf",&t,&m,&e) != EOF) 
    {
        t_int = (int)t;
        if (t_int == t_int_old) n_rep += 1;
        else 
        {
            n_rep = 1;
            t_int_old = t_int;
        }
        if (n_rep > 3) continue;
        hjd[npts] = t;
        mag[npts] = m;
        err[npts] = e;
        npts += 1;
    }
    fclose(fp);
    
    if (npts < 30) {
        fprintf(stderr,"Too little data in %s. Exiting\n",argv[1]);
        return 1;
    }
    
    freq_min = 1.0/(atof(argv[3]));
    freq_max = 1.0/(atof(argv[2]));
    delta_freq = 0.1/(hjd[npts-1]-hjd[0]);
    n_freq = (int)((freq_max-freq_min)/delta_freq);
    
    mean_mag = 0.0;
    for (i=0; i<npts; i++) mean_mag += mag[i];
    mean_mag /= (double)npts;
    
    power_max = 0.0;
    freq = freq_min;
    for (i=0; i<n_freq; i++) {
        power = AOV(freq,hjd,mag,mean_mag,npts,10);
        if (power > power_max) {
            power_max = power;
            freq_best = freq;
        }
        freq += delta_freq;
    }
    
    printf("%-35s %10.6f %8.3f\n",argv[1],1.0/freq_best,power_max);
    
    return 0;
    
    
}
