#!/bin/bash

source  /home/ebellm/setup_lsd.sh
shopt -s expand_aliases

lsd-query \
--define "zp=FileTable('/smallfiles/ebellm/LSD/zps.npz', missing='90.0')" \
--define "sys_err=FileTable('/smallfiles/ebellm/LSD/sys_err.npz', missing='90.0')" \
--define "best=FileTable('/smallfiles/ebellm/LSD/current_pids.npz')" \
'SELECT  g.ra as ra, g.dec as dec, \
g.phot_g_mean_mag as Gmag, g.radial_velocity as rv, g.radial_velocity_error as rvErr, g.parallax as parallax, g.rv_nb_transits as ntransits\
FROM gaia_dr2 as g\
WHERE (Gmag < 15) & ((sqrt(rvErr*rvErr-0.11*0.11)*sqrt(ntransits*0.636619772))>70)  & (parallax > 0)' > gaia_large_rv.dat
