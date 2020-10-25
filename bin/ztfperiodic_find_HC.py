import os
from astropy import time
import astropy.units as u
from astropy.table import Table, unique
from astropy.coordinates import SkyCoord
import numpy as np
import pyvo.dal
import requests

client = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP',)

def ztf_references():
    refstable = client.search("""
    SELECT field, fid, maglimit FROM ztf.ztf_current_meta_ref
    WHERE (nframes >= 15) AND (startobsdate >= '2018-02-05T00:00:00Z')
    AND (field < 2000)
    """).to_table()

    refs = refstable.group_by(['field', 'fid']).groups.aggregate(np.mean)
    refs = refs.filled()

    refs_grouped_by_field = refs.group_by('field').groups
    for field_id, rows in zip(refs_grouped_by_field.keys,
                              refs_grouped_by_field):
        print(field_id, rows)

def ztf_obs(field_id, start_time=None, end_time=None):

    if start_time is None:
        start_time = time.Time('2018-02-05T00:00:00', format='isot')
    if end_time is None:
        end_time = time.Time.now()

    obstable = client.search("""
    SELECT field,rcid,fid,expid,obsjd,exptime,seeing,airmass,maglimit,ipac_gid
    FROM ztf.ztf_current_meta_sci WHERE (obsjd BETWEEN {0} AND {1})
    AND (field = {2})
    """.format(start_time.jd, end_time.jd, field_id)).to_table()

    obstable = obstable.filled()
    if len(obstable) > 0:
        obstable = unique(obstable,keys="obsjd")

    return obstable

field_ids = np.arange(245,880)
#field_ids = np.arange(487,490)
filename = '../input/nobs.dat'

tesspath = '../input/%s.tess' % 'ZTF'
fields = np.recfromtxt(tesspath, usecols=range(3),
                       names=['field_id', 'ra', 'dec'])

if not os.path.isfile(filename):
    outfile = open('../input/nobs.dat','w')
    for field_id in field_ids:
        obstable = ztf_obs(field_id)
        idx = np.where(fields['field_id'] == field_id)[0]
        ra, dec = fields['ra'][idx], fields['dec'][idx]

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        gal = coord.galactic
        b = gal.b.deg

        idxp1 = np.where(obstable["ipac_gid"] == 1)[0]
        idxp2 = np.where(obstable["ipac_gid"] == 2)[0]
        idxp3 = np.where(obstable["ipac_gid"] == 3)[0]
    
        jd = np.array(obstable["obsjd"])
        fid = np.array(obstable["fid"])
        idx = np.argsort(jd)
        jd, fid = jd[idx], fid[idx]
   
        idx = []
        for ii, t in enumerate(jd):
            if ii == 0:
                idx.append(ii)
            else:
                dt = jd[ii] - jd[idx[-1]]
                if dt >= 30.0*60.0/86400.0:
                    idx.append(ii)
        idx = np.array(idx)
        jd1, fid1 = jd[idx], fid[idx]
        idxg1 = np.where(fid1 == 1)[0]
        idxr1 = np.where(fid1 == 2)[0]
        idxi1 = np.where(fid1 == 3)[0]
 
        idx = np.setdiff1d(np.arange(len(jd)), idx)

        jd2, fid2 = jd[idx], fid[idx]
        data_out = {}
        for jj in [1,2,3]:
            idx2 = np.where(fid2 == jj)[0]
            if len(idx2) < 2:
                data_out[jj] = []
                continue

            jj_slice = jd2[idx2]
            bins = np.arange(np.floor(np.min(jj_slice)),
                             np.ceil(np.max(jj_slice)))
            hist, bin_edges = np.histogram(jj_slice, bins=bins)
            bins = (bin_edges[1:] + bin_edges[:-1])/2.0

            if len(hist) == 0:
                data_out[jj] = []
                continue

            idx3 = np.argmax(hist)
            bin_start, bin_end = bin_edges[idx3], bin_edges[idx3+1]                
            idx_night = np.where((jj_slice >= bin_start) & (jj_slice <= bin_end))[0]
            data_out[jj] = jj_slice[idx_night]

        print('%d %.5f %.5f %.5f %d %d %d %d %d %d %d %d %d'%(field_id, ra, dec, b, len(idxp1), len(idxp2), len(idxp3), len(idxg1), len(idxr1), len(idxi1), len(data_out[1]), len(data_out[2]), len(data_out[3])), file=outfile, flush=True)
    outfile.close()

data_out = np.loadtxt(filename)
idxg, idxr = np.argsort(data_out[:,4])[::-1], np.argsort(data_out[:,5])[::-1]

for ii in idxg[:10]:
    print('peak g-band: %d %d' % (data_out[ii,0], data_out[ii,4]))

for ii in idxr[:10]:
    print('peak r-band: %d %d' % (data_out[ii,0], data_out[ii,5]))
