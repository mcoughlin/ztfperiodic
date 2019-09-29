# ztfperiodic
Scripts for analyzing ZTF periodic data

Reminder: penquins required
pip install git+https://github.com/dmitryduev/broker.git

## Period Search

This package implements two period-searching algorithms: Gneralized Conditional Entropy (GCE) and Analysis Of Variance (AOV).

- For GCE: ztfperiodic uses code written by Michael Katz (in cuda)</br>
Download: `git clone https://github.com/mikekatz04/gce`</br>
Installation: `python setup.py install`<br>
[TBD]: not every computer is CUDA-capable.

- For AOV: </br>
`source build.sh` in ztfperiodic/pyaov/
copy the .so file to lib/python3.7/site-packages/ or equivalent 
