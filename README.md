# ztfperiodic
Scripts for analyzing ZTF periodic data

Reminder: penquins required
pip install git+https://github.com/dmitryduev/broker.git

## Period Search

This package implements several period-searching algorithms. Below we explain some of them.

- Generalized Conditional Entropy (GCE) </br>
ztfperiodic uses code written by Michael Katz (in cuda); Algorithm paper: [Graham et al. (2013)](https://academic.oup.com/mnras/article/434/3/2629/1044188)</br>
Download: `git clone https://github.com/mikekatz04/gce`</br>
Installation: `python setup.py install` (you need a CUDA-capable machine.)<br>

- Analysis Of Variance (AOV) </br>
Code originally from [this website](http://users.camk.edu.pl/alex/soft/aovgui.tgz); Algorithm paper: [Schwarzenberg-Czerny (1996)](https://ui.adsabs.harvard.edu/abs/1996ApJ...460L.107S/abstract).</br>
First, run `source build.sh` in the **ztfperiodic/pyaov/** directory, then copy the `aov.cpython-36m-darwin.so` file to **lib/python3.7/site-packages/** or equivalent.

- Lomb-Scargle (LS)

- Box-fitting Least Squares (BLS)

- Phase Dispersion Minimization (PDM, PDM2)<br>
Algorithm papers: [Stellingwerf (1978)](https://ui.adsabs.harvard.edu/abs/1978ApJ...224..953S/abstract)

- Fast Fourier Transform (FFT)

Consult [this paper](https://arxiv.org/abs/1307.2209) if you're not sure which algorithm to adopt.

## Example 

Below we list several examples to run scripts in the **/bin** directory.

`python ztfperiodic_object_followup.py --doGPU --ra your_ra --declination your_dec --doSpectra --user your_kowalski_username --pwd your_kowalski_password`

## Docker 

If copying to schoty, copy a netrc that can talk to schoty into the current directory, otherwise touch netrc should do. 

nvidia-docker build -t python-ztfperiodic .
nvidia-docker run -it python-ztfperiodic

For debugging:
docker run -it python-ztfperiodic    
