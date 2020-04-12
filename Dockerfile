#FROM debian:stable-slim
FROM docker.io/nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

RUN apt-get update && \
    apt-get -y install --no-install-recommends openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install \
    cython3 \
    gfortran \
    git \
    python3-astroplan \
    python3-astropy \
    python3-astroquery \
    python3-dateutil \
    python3-f2py \
    python3-future \
    python3-healpy \
    python3-h5py \
    python3-matplotlib \
    python3-numpy \
    python3-pandas \
    python3-pip \
    python3-pyvo \
    python3-scipy \
    python3-seaborn \
    python3-tqdm \
    rsync && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64/stubs/:${LD_LIBRARY_PATH}$"

# Install requirements. Do this before installing our own package, because
# presumably the requirements change less frequently than our own code.
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r \
    /requirements.txt \
    git+https://github.com/dmitryduev/broker.git \
    git+https://github.com/mikekatz04/gce.git \
    git+https://github.com/mcoughlin/cuvarbase.git
RUN rm /requirements.txt

COPY . /src
RUN pip3 install --no-cache-dir /src
RUN cd /src/ztfperiodic/pyaov && f2py3 -m aov -c aovconst.f90 aovsub.f90 aov.f90 && cp aov.cpython-36m-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/

RUN useradd -mr ztfperiodic
USER ztfperiodic:ztfperiodic
WORKDIR /home/ztfperiodic

COPY id_rsa /home/ztfperiodic/.ssh/id_rsa
COPY docker/etc/ssh/ssh_known_hosts /home/ztfperiodic/.ssh/known_hosts

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64"

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["python3","/src/bin/ztfperiodic_period_search.py"]
