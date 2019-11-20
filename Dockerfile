FROM debian:stable-slim

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
    python3-future \
    python3-healpy \
    python3-h5py \
    python3-matplotlib \
    python3-numpy \
    python3-pandas \
    python3-pip \
    python3-scipy \
    python3-seaborn \
    python3-tqdm \
    python3-pyvo \
    rsync && \
    rm -rf /var/lib/apt/lists/*

# Install requirements. Do this before installing our own package, because
# presumably the requirements change less frequently than our own code.
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r \
    /requirements.txt \
    git+https://github.com/dmitryduev/broker.git
    #git+https://github.com/mikekatz04/gce.git
RUN rm /requirements.txt

COPY . /src
RUN pip3 install --no-cache-dir /src

#COPY docker/etc/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts

RUN useradd -mr ztfperiodic
USER ztfperiodic:ztfperiodic
WORKDIR /home/ztfperiodic

COPY id_rsa /home/ztfperiodic/.ssh/id_rsa
COPY docker/etc/ssh/ssh_known_hosts /home/ztfperiodic/.ssh/known_hosts

ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["python3","ztfperiodic_period_search.py"]
