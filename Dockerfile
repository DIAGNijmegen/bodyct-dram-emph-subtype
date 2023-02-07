ARG UBUNTU_VERSION=20.04
ARG CUDA_MAJOR_VERSION=11.3.1
ARG CUDNN_MAJOR_VERSION=8

ARG BUILD_JOBS=32

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# === Propagate build args ===
ARG BUILD_JOBS

# === Install build packages ===
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
		cmake \
		gcc \
		git \
        ca-certificates \
        libatlas-base-dev \
        libblas-dev \
        libopenblas-dev \
        liblapack-dev \
        openjdk-8-jdk \
		g++ libgomp1 \
        python-dev \
        python3-pip \
        python3.8 \
        python3.8-dev \
        htop \
        bzip2 \
        zip unzip \
        libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libbz2-dev \
		libdb5.3-dev libexpat1-dev liblzma-dev tk-dev \
        libffi-dev \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*



RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2 && \
    update-alternatives --config python
    
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --config python3
   
RUN python -m pip install --upgrade --force pip

RUN mkdir /root/python-packages

# === Install python libraries ===
COPY install_files/requirements.in /root/python-packages/


RUN cd /root/python-packages && \
    pip install -r requirements.in --verbose --find-links https://download.pytorch.org/whl/torch_stable.html  --find-links https://data.dgl.ai/wheels/repo.html && \
    rm -rf ~/.cache/pip*


# === Set some environment variables and options. ===
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm && gpasswd -a algorithm sudo

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN ldconfig

USER algorithm
WORKDIR /opt/algorithm
COPY . /opt/algorithm/
RUN sudo chmod +x /opt/algorithm/run.sh
ENTRYPOINT ["/opt/algorithm/run.sh"]


# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=emphysema_subtyping
LABEL nl.diagnijmegen.rse.algorithm.author="Weiyi Xie"
LABEL nl.diagnijmegen.rse.algorithm.ticket=10039

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=32G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=8G

