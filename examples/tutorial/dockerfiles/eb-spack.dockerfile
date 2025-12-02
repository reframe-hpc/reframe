#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:8.4.12

ENV _SPACK_VER=0.16
ENV _EB_VER=4.4.1

RUN apt-get -y update && \
    apt-get -y install curl && \
    apt-get -y install sudo && \
    apt-get -y install python3-pip && \
    apt-get -y install gcc git jq libomp-dev tree vim

# Install reframe
ARG REFRAME_TAG=develop
WORKDIR /usr/local/share
RUN git clone --depth 1 --branch $REFRAME_TAG https://github.com/reframe-hpc/reframe.git && \
    cd reframe/ && ./bootstrap.sh
ENV PATH=/usr/local/share/reframe/bin:$PATH

# Install EasyBuild
RUN pip3 install easybuild==${_EB_VER}

# Add tutorial user
RUN useradd -ms /bin/bash -G sudo user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user
WORKDIR /home/user

# Install Spack
RUN mkdir .local && cd .local && \
    git clone --branch releases/v${_SPACK_VER} --depth 1 https://github.com/spack/spack

RUN echo '. /usr/local/lmod/lmod/init/profile && . /home/user/.local/spack/share/spack/setup-env.sh' > /home/user/.profile
ENV BASH_ENV=/home/user/.profile
