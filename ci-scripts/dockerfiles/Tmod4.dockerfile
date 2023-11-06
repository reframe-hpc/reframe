FROM ubuntu:20.04

ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV _TMOD_VER=4.6.0

# Setup apt
RUN \
  apt-get -y update && \
  apt-get -y install ca-certificates && \
  update-ca-certificates

# Required utilities
RUN apt-get -y install wget less

# Install Tmod4
RUN \
  apt-get -y install autoconf tcl-dev && \
  wget -q https://github.com/cea-hpc/modules/archive/v${_TMOD_VER}.tar.gz -O tmod.tar.gz && \
  tar xzf tmod.tar.gz && \
  cd modules-${_TMOD_VER} && \
  ./configure && make install && \
  cd .. && rm -rf tmod.tar.gz modules-${_TMOD_VER} && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV BASH_ENV=/usr/local/Modules/init/profile.sh
