FROM ubuntu:20.04

ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV _LMOD_VER=8.4.12

# ReFrame requirements
RUN \
  apt-get -y update && \
  apt-get -y install ca-certificates && \
  update-ca-certificates && \
  apt-get -y install gcc make git python3 python3-pip python3-venv

# Required utilities
RUN apt-get -y install wget

# Install Lmod
RUN \
  apt-get -y install lua5.3 lua-bit32:amd64 lua-posix:amd64 lua-posix-dev liblua5.3-0:amd64 liblua5.3-dev:amd64 tcl tcl-dev tcl8.6 tcl8.6-dev:amd64 libtcl8.6:amd64 && \
  wget -q https://github.com/TACC/Lmod/archive/${_LMOD_VER}.tar.gz -O lmod.tar.gz && \
  tar xzf lmod.tar.gz && \
  cd Lmod-${_LMOD_VER} && \
  ./configure && make install && \
  cd .. && rm -rf lmod.tar.gz Lmod-${_LMOD_VER} && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV BASH_ENV=/usr/local/lmod/lmod/init/profile
