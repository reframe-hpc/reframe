# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# LMod versions prior to 8.2 emitted Python commands differently, so we use this
# Dockerfile to test the bindings of older versions
#


FROM ubuntu:20.04

ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV _LMOD_VER=7.7

# Setup apt
RUN \
  apt-get -y update && \
  apt-get -y install ca-certificates && \
  update-ca-certificates

# Required utilities
RUN apt-get -y install wget

# Install Lmod
RUN \
  apt-get -y install lua5.3 lua-bit32:amd64 lua-posix:amd64 lua-posix-dev liblua5.3-0:amd64 liblua5.3-dev:amd64 tcl tcl-dev tcl8.6 tcl8.6-dev:amd64 libtcl8.6:amd64 lua-filesystem:amd64 lua-filesystem-dev:amd64 && \
  wget -q https://github.com/TACC/Lmod/archive/${_LMOD_VER}.tar.gz -O lmod.tar.gz && \
  tar xzf lmod.tar.gz && \
  cd Lmod-${_LMOD_VER} && \
  ./configure && make install && \
  cd .. && rm -rf lmod.tar.gz Lmod-${_LMOD_VER} && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV BASH_ENV=/usr/local/lmod/lmod/init/profile
