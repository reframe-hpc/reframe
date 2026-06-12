# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install -y \
    build-essential \
    clang jq libomp-dev tree vim \
    git \
    munge \
    slurm-client \
    slurm-wlm-torque \
    sudo \
    python3 \
    python3-pip \
    wget \
    curl \
    mpich \
    libmpich-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && \
    echo "admin:admin" | chpasswd && adduser admin sudo && \
    echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY examples/tutorial/dockerfiles/slurm-cluster/reframe/docker-entrypoint.sh /usr/local/bin/

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Prepare reframe installation
WORKDIR /workspace
COPY . /workspace/reframe

WORKDIR /home/admin
USER admin
RUN uv tool install /workspace/reframe && \
    echo "export PATH=/home/admin/.local/bin:\$PATH" >> /home/admin/.profile && \
    echo "export MANPATH=/home/admin/.local/share/uv/tools/reframe-hpc/share/man:" >> /home/admin/.profile

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
