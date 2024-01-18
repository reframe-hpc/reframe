FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install -y \
    build-essential \
    clang jq libomp-dev tree vim \
    git \
    mariadb-client \
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

COPY slurm.conf /etc/slurm-llnl/
COPY cgroup.conf /etc/slurm-llnl/
COPY docker-entrypoint.sh /etc/slurm-llnl/
COPY container_cluster.py /reframe/container_cluster.py

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

RUN chmod +rx /etc/slurm-llnl/docker-entrypoint.sh

# Install reframe
ARG REFRAME_TAG=develop
WORKDIR /usr/local/share
RUN git clone --depth 1 --branch $REFRAME_TAG https://github.com/reframe-hpc/reframe.git && \
    cd reframe/ && ./bootstrap.sh
ENV PATH=/usr/local/share/reframe/bin:$PATH

WORKDIR /home/admin

ENV USER admin
ENV SHELL bash

ENTRYPOINT ["/etc/slurm-llnl/docker-entrypoint.sh"]
