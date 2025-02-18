FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    python3 \
    python3-pip \
    build-essential \
    libomp-dev \
    mpich \
    libmpich-dev \
    git \
    munge \
    slurm-client \
    slurmd \
    slurm-wlm-torque \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin

COPY docker-entrypoint.sh /usr/bin/

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

EXPOSE 6817 6818 6819

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
