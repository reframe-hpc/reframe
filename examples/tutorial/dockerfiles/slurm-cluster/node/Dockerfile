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
    sudo \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && \
    echo "admin:admin" | chpasswd && \
    adduser admin sudo && \
    echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY slurm.conf /etc/slurm/
COPY cgroup.conf /etc/slurm/
COPY docker-entrypoint.sh /etc/slurm/

RUN chmod +r /etc/slurm/slurm.conf
RUN chmod +r /etc/slurm/cgroup.conf
RUN chmod +rx /etc/slurm/docker-entrypoint.sh

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

EXPOSE 6817 6818 6819

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
