FROM ubuntu:20.04

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

COPY slurm.conf /etc/slurm-llnl/
COPY cgroup.conf /etc/slurm-llnl/
COPY docker-entrypoint.sh /etc/slurm-llnl/

RUN chmod +r /etc/slurm-llnl/slurm.conf
RUN chmod +r /etc/slurm-llnl/cgroup.conf
RUN chmod +rx /etc/slurm-llnl/docker-entrypoint.sh

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

EXPOSE 6817 6818 6819

ENTRYPOINT ["/etc/slurm-llnl/docker-entrypoint.sh"]
