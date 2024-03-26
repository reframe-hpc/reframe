FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    build-essential \
    git \
    munge \
    slurmctld \
    slurm-wlm-torque \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && \
    echo "admin:admin" | chpasswd && \
    adduser admin sudo && \
    echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY slurm.conf /etc/slurm-llnl/
COPY cgroup.conf /etc/slurm-llnl/
COPY docker-entrypoint.sh /etc/slurm-llnl/

RUN chmod +rx /etc/slurm-llnl/docker-entrypoint.sh

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

EXPOSE 6817 6818 6819 3306

WORKDIR /home/admin

ENTRYPOINT ["/etc/slurm-llnl/docker-entrypoint.sh"]
