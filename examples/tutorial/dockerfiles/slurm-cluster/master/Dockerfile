FROM ubuntu:22.04

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

COPY slurm.conf /etc/slurm/
COPY cgroup.conf /etc/slurm/
COPY docker-entrypoint.sh /etc/slurm/

RUN chmod +rx /etc/slurm/docker-entrypoint.sh

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

EXPOSE 6817 6818 6819

WORKDIR /home/admin

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
