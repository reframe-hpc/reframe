# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    munge \
    slurmctld \
    slurm-wlm-torque \
    slurmdbd \
    && rm -rf /var/lib/apt/lists/*

# Install mariadb-server separately so that `mysql` user, is created after the
# `munge` and `slurm` users created above in order to match the uid:gid pairs
# of the other containers.
RUN apt update -y && apt install -y \
    mariadb-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin

RUN mkdir /var/run/slurmdbd \
    && chown slurm:slurm /var/run/slurmdbd

COPY slurm.conf /
COPY slurmdbd.conf /
COPY cgroup.conf /
COPY docker-entrypoint.sh /usr/bin/

RUN chmod +x /usr/bin/docker-entrypoint.sh

EXPOSE 6819

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
