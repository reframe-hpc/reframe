FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    munge \
    slurmctld \
    slurm-wlm-torque \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin

COPY docker-entrypoint.sh /usr/bin/docker-entrypoint.sh

RUN chmod +rx /usr/bin/docker-entrypoint.sh

EXPOSE 6817 6818 6819

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
