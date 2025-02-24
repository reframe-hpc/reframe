FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    munge \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && \
    echo "admin:admin" | chpasswd && \
    adduser admin sudo && \
    echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

COPY entrypoint.sh /etc/munge/

RUN chmod +rx /etc/munge/entrypoint.sh

ENTRYPOINT ["/etc/munge/entrypoint.sh"]
