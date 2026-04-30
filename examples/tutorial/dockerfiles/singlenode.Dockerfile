FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get -y update && \
    apt-get -y install curl && \
    apt-get -y install sudo && \
    apt-get -y install clang gcc git jq libomp-dev tree vim

# Install stream
WORKDIR /usr/local/share
RUN mkdir -p stream/bin && \
    cd stream && \
    curl -fsSLJO https://www.cs.virginia.edu/stream/FTP/Code/stream.c && \
    gcc -DSTREAM_ARRAY_SIZE=100000000 -O3 -Wall -fopenmp -o bin/stream.x stream.c
ENV PATH=/usr/local/share/stream/bin:$PATH

# Prepare reframe installation
WORKDIR /workspace
COPY . /workspace/reframe

# Add tutorial user
RUN useradd -ms /bin/bash -G sudo user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/user

# Install reframe
USER user
RUN uv tool install /workspace/reframe
ENV PATH=/home/user/.local/bin:$PATH
ENV MANPATH=/home/user/.local/share/uv/tools/reframe-hpc/share/man