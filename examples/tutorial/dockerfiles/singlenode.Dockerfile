FROM ubuntu:23.04

RUN apt-get -y update && \
    apt-get -y install curl && \
    apt-get -y install sudo && \
    apt-get -y install python3-pip && \
    apt-get -y install clang gcc git jq libomp-dev tree vim

# Install reframe
ARG REFRAME_TAG=develop
WORKDIR /usr/local/share
RUN git clone --depth 1 --branch $REFRAME_TAG https://github.com/reframe-hpc/reframe.git && \
    cd reframe/ && ./bootstrap.sh
ENV PATH=/usr/local/share/reframe/bin:$PATH

# Install stream
RUN mkdir -p stream/bin && \
    cd stream && \
    curl -fsSLJO https://www.cs.virginia.edu/stream/FTP/Code/stream.c && \
    gcc -DSTREAM_ARRAY_SIZE=100000000 -O3 -Wall -fopenmp -o bin/stream.x stream.c
ENV PATH=/usr/local/share/stream/bin:$PATH

# Add tutorial user
RUN useradd -ms /bin/bash -G sudo user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY examples /home/user/reframe-examples
RUN chown -R user:user /home/user/reframe-examples
WORKDIR /home/user

USER user
