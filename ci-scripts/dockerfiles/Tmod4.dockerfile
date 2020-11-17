#
# Execute this from the top-level ReFrame source directory
#

FROM ubuntu:20.04

ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV _TMOD_VER=4.6.0

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

# ReFrame requirements
RUN \
  apt-get -y update && \
  apt-get -y install gcc && \
  apt-get -y install make && \
  apt-get -y install git && \
  apt-get -y install python3 python3-pip

# Required utilities
RUN apt-get -y install wget

# Install Tmod4
RUN \
  apt-get -y install autoconf && \
  apt-get -y install tcl-dev && \
  wget -q https://github.com/cea-hpc/modules/archive/v${_TMOD_VER}.tar.gz -O tmod.tar.gz && \
  tar xzf tmod.tar.gz && \
  cd modules-${_TMOD_VER} && \
  ./configure && make install

ENV BASH_ENV=/usr/local/Modules/init/profile.sh

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/tmod4.py -v"]
