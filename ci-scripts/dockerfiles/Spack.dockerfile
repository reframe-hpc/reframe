#
# Execute this from the top-level ReFrame source directory
#

FROM ubuntu:20.04

ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV _SPACK_VER=0.16

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

# ReFrame requirements
RUN \
  apt-get -y update && \
  apt-get -y install ca-certificates && \
  update-ca-certificates && \
  apt-get -y install gcc && \
  apt-get -y install make && \
  apt-get -y install git && \
  apt-get -y install python3 python3-pip python3-venv

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

USER rfmuser

# Install Spack
RUN git clone https://github.com/spack/spack ~/spack && \
    cd ~/spack && \
    git checkout releases/v${_SPACK_VER}

ENV BASH_ENV /home/rfmuser/spack/share/spack/setup-env.sh

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/spack.py -v"]
