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
  apt-get -y install ca-certificates && \
  update-ca-certificates && \
  apt-get -y install gcc && \
  apt-get -y install make && \
  apt-get -y install git && \
  apt-get -y install python3 python3-pip

# Required utilities
RUN apt-get -y install wget

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

USER rfmuser

# Install Spack
RUN git clone https://github.com/spack/spack ~/spack && \
    cd ~/spack && \
    git checkout releases/v0.16 && \
    echo 'source ~/spack/share/spack/setup-env.sh' >> ~/.bashrc

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/spack.py -v"]
