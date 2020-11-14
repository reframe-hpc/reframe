#
# Execute this from the top-level ReFrame source directory
#

FROM centos:7

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

# ReFrame requirements
RUN \
  yum -y install gcc && \
  yum -y install make && \
  yum -y install git && \
  yum -y install python3

# # Required utilities
# RUN apt-get -y install wget

# Install Tmod 3.2
RUN yum -y install environment-modules

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/tmod32.py -v"]
